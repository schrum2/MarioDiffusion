"""
Coordinator for distributing captioning work across multiple machines.

The coordinator knows which LLM/model configurations must be completed. Workers advertise
their own LLM/model when requesting work, so different workers can use different models and
the same scene can be assigned once to every required model.

Example:

    python caption_coordinator.py \
        --levels ../TheVGLC/MegaMan/Enhanced \
        --output captions.json \
        --model ollama:qwen3.5:9b \
        --model ollama:gemma4:12b

Then run workers such as:

    python caption_worker.py --coordinator http://<coordinator-ip>:8765 \
        --llm ollama --model qwen3.5:9b

and:

    python caption_worker.py --coordinator http://<coordinator-ip>:8765 \
        --llm ollama --model gemma4:12b

A worker may be run on multiple machines. All completed (scene, model) pairs are written
immediately to the checkpoint, so worker or coordinator crashes do not lose completed work.
"""

import argparse
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

from llm_ascii_to_caption import (
    DEFAULT_MODELS,
    CheckpointWriter,
    default_checkpoint_path,
    deterministic_caption,
    filter_tile_set,
    load_checkpoint,
    load_dataset,
    resolve_resume,
    scene_to_ASCII,
)
from captions.util import extract_tileset
from util.descriptive_tilesets import GAMES


def get_local_ip() -> str:
    """Return the local address that this machine uses for outbound LAN traffic."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packets need to be sent. connect() lets the OS select the normal route/interface.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def parse_model_spec(spec: str) -> dict:
    """
    Parse --model values of the form LLM:MODEL.

    The split is only on the first colon because Ollama model names can themselves contain
    colons, e.g. qwen2.5:9b.
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid --model '{spec}'. Use LLM:MODEL, e.g. ollama:qwen2.5:9b."
        )
    llm, model = spec.split(":", 1)
    if llm not in DEFAULT_MODELS:
        raise ValueError(
            f"Invalid LLM '{llm}' in --model '{spec}'. "
            f"Choose from: {', '.join(DEFAULT_MODELS)}"
        )
    if not model:
        raise ValueError(f"Invalid --model '{spec}': model name is empty.")

    # The model name is deliberately the output key because llm_ascii_to_caption.py
    # uses '<model>_captions' as its keyed-mode convention.
    return {
        "llm": llm,
        "model": model,
        "caption_key": f"{model}_captions",
    }


def format_model(spec: dict) -> str:
    return f"{spec['llm']}:{spec['model']}"


class WorkPool:
    """Thread-safe pool of (scene, model) jobs."""

    def __init__(self, scene_data: dict, model_specs: list[dict], num_captions: int,
                 lease_seconds: int, completed: set[tuple[int, str]]):
        self.scene_data = scene_data
        self.model_specs = {s["caption_key"]: s for s in model_specs}
        self.num_captions = num_captions
        self.lease_seconds = lease_seconds
        self.lock = threading.Lock()

        self.pending = []
        self.assigned = {}  # (index, caption_key) -> (worker_id, claimed_at)
        self.done = set(completed)
        self.workers = {}   # worker_id -> last request time

        for index in scene_data:
            for key in self.model_specs:
                job = (index, key)
                if job not in self.done:
                    self.pending.append(job)

    def _reclaim_expired(self):
        now = time.time()
        expired = [
            job for job, (_, claimed_at) in self.assigned.items()
            if now - claimed_at > self.lease_seconds
        ]
        for job in expired:
            worker_id, _ = self.assigned.pop(job)
            self.pending.append(job)
            print(
                f"[lease] Reclaimed scene {job[0]} for {job[1]} "
                f"from worker {worker_id}; lease expired."
            )

    def claim(self, worker_id: str, llm: str, model: str, n: int) -> list[tuple[int, str]]:
        caption_key = f"{model}_captions"
        with self.lock:
            self._reclaim_expired()
            self.workers[worker_id] = time.time()

            spec = self.model_specs.get(caption_key)
            if spec is None or spec["llm"] != llm:
                return []

            claimed = []
            remaining = []
            now = time.time()

            for job in self.pending:
                if len(claimed) >= n:
                    remaining.append(job)
                    continue
                if job[1] == caption_key:
                    claimed.append(job)
                    self.assigned[job] = (worker_id, now)
                else:
                    remaining.append(job)

            self.pending = remaining
            return claimed

    def complete(self, job: tuple[int, str]):
        with self.lock:
            self.assigned.pop(job, None)
            self.done.add(job)

    def release(self, job: tuple[int, str]):
        with self.lock:
            self.assigned.pop(job, None)
            if job not in self.done and job not in self.pending:
                self.pending.append(job)

    def is_done(self, job: tuple[int, str]) -> bool:
        with self.lock:
            return job in self.done

    def expected_job_count(self) -> int:
        return len(self.scene_data) * len(self.model_specs)

    def status(self) -> dict:
        with self.lock:
            self._reclaim_expired()
            total = self.expected_job_count()
            done = len(self.done)
            pending = len(self.pending)
            assigned = len(self.assigned)
            by_model = {}
            for key in self.model_specs:
                model_done = sum(1 for i, k in self.done if k == key)
                model_total = len(self.scene_data)
                by_model[key] = {
                    "done": model_done,
                    "total": model_total,
                    "remaining": model_total - model_done,
                }
            return {
                "total_jobs": total,
                "pending": pending,
                "assigned": assigned,
                "done": done,
                "remaining": total - done,
                "finished": done == total,
                "by_model": by_model,
                "workers": len(self.workers),
            }


def load_task_checkpoint(path: str, model_specs: list[dict]) -> dict[tuple[int, str], list[str]]:
    """Load the multi-model task checkpoint without collapsing records by scene index."""
    results = {}
    if not os.path.exists(path):
        return results

    legacy_key = model_specs[0]["caption_key"] if len(model_specs) == 1 else None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave a partial final JSONL line. Ignore only that line.
                continue

            try:
                index = int(entry["_index"])
            except (KeyError, TypeError, ValueError):
                continue

            key = entry.get("_model_key")
            captions = entry.get("_captions")

            if key is not None and isinstance(captions, list):
                results[(index, key)] = captions
                continue

            # Compatibility with the previous single-model coordinator, whose records
            # stored a whole output entry under _index and used coordinated_captions.
            if legacy_key is not None:
                legacy_captions = entry.get("coordinated_captions")
                if isinstance(legacy_captions, list):
                    results[(index, legacy_key)] = legacy_captions

    return results


def precheck_dataset(scene_data: dict, model_specs: list[dict], num_captions: int,
                     checkpoint_results: dict[tuple[int, str], list[str]]) -> set[tuple[int, str]]:
    """Validate input structure and report exactly what still needs to be generated."""
    print("[pre-check] Validating every input sample...")

    invalid = []
    for index, data in scene_data.items():
        scene = data["scene"]
        if not isinstance(scene, list) or not scene:
            invalid.append(index)
            continue
        if not all(isinstance(row, list) for row in scene):
            invalid.append(index)

    if invalid:
        raise ValueError(
            f"[pre-check] {len(invalid)} input sample(s) do not contain a valid 2D scene. "
            f"First invalid index: {invalid[0]}"
        )

    total_samples = len(scene_data)
    total_jobs = total_samples * len(model_specs)
    completed = set()

    for index, data in scene_data.items():
        attrs = data["attrs"]
        for spec in model_specs:
            key = spec["caption_key"]
            job = (index, key)

            checkpoint_captions = checkpoint_results.get(job)
            if isinstance(checkpoint_captions, list) and len(checkpoint_captions) == num_captions:
                completed.add(job)
                continue

            input_captions = attrs.get(key)
            if isinstance(input_captions, list) and len(input_captions) == num_captions:
                completed.add(job)

    print(f"[pre-check] Input samples: {total_samples:,}")
    print(f"[pre-check] Required caption sets: {total_jobs:,}")

    for spec in model_specs:
        key = spec["caption_key"]
        already = sum(1 for i in scene_data if (i, key) in completed)
        print(
            f"[pre-check] {format_model(spec)} -> {key}: "
            f"{already:,}/{total_samples:,} already complete; "
            f"{total_samples - already:,} still required."
        )

    print(
        f"[pre-check] Complete caption sets already available: {len(completed):,}; "
        f"jobs remaining: {total_jobs - len(completed):,}."
    )

    return completed


def build_final_output(scene_data: dict, model_specs: list[dict],
                       checkpoint_results: dict[tuple[int, str], list[str]],
                       num_captions: int, output_path: str) -> tuple[bool, list[dict]]:
    """Merge all model-specific checkpoint results back onto every input sample."""
    output = []
    missing = []

    for index in sorted(scene_data):
        data = scene_data[index]
        entry = dict(data["attrs"])
        entry["scene"] = data["scene"]

        for spec in model_specs:
            key = spec["caption_key"]
            captions = checkpoint_results.get((index, key))

            # An already-captioned input dataset can be used as a starting point.
            if captions is None:
                existing = data["attrs"].get(key)
                if isinstance(existing, list):
                    captions = existing

            if not isinstance(captions, list) or len(captions) != num_captions:
                missing.append((index, key))
            else:
                entry[key] = captions

        output.append(entry)

    complete = not missing and len(output) == len(scene_data)

    if complete:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(
            f"[coordinator] Final verification PASSED: {len(output):,} samples, "
            f"{len(model_specs)} model(s), {len(output) * len(model_specs):,} caption sets."
        )
        print(f"[coordinator] Final output saved to {output_path}")
    else:
        print(
            f"[coordinator] Final verification FAILED: {len(missing):,} "
            f"(sample, model) caption sets are missing or incomplete."
        )
        if missing:
            print(f"[coordinator] First missing job: scene {missing[0][0]}, {missing[0][1]}")

    return complete, output


def make_handler(pool: WorkPool, writer: CheckpointWriter, args, model_specs,
                 checkpoint_results: dict, scene_data: dict, server_holder: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            # We print our own useful progress messages below.
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)

            if parsed.path == "/work":
                worker_id = qs.get("worker", ["unknown"])[0]
                llm = qs.get("llm", [""])[0]
                model = qs.get("model", [""])[0]
                n = max(1, int(qs.get("n", [str(args.batch_size)])[0]))

                if server_holder["stopping"]:
                    self._send_json({
                        "work": [],
                        "shutdown": True,
                        "message": "Coordinator has completed all required captioning jobs. Exit now.",
                        "status": pool.status(),
                    })
                    return

                expected = pool.model_specs.get(f"{model}_captions")
                if expected is None or expected["llm"] != llm:
                    self._send_json({
                        "work": [],
                        "shutdown": False,
                        "error": (
                            f"This coordinator is not configured for {llm}:{model}. "
                            "Start a worker with one of the configured LLM/model pairs."
                        ),
                    }, code=400)
                    return

                jobs = pool.claim(worker_id, llm, model, n)
                items = []

                for index, key in jobs:
                    d = scene_data[index]
                    items.append({
                        "index": index,
                        "label": d["label"],
                        "scene_str": d["scene_str"],
                        "tileset": d["tileset"],
                        "deterministic": d["deterministic"],
                        "game_name": d["game_name"],
                        "prompt_vocab": d["prompt_vocab"],
                        "prompt_rules": d["prompt_rules"],
                        "num_captions": pool.num_captions,
                        "llm": llm,
                        "model": model,
                        "caption_key": key,
                    })

                status = pool.status()

                if items:
                    print(
                        f"[work] {worker_id} claimed {len(items)} {model} job(s); "
                        f"progress {status['done']:,}/{status['total_jobs']:,} "
                        f"jobs complete, {status['assigned']:,} assigned."
                    )
                elif status["finished"]:
                    self._send_json({
                        "work": [],
                        "shutdown": True,
                        "message": "All required captions are complete. Coordinator is shutting down.",
                        "status": status,
                    })
                    threading.Thread(
                        target=server_holder["finish"],
                        name="coordinator-shutdown",
                        daemon=True,
                    ).start()
                    return

                self._send_json({
                    "work": items,
                    "shutdown": False,
                    "status": status,
                })

            elif parsed.path == "/status":
                self._send_json(pool.status())
            else:
                self._send_json({"error": "not found"}, code=404)

        def do_POST(self):
            if self.path == "/result":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")

                try:
                    index = int(body["index"])
                    llm = body["llm"]
                    model = body["model"]
                    worker_id = body.get("worker", "unknown")
                    captions = body.get("captions", [])
                except (KeyError, TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": f"Invalid result payload: {exc}"}, code=400)
                    return

                key = f"{model}_captions"
                job = (index, key)

                if index not in scene_data:
                    self._send_json({"ok": False, "error": "unknown scene index"}, code=400)
                    return

                spec = pool.model_specs.get(key)
                if spec is None or spec["llm"] != llm:
                    self._send_json({
                        "ok": False,
                        "error": f"Unexpected LLM/model pair: {llm}:{model}",
                    }, code=400)
                    return

                if pool.is_done(job):
                    # Idempotence matters if the coordinator restarted after a worker
                    # generated a result but before the worker received the HTTP response.
                    self._send_json({
                        "ok": True,
                        "already_complete": True,
                        "message": f"{key} for scene {index} was already recorded.",
                        "shutdown": pool.status()["finished"],
                    })
                    return

                if len(captions) != pool.num_captions:
                    pool.release(job)
                    print(
                        f"[retry] {worker_id} returned {len(captions)} caption(s) for "
                        f"scene {index}, {key}; expected {pool.num_captions}. Job released."
                    )
                    self._send_json({
                        "ok": False,
                        "retry": True,
                        "error": f"Expected {pool.num_captions} captions, got {len(captions)}.",
                    })
                    return

                # The checkpoint is a task record, not a whole output sample. This is what
                # allows several model results for the same scene to coexist safely.
                record = {
                    "_index": index,
                    "_model_key": key,
                    "_llm": llm,
                    "_model": model,
                    "_captions": captions,
                }
                writer.write(index, record)
                checkpoint_results[job] = captions
                pool.complete(job)

                status = pool.status()
                print(
                    f"[result] {worker_id} completed scene {index} with {key}; "
                    f"progress {status['done']:,}/{status['total_jobs']:,} jobs complete "
                    f"({status['remaining']:,} remaining)."
                )

                shutdown = status["finished"]
                self._send_json({
                    "ok": True,
                    "shutdown": shutdown,
                    "message": (
                        "All required captions are complete. Exit after this response."
                        if shutdown else "Result recorded."
                    ),
                    "status": status,
                })

                if shutdown:
                    threading.Thread(
                        target=server_holder["finish"],
                        name="coordinator-shutdown",
                        daemon=True,
                    ).start()
                return

            if self.path == "/shutdown-ack":
                self._send_json({"ok": True})
                return

            self._send_json({"error": "not found"}, code=404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="../TheVGLC/MegaMan/Enhanced")
    ap.add_argument("--game", default="MM-Full", choices=list(GAMES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_captions", type=int, default=5)
    ap.add_argument("--output", default="captions.json")

    resume_group = ap.add_mutually_exclusive_group()
    resume_group.add_argument("--force-resume", action="store_true",
                              help="Resume an existing .jsonl checkpoint without prompting.")
    resume_group.add_argument("--force-restart", action="store_true",
                              help="Delete an existing .jsonl checkpoint and start fresh.")

    ap.add_argument("--host", default="0.0.0.0",
                    help="Address to bind the coordinator HTTP server to.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch-size", type=int, default=3,
                    help="Scenes handed to a worker per request.")
    ap.add_argument("--shutdown-grace-seconds", type=float, default=15.0,
                    help="Seconds to keep the server available after completion so idle workers can receive the shutdown message.")
    ap.add_argument("--lease-seconds", type=int, default=600,
                    help="Seconds before an uncompleted job is automatically reassigned.")

    ap.add_argument(
        "--model", action="append", required=True,
        help=(
            "Required LLM/model pair in the form LLM:MODEL. Repeat this option for every "
            "model that every input scene must receive captions from. Example: "
            "--model ollama:qwen2.5:9b --model ollama:gemma4:12b"
        ),
    )

    args = ap.parse_args()

    if args.num_captions < 1:
        ap.error("--num_captions must be at least 1")

    try:
        model_specs = [parse_model_spec(s) for s in args.model]
    except ValueError as exc:
        ap.error(str(exc))

    keys = [s["caption_key"] for s in model_specs]
    if len(keys) != len(set(keys)):
        ap.error("Each --model must have a unique model name because it becomes the caption field name.")

    game = GAMES[args.game]
    game_name = game["name"]
    tile_names = game["tiles"]["tiles"]
    tileset_path = game["tileset"]
    prompt_vocab = game.get("prompt_vocab", [])
    prompt_rules = game.get("prompt_rules", [])

    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)
    null_ids = frozenset(
        tid for tid, ch in id_to_char.items()
        if "null" in tile_descriptors.get(ch, set())
    )

    scenes = load_dataset(args.levels, char_to_id)[:args.limit]
    print(f"[coordinator] Preparing prompt data for {len(scenes):,} input sample(s)...")

    scene_data = {}
    for i, (scene, label, attrs) in enumerate(scenes):
        scene_str = "\n".join(scene_to_ASCII(scene, id_to_char, null_ids))
        scene_data[i] = {
            "label": label,
            "scene": scene,
            "attrs": attrs,
            "scene_str": scene_str,
            "tileset": filter_tile_set(scene_str, tile_names),
            "deterministic": deterministic_caption(
                scene, id_to_char, char_to_id, tile_descriptors, names=tile_names
            ),
            "game_name": game_name,
            "prompt_vocab": prompt_vocab,
            "prompt_rules": prompt_rules,
        }

    checkpoint_path = default_checkpoint_path(args.output, 0, 1)
    resume = resolve_resume(checkpoint_path, args.force_resume, args.force_restart)
    checkpoint_results = load_task_checkpoint(checkpoint_path, model_specs) if resume else {}

    if resume:
        print(
            f"[resume] Loaded {len(checkpoint_results):,} completed "
            f"(scene, model) result(s) from {checkpoint_path}."
        )

    completed = precheck_dataset(
        scene_data, model_specs, args.num_captions, checkpoint_results
    )

    # A checkpoint result supersedes an input copy of the same model's captions.
    for job, captions in checkpoint_results.items():
        if len(captions) == args.num_captions:
            completed.add(job)

    writer = CheckpointWriter(checkpoint_path, resume=resume)

    pool = WorkPool(
        scene_data,
        model_specs,
        args.num_captions,
        args.lease_seconds,
        completed,
    )

    local_ip = get_local_ip()
    server_holder = {"stopping": False, "finish": None}

    finalized = threading.Event()

    def finish():
        if finalized.is_set():
            return

        finalized.set()
        print("\n[coordinator] All required caption jobs have been received.")
        print("[coordinator] Performing final verification of every sample/model pair...")

        # Reload the checkpoint so verification is based on what is actually durable on disk.
        durable_results = load_task_checkpoint(checkpoint_path, model_specs)
        complete, _ = build_final_output(
            scene_data,
            model_specs,
            durable_results,
            args.num_captions,
            args.output,
        )

        if not complete:
            finalized.clear()
            print(
                "[coordinator] WARNING: verification found missing data. "
                "The coordinator will remain running so the missing jobs can be retried."
            )
            return

        server_holder["stopping"] = True
        print("[coordinator] Verification passed.")
        print("[coordinator] Sending shutdown message to workers.")
        print("[coordinator] Coordinator will exit automatically.")

        def stop_server():
            time.sleep(args.shutdown_grace_seconds)
            server_holder["httpd"].shutdown()

        threading.Thread(target=stop_server, name="coordinator-stop", daemon=True).start()

    server_holder["finish"] = finish

    handler = make_handler(
        pool, writer, args, model_specs, checkpoint_results, scene_data, server_holder
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server_holder["httpd"] = server

    print()
    print("============================================================")
    print("Caption coordinator")
    print("============================================================")
    print(f"[coordinator] LLM/model targets:")
    for spec in model_specs:
        print(f"[coordinator]   {format_model(spec)} -> {spec['caption_key']}")
    print(f"[coordinator] Number of captions per model: {args.num_captions}")
    print(f"[coordinator] Input samples: {len(scene_data):,}")
    print(f"[coordinator] Checkpoint: {checkpoint_path}")
    print(f"[coordinator] Final output: {args.output}")
    print(f"[coordinator] Actual coordinator IP: {local_ip}")
    print(f"[coordinator] Listening on: http://{local_ip}:{args.port}")
    print(
        "[coordinator] Workers should use, for example: "
        f"python caption_worker.py --coordinator http://{local_ip}:{args.port} "
        f"--llm {model_specs[0]['llm']} --model {model_specs[0]['model']}"
    )
    print()

    initial_status = pool.status()
    print(
        f"[coordinator] Work remaining: {initial_status['remaining']:,} "
        f"(pending {initial_status['pending']:,}, assigned {initial_status['assigned']:,})."
    )

    if initial_status["finished"]:
        # This also performs final verification and causes a clean shutdown.
        finish()
    else:
        print("[coordinator] Waiting for workers...\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[coordinator] Interrupted by Ctrl-C. Finalizing durable checkpoint.")
    finally:
        if not finalized.is_set():
            writer.close()
            durable_results = load_task_checkpoint(checkpoint_path, model_specs)
            build_final_output(
                scene_data,
                model_specs,
                durable_results,
                args.num_captions,
                args.output,
            )
        else:
            writer.close()

        server.server_close()
        print("[coordinator] Exited cleanly.")


if __name__ == "__main__":
    main()