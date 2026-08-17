"""
Coordinator for distributing captioning work across multiple machines (e.g. a computer lab),
each running its own local ollama (or other) LLM.

Run this ONCE, on any one machine that every worker machine can reach over the network
(same LAN is enough -- it just needs to be a reachable IP:port):

    python caption_coordinator.py --levels ../TheVGLC/MegaMan/Enhanced --output captions.json

Then, on every lab machine (including optionally this one), run:

    python caption_worker.py --coordinator http://<coordinator-ip>:8765 --llm ollama

Each worker repeatedly asks the coordinator for a small batch of scenes, captions them with
its own local LLM, and posts the captions back. The coordinator writes every finished scene
straight to a checkpoint (crash-safe, same format and naming convention llm_ascii_to_caption.py
uses -- --output with ".json" swapped for ".jsonl") and assembles --output automatically once
everything is done -- no manual merge step, no shared filesystem required. If a worker dies
mid-batch, its unfinished scenes are automatically handed back out to another worker after
--lease-seconds. If the coordinator itself is restarted and finds a leftover checkpoint from a
previous run, it will ask whether to resume from it, same as llm_ascii_to_caption.py
(--force-resume / --force-restart skip that prompt).

This only uses the Python standard library (http.server / urllib), so no new dependencies.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from llm_ascii_to_caption import (
    load_dataset, scene_to_ASCII, filter_tile_set, deterministic_caption,
    CheckpointWriter, load_checkpoint, default_checkpoint_path, resolve_resume, finalize_output,
)
from captions.util import extract_tileset
from util.descriptive_tilesets import GAMES


class WorkPool:
    """Thread-safe pending/assigned/done state for one captioning run."""

    def __init__(self, scene_data: dict, num_captions: int, lease_seconds: int):
        self.scene_data = scene_data  # {index: {label, scene_str, tileset, deterministic, ...}}
        self.num_captions = num_captions
        self.lease_seconds = lease_seconds
        self.lock = threading.Lock()
        self.pending = list(scene_data.keys())
        self.assigned = {}  # index -> (worker_id, claimed_at)
        self.done = set()
        self.skipped = set()

    def _reclaim_expired(self):
        now = time.time()
        expired = [i for i, (_, t) in self.assigned.items() if now - t > self.lease_seconds]
        for i in expired:
            del self.assigned[i]
            self.pending.append(i)

    def claim(self, worker_id: str, n: int) -> list[int]:
        with self.lock:
            self._reclaim_expired()
            claimed = self.pending[:n]
            self.pending = self.pending[n:]
            now = time.time()
            for i in claimed:
                self.assigned[i] = (worker_id, now)
            return claimed

    def mark_done(self, index: int):
        with self.lock:
            self.assigned.pop(index, None)
            self.done.add(index)

    def mark_skipped(self, index: int):
        with self.lock:
            self.assigned.pop(index, None)
            self.skipped.add(index)

    def status(self) -> dict:
        with self.lock:
            total = len(self.scene_data)
            remaining = total - len(self.done) - len(self.skipped)
            return {
                "total": total,
                "pending": len(self.pending),
                "assigned": len(self.assigned),
                "done": len(self.done),
                "skipped": len(self.skipped),
                "remaining": remaining,
                "finished": remaining == 0,
            }


def make_handler(pool: WorkPool, writer: CheckpointWriter, args, finalize_cb):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # keep stdout clean; use /status to monitor instead

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
                n = int(qs.get("n", [str(args.batch_size)])[0])
                indices = pool.claim(worker_id, n)
                items = []
                for i in indices:
                    d = pool.scene_data[i]
                    items.append({
                        "index": i,
                        "label": d["label"],
                        "scene_str": d["scene_str"],
                        "tileset": d["tileset"],
                        "deterministic": d["deterministic"],
                        "game_name": d["game_name"],
                        "num_captions": pool.num_captions,
                    })
                st = pool.status()
                self._send_json({"work": items, "finished": st["finished"], "status": st})

            elif parsed.path == "/status":
                self._send_json(pool.status())

            else:
                self._send_json({"error": "not found"}, code=404)

        def do_POST(self):
            if self.path != "/result":
                self._send_json({"error": "not found"}, code=404)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            index = body["index"]
            worker_id = body.get("worker", "unknown")
            captions = body.get("captions", [])

            d = pool.scene_data.get(index)
            if d is None:
                self._send_json({"ok": False, "error": "unknown index"}, code=400)
                return

            if len(captions) != pool.num_captions:
                print(f"[skip] {d['label']} (index {index}) from worker {worker_id}: "
                      f"got {len(captions)} caption(s), expected {pool.num_captions}")
                pool.mark_skipped(index)
                self._send_json({"ok": True, "skipped": True})
            else:
                entry = dict(d["attrs"])
                entry["scene"] = d["scene"]
                if args.caption_mode == "legacy":
                    entry["caption"] = captions[0]
                    for idx, caption in enumerate(captions[1:], start=1):
                        entry[f"caption{idx}"] = caption
                    entry["model"] = f"{args.expected_llm} - {args.expected_model} (worker: {worker_id})"
                else:
                    entry[args.caption_key] = captions
                writer.write(index, entry)
                pool.mark_done(index)
                self._send_json({"ok": True})

            if pool.status()["finished"]:
                finalize_cb()

        def do_ANY(self):  # unused, kept for clarity that other methods are unsupported
            self._send_json({"error": "method not allowed"}, code=405)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="../TheVGLC/MegaMan/Enhanced")
    ap.add_argument("--game", default="MM-Full", choices=list(GAMES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_captions", type=int, default=5)
    ap.add_argument("--output", default="captions.json")
    # Checkpointing is always on and its path is never user-chosen: it's --output with ".json"
    # swapped for ".jsonl", exactly matching llm_ascii_to_caption.py's own convention.
    resume_group = ap.add_mutually_exclusive_group()
    resume_group.add_argument("--force-resume", action="store_true",
                     help="If the checkpoint .jsonl already exists, resume from it automatically (skip scenes "
                          "already recorded there) without prompting.")
    resume_group.add_argument("--force-restart", action="store_true",
                     help="If the checkpoint .jsonl already exists, delete it and start completely fresh, "
                          "without prompting.")
    ap.add_argument("--host", default="0.0.0.0", help="Address to bind the coordinator's HTTP server to.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--batch-size", type=int, default=3, help="Scenes handed to a worker per /work request.")
    ap.add_argument("--lease-seconds", type=int, default=600,
                     help="If a worker doesn't report back on a claimed scene within this many seconds, "
                          "it's automatically handed to another worker.")
    ap.add_argument("--caption-mode", default="keyed", choices=["legacy", "keyed"])
    ap.add_argument("--caption-key", default="coordinated_captions",
                     help="Key to store captions under in keyed mode. Since multiple workers may run "
                          "different models, this defaults to a fixed key rather than '<model>_captions'.")
    # Only used to label legacy-mode entries; the coordinator itself never calls an LLM.
    ap.add_argument("--expected_llm", default="mixed")
    ap.add_argument("--expected_model", default="mixed")
    args = ap.parse_args()

    game = GAMES[args.game]
    game_name = game["name"]
    tile_names = game["tiles"]["tiles"]
    tileset_path = game["tileset"]
    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)
    null_ids = frozenset(tid for tid, ch in id_to_char.items() if "null" in tile_descriptors.get(ch, set()))

    scenes = load_dataset(args.levels, char_to_id)[:args.limit]
    print(f"[coordinator] Preparing prompt data for {len(scenes)} scene(s)...")

    scene_data = {}
    for i, (scene, label, attrs) in enumerate(scenes):
        scene_str = "\n".join(scene_to_ASCII(scene, id_to_char, null_ids))
        scene_data[i] = {
            "label": label,
            "scene": scene,
            "attrs": attrs,
            "scene_str": scene_str,
            "tileset": filter_tile_set(scene_str, tile_names),
            "deterministic": deterministic_caption(scene, id_to_char, char_to_id, tile_descriptors, names=tile_names),
            "game_name": game_name,
        }

    checkpoint_path = default_checkpoint_path(args.output, 0, 1)

    # If a checkpoint from a previous coordinator run is already sitting there, decide whether
    # to resume from it or wipe it and start over -- same prompt/force-flag behavior as
    # llm_ascii_to_caption.py, so a leftover checkpoint is never silently reused or discarded.
    resume = resolve_resume(checkpoint_path, args.force_resume, args.force_restart)

    already_done = load_checkpoint(checkpoint_path) if resume else {}
    if already_done:
        print(f"[resume] {len(already_done)} scene(s) already captioned in {checkpoint_path}; skipping those.")
        for i in already_done:
            scene_data.pop(i, None)

    writer = CheckpointWriter(checkpoint_path, resume=resume)
    pool = WorkPool(scene_data, args.num_captions, args.lease_seconds)

    finalized = threading.Event()

    def finalize_cb():
        if finalized.is_set():
            return
        finalized.set()
        writer.close()
        finalize_output(checkpoint_path, args.output)
        print("[coordinator] All scenes done. You can Ctrl+C the coordinator now.")

    handler = make_handler(pool, writer, args, finalize_cb)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"[coordinator] Serving {len(scene_data)} scene(s) to caption on {args.host}:{args.port}")
    print(f"[coordinator] Checkpoint: {checkpoint_path}  |  Final output: {args.output}")
    print("[coordinator] Point workers at this machine with: "
          f"python caption_worker.py --coordinator http://<this-machine-ip>:{args.port} --llm ollama\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Whatever finished before Ctrl+C is still safely in the checkpoint; finalize once more
        # so --output reflects it even if the run was stopped early.
        if not finalized.is_set():
            writer.close()
            finalize_output(checkpoint_path, args.output)


if __name__ == "__main__":
    main()
