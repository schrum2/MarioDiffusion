"""
Worker for the distributed captioning coordinator.

A worker advertises its LLM/model to the coordinator. Multiple workers can use the same
model, while other workers use different models. The coordinator only gives a worker jobs
for the model it was started with.

Example:

    python caption_worker.py --coordinator http://192.168.1.20:8765 \
        --llm ollama --model qwen2.5:9b

Workers exit cleanly when the coordinator sends the shutdown message after every required
(scene, model) pair has been completed.
"""

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from llm_ascii_to_caption import DEFAULT_MODELS, llm_caption


def get_work(coordinator: str, worker_id: str, llm: str, model: str, n: int) -> dict:
    query = urllib.parse.urlencode({
        "worker": worker_id,
        "llm": llm,
        "model": model,
        "n": n,
    })
    with urllib.request.urlopen(f"{coordinator}/work?{query}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_result(coordinator: str, worker_id: str, llm: str, model: str,
                index: int, captions: list[str], retry_interval: float) -> dict:
    """
    Keep trying to deliver a completed caption set.

    This is important if the coordinator crashes immediately after a worker finishes
    inference. The coordinator's /result endpoint is idempotent, so resending a result
    that was already recorded is safe.
    """
    payload = json.dumps({
        "index": index,
        "worker": worker_id,
        "llm": llm,
        "model": model,
        "captions": captions,
    }).encode("utf-8")

    while True:
        req = urllib.request.Request(
            f"{coordinator}/result",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                details = json.loads(exc.read().decode("utf-8"))
            except Exception:
                details = {"error": str(exc)}
            # A normal coordinator rejection (for example a bad caption count) is a
            # response, not a network failure. Return it so the caller can react.
            return details
        except (urllib.error.URLError, TimeoutError) as exc:
            print(
                f"[worker {worker_id}] Couldn't deliver result for scene {index} "
                f"({exc}); retrying in {retry_interval}s..."
            )
            time.sleep(retry_interval)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--coordinator", required=True,
        help="Base URL of the coordinator, e.g. http://192.168.1.20:8765"
    )
    ap.add_argument("--llm", choices=list(DEFAULT_MODELS), default="ollama")
    ap.add_argument(
        "--model", default=None,
        help="Model used by this worker. Defaults to the model for --llm in llm_ascii_to_caption.py."
    )
    ap.add_argument("--worker-id", default=None, help="Defaults to '<hostname>-<pid>'")
    ap.add_argument("--batch-size", type=int, default=3, help="Scenes requested per poll.")
    ap.add_argument(
        "--poll-interval", type=float, default=5.0,
        help="Seconds to wait when no compatible work is available."
    )
    args = ap.parse_args()

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    model = args.model or DEFAULT_MODELS[args.llm]
    coordinator = args.coordinator.rstrip("/")

    print(
        f"[worker {worker_id}] Using --llm {args.llm} --model {model}, "
        f"polling {coordinator}"
    )
    print(
        f"[worker {worker_id}] Caption field will be '{model}_captions'.\n"
    )

    done_count = 0

    while True:
        try:
            resp = get_work(
                coordinator, worker_id, args.llm, model, args.batch_size
            )
        except urllib.error.HTTPError as exc:
            try:
                details = exc.read().decode("utf-8")
            except Exception:
                details = str(exc)

            # A configuration error should not become an infinite retry loop.
            print(f"[worker {worker_id}] Coordinator rejected this worker: {details}")
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            print(
                f"[worker {worker_id}] Couldn't reach coordinator ({exc}); "
                f"retrying in {args.poll_interval}s..."
            )
            time.sleep(args.poll_interval)
            continue

        if resp.get("shutdown"):
            print(
                f"[worker {worker_id}] Coordinator says all required captions are complete. "
                f"Captioned {done_count} scene(s) here. Exiting cleanly."
            )
            break

        items = resp.get("work", [])

        if not items:
            status = resp.get("status", {})
            remaining = status.get("remaining", "?")
            print(
                f"[worker {worker_id}] No {model} work available right now "
                f"(coordinator reports {remaining} total job(s) remaining); "
                f"waiting {args.poll_interval}s."
            )
            time.sleep(args.poll_interval)
            continue

        for item in items:
            index = item["index"]
            label = item["label"]

            print(
                f"[worker {worker_id}] Captioning scene {index} ({label}) "
                f"with {model}..."
            )

            try:
                captions = llm_caption(
                    item["scene_str"],
                    deterministic=item["deterministic"],
                    game=item["game_name"],
                    tileset=item["tileset"],
                    llm=args.llm,
                    model=model,
                    num_captions=item["num_captions"],
                )
            except Exception as exc:
                # Do not kill the worker. The coordinator's lease will eventually
                # reclaim this job if it remains assigned.
                print(
                    f"[worker {worker_id}] LLM failure on scene {index}: {exc}. "
                    "Continuing with the next job."
                )
                continue

            if len(captions) != item["num_captions"]:
                print(
                    f"[worker {worker_id}] Scene {index} returned "
                    f"{len(captions)}/{item['num_captions']} captions. "
                    "Reporting it for retry."
                )
                result = post_result(
                    coordinator, worker_id, args.llm, model,
                    index, captions, args.poll_interval
                )
                if result.get("retry"):
                    continue

            result = post_result(
                coordinator, worker_id, args.llm, model,
                index, captions, args.poll_interval
            )

            if result.get("ok"):
                done_count += 1
                status = result.get("status", {})
                progress = (
                    f"{status.get('done', '?')}/{status.get('total_jobs', '?')} "
                    "total jobs complete"
                )
                print(
                    f"[worker {worker_id}] Scene {index} ({label}) recorded successfully. "
                    f"{progress}"
                )

                if result.get("shutdown"):
                    print(
                        f"[worker {worker_id}] Coordinator sent the final shutdown message. "
                        "Exiting cleanly."
                    )
                    return
            else:
                print(
                    f"[worker {worker_id}] Coordinator did not accept scene {index}: "
                    f"{result.get('error', 'unknown error')}"
                )


if __name__ == "__main__":
    main()
