"""
Worker for the distributed captioning coordinator (caption_coordinator.py). Run one of these
per lab machine, pointed at whichever machine is running the coordinator:

    python caption_worker.py --coordinator http://<coordinator-ip>:8765 --llm ollama

Each worker repeatedly: asks the coordinator for a small batch of scenes, captions them with
its own local LLM (so N lab machines running ollama give you roughly N-way throughput), and
posts the captions back. It exits automatically once the coordinator reports every scene done.
No shared filesystem is needed -- only network access to the coordinator's host:port.
"""
import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from llm_ascii_to_caption import llm_caption, DEFAULT_MODELS


def get_work(coordinator: str, worker_id: str, n: int) -> dict:
    url = f"{coordinator}/work?worker={urllib.parse.quote(worker_id)}&n={n}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_result(coordinator: str, worker_id: str, index: int, captions: list[str]):
    payload = json.dumps({"index": index, "worker": worker_id, "captions": captions}).encode("utf-8")
    req = urllib.request.Request(f"{coordinator}/result", data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coordinator", required=True, help="Base URL of the coordinator, e.g. http://192.168.1.20:8765")
    ap.add_argument("--llm", choices=["claude", "openai", "gemini", "ollama"], default="ollama")
    ap.add_argument("--model", default=None, help="Defaults per --llm branch, same as llm_ascii_to_caption.py")
    ap.add_argument("--worker-id", default=None, help="Defaults to '<hostname>-<pid>'")
    ap.add_argument("--batch-size", type=int, default=3, help="Scenes requested per poll.")
    ap.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to wait when no work is available yet.")
    args = ap.parse_args()

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    model = args.model or DEFAULT_MODELS[args.llm]
    coordinator = args.coordinator.rstrip("/")

    print(f"[worker {worker_id}] Using --llm {args.llm} --model {model}, polling {coordinator}\n")

    done_count = 0
    while True:
        try:
            resp = get_work(coordinator, worker_id, args.batch_size)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[worker {worker_id}] Couldn't reach coordinator ({e}); retrying in {args.poll_interval}s...")
            time.sleep(args.poll_interval)
            continue

        items = resp.get("work", [])
        if not items:
            if resp.get("finished"):
                print(f"[worker {worker_id}] Coordinator reports all scenes done. Captioned {done_count} here. Exiting.")
                break
            time.sleep(args.poll_interval)
            continue

        for item in items:
            captions = llm_caption(
                item["scene_str"], deterministic=item["deterministic"], game=item["game_name"],
                tileset=item["tileset"], llm=args.llm, model=model, num_captions=item["num_captions"],
            )
            post_result(coordinator, worker_id, item["index"], captions)
            done_count += 1
            status = "ok" if len(captions) == item["num_captions"] else \
                f"skipped ({len(captions)}/{item['num_captions']} captions)"
            print(f"[worker {worker_id}] scene {item['index']} ({item['label']}): {status} [{done_count} total]")


if __name__ == "__main__":
    main()
