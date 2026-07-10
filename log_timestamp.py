"""
Append a timestamped step-completion record to a single-execution JSONL timing log.

This is a tiny, dependency-free helper meant to be called between the major
steps of a Mega Man pipeline batch file (download, VGLC conversion, captioning,
splitting, training, evaluation, ...). Each call appends one JSON line to the
log file for the current pipeline execution, so the time spent on each component
can be reconstructed afterward.

Each log file belongs to exactly one full pipeline execution: the batch file
picks a fresh, uniquely-named file at the start of the run and every step
appends to it. Nothing should append to a log once its run has finished.

Timestamps use the same "%Y-%m-%d %H:%M:%S" format as the per-model training
logs (see calculate_execution_time.py), so the two can be compared directly.

Example:
    python log_timestamp.py --log_file timing_logs/MM-unconditional-big_20260629_164530.jsonl --status start --event "pipeline start"
    python log_timestamp.py --log_file timing_logs/MM-unconditional-big_20260629_164530.jsonl --event "MMLV download"
    python log_timestamp.py --log_file timing_logs/MM-unconditional-big_20260629_164530.jsonl --event "MMLV to VGLC conversion"

Each record looks like:
    {"timestamp": "2026-06-29 14:03:12", "event": "MMLV download", "status": "complete",
     "elapsed_seconds_since_prev": 842.0, "prev_event": "pipeline start"}
"""

import argparse
import json
import os
from datetime import datetime

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def find_previous_entry(log_file):
    """Return the last entry already in log_file, or None if it's new/empty."""
    if not os.path.exists(log_file):
        return None
    last = None
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def main():
    parser = argparse.ArgumentParser(
        description="Append a timestamped step-completion record to a single-execution JSONL timing log."
    )
    parser.add_argument("--log_file", required=True,
                        help="JSONL file for the current pipeline execution to append to.")
    parser.add_argument("--event", required=True,
                        help="Name of the step that just completed (or started).")
    parser.add_argument("--status", default="complete",
                        help="Status of the step: 'start', 'complete', or 'failed' (default: complete).")
    parser.add_argument("--note", default=None,
                        help="Optional free-form note to store alongside the record.")
    args = parser.parse_args()

    now = datetime.now()
    timestamp = now.strftime(TIMESTAMP_FORMAT)

    elapsed_seconds = None
    prev_event = None
    prev = find_previous_entry(args.log_file)
    if prev is not None and "timestamp" in prev:
        try:
            prev_dt = datetime.strptime(prev["timestamp"], TIMESTAMP_FORMAT)
            elapsed_seconds = round((now - prev_dt).total_seconds(), 1)
            prev_event = prev.get("event")
        except ValueError:
            pass

    record = {
        "timestamp": timestamp,
        "event": args.event,
        "status": args.status,
    }
    if args.note is not None:
        record["note"] = args.note
    if elapsed_seconds is not None:
        record["elapsed_seconds_since_prev"] = elapsed_seconds
        record["prev_event"] = prev_event

    log_dir = os.path.dirname(os.path.abspath(args.log_file))
    os.makedirs(log_dir, exist_ok=True)
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    message = f"[timing] {timestamp} | {args.event} ({args.status})"
    if elapsed_seconds is not None:
        message += f" | +{elapsed_seconds:.1f}s since '{prev_event}'"
    print(message)


if __name__ == "__main__":
    main()
