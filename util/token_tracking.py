"""
Per-run token accounting, persisted to a CSV.

llm_ascii_to_caption.py already counts backend-reported tokens during a run (see its
TokenUsage class) and prints a summary when it finishes, but that summary scrolls away
with the terminal. This module appends one row per run to token_usage.csv, in the same
spirit as util/energy_tracking.py's energy_summary.csv: a flat, spreadsheet-friendly
history that makes runs comparable to each other after the fact.

A row records three things:

  when    start/end timestamps and duration, so a row can be lined up with the matching
          energy_summary.csv / emissions.csv row for the same run
  what    the game, the dataset that was captioned, the checkpoint/output paths, and
          every other knob set on the command line -- one column per CLI flag, so runs
          can be filtered and grouped by configuration rather than by reading `command`
  cost    this run's token totals (plus derived per-scene and per-second rates), and the
          cumulative totals across the checkpoint when the run resumed an earlier one

Why columns per flag when `command` already holds the whole command line: `command` is
only good for eyeballing. Broken-out columns let a spreadsheet answer "how many tokens
per scene does --grid-format tokens cost versus ascii" without parsing anything, and
they normalize defaults -- a flag left off the command line still lands in its column
with the value that was actually used.

Usage
-----
    from util.token_tracking import log_run

    log_run(usage=run_usage, scenes=scenes_captioned, started_at=start_time,
            args=args, script="llm_ascii_to_caption", extra={"model": model})

`usage` is duck-typed rather than imported: anything exposing TokenUsage's fields works,
which keeps util/ from importing back out of the scripts that use it.

Every caller's arguments are optional beyond that, and unknown/missing ones are written
blank, so one CSV can hold rows from both a single-machine run (llm_ascii_to_caption)
and a distributed worker (caption_worker) without either needing the other's columns.
The `script` column says which produced a row.
"""

import csv
import os
import sys
import time
from datetime import datetime

# Written to the current working directory, beside energy_summary.csv.
OUTPUT_FILE = "token_usage.csv"

# Columns pulled straight off the caller's argparse Namespace, by identical name. A run
# whose parser has no such flag (a worker has no --game) writes that column blank.
#
# --api-key-file is deliberately absent: a key path is credential-adjacent and says
# nothing about what the run cost. The full `command` column is the escape hatch for
# anyone who needs to see exactly what was typed.
ARG_COLUMNS = [
    # what was captioned
    "game",
    "levels",
    "output",
    "limit",
    "shard_index",
    "shard_count",
    # which model, and how it was asked
    "llm",
    "model",
    "num_captions",
    "caption_mode",
    "caption_key",
    "grid_format",
    "temperature",
    "max_tokens",
    "num_ctx",
    "max_num_ctx",
    "timeout",
    "retries",
    # reprompt policy -- these directly drive token spend, so they matter to a cost row
    "retry_on_empty",
    "retry_on_nonascii",
    "max_reprompts",
    "max_caption_retries",
    # distributed worker only
    "coordinator",
    "batch_size",
    "poll_interval",
]

FIELDNAMES = (
    [
        "start_time",
        "end_time",
        "duration_seconds",
        "script",
    ]
    + ARG_COLUMNS
    + [
        # Resolved at runtime rather than read off a flag.
        "worker_id",
        "checkpoint",
        "resumed",
        # This run's spend.
        "scenes",
        "calls",
        "unreported_calls",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "tokens_per_scene",
        "tokens_per_second",
        "output_tokens_per_second",
        # The whole checkpoint's spend, including runs that came before this one.
        # Equal to this run's totals when nothing was resumed.
        "cumulative_scenes",
        "cumulative_input_tokens",
        "cumulative_output_tokens",
        "cumulative_total_tokens",
        "cumulative_scenes_missing_usage",
        "command",
    ]
)

# Decimal places per column, so rates don't land in the CSV at full float repr precision
# ("1234.5678901234567"), which is unreadable in a spreadsheet.
_PLACES = {
    "duration_seconds": 2,
    "tokens_per_scene": 1,
    "tokens_per_second": 1,
    "output_tokens_per_second": 1,
}


def _rate(value, per):
    """value/per, or 0.0 when the denominator is zero (an empty or instant run)."""
    return value / per if per else 0.0


def _cell(value):
    """
    Renders one value for the CSV.

    None becomes blank rather than the string "None": an unset --output or --limit means
    "not specified", and a blank cell reads that way in a spreadsheet while "None" sorts
    and filters as if it were data.
    """
    return "" if value is None else value


def _row(fields):
    row = {}
    for field in FIELDNAMES:
        value = _cell(fields.get(field))
        places = _PLACES.get(field)
        row[field] = f"{value:.{places}f}" if places is not None and value != "" else value
    return row


def _write_row(row):
    # Append, writing the header only when creating the file, so repeated runs accumulate
    # in one CSV the way codecarbon's emissions.csv does.
    write_header = not os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_run(usage, scenes, started_at, args=None, script="", extra=None,
            cumulative=None, cumulative_scenes=0, cumulative_missing=0,
            checkpoint="", resumed=False, command=None, quiet=False):
    """
    Appends one row describing a finished captioning run to OUTPUT_FILE.

    usage               object with TokenUsage's fields (input_tokens, output_tokens,
                        cached_input_tokens, reasoning_tokens, calls, unreported)
    scenes              scenes this run captioned, for the per-scene average
    started_at          run start as epoch seconds (time.time() at the top of the run);
                        the end time and duration are taken from now
    args                argparse Namespace; every ARG_COLUMNS name found on it is recorded
    script              which entry point produced the row ("llm_ascii_to_caption", ...)
    extra               values resolved at runtime that override/augment the ones read from
                        args, e.g. {"model": model} when --model defaulted to None
    cumulative          usage for the whole checkpoint including earlier resumed runs;
                        defaults to this run's usage when nothing was resumed
    cumulative_missing  scenes in the checkpoint from before token accounting existed, and
                        so absent from `cumulative` -- recorded so a low total is explicable
    command             defaults to the current command line
    quiet               suppress the one-line "appended" confirmation

    Never raises: a failure to write a bookkeeping CSV should not take down (or mask the
    real error of) a captioning run that has already finished its real work.
    """
    try:
        ended = time.time()
        duration = max(ended - started_at, 0.0)
        cumulative = cumulative if cumulative is not None else usage

        fields = {
            "start_time": datetime.fromtimestamp(started_at).strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": datetime.fromtimestamp(ended).strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "script": script,
            "worker_id": "",
            "checkpoint": str(checkpoint),
            "resumed": bool(resumed),
            "scenes": scenes,
            "calls": usage.calls,
            "unreported_calls": usage.unreported,
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_tokens": usage.total,
            "tokens_per_scene": _rate(usage.total, scenes),
            "tokens_per_second": _rate(usage.total, duration),
            "output_tokens_per_second": _rate(usage.output_tokens, duration),
            "cumulative_scenes": cumulative_scenes or scenes,
            "cumulative_input_tokens": cumulative.input_tokens,
            "cumulative_output_tokens": cumulative.output_tokens,
            "cumulative_total_tokens": cumulative.total,
            "cumulative_scenes_missing_usage": cumulative_missing,
            "command": command if command is not None else " ".join(sys.argv),
        }

        for name in ARG_COLUMNS:
            fields[name] = getattr(args, name, None)

        # Resolved values win: --model may have been left off the command line and filled
        # in from DEFAULT_MODELS, and the row should say which model actually ran.
        fields.update(extra or {})

        _write_row(_row(fields))

        if not quiet:
            print(f"[tokens] Run appended to {OUTPUT_FILE}\n")
    except Exception as exc:
        print(f"[tokens] Could not write {OUTPUT_FILE}: {exc}\n")
