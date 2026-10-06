"""
Per-run token accounting, persisted to the run's JSON cost report.

llm_ascii_to_caption.py already counts backend-reported tokens during a run (see its
TokenUsage class) and prints a summary when it finishes, but that summary scrolls away
with the terminal. This module records it as the "tokens" section of the same report that
util/energy_tracking.py writes the run's energy to (see util/run_report.py), so one file
beside the captioned dataset holds everything the dataset cost to make.

The section records three things:

  when      start/end timestamps and duration of the captioning itself
  what      the game, the dataset that was captioned, the checkpoint/output paths, and
            every other knob set on the command line, under `settings` -- so runs can be
            grouped by configuration rather than by reading `command`
  cost      this run's token totals (plus derived per-scene and per-second rates), and,
            under `cumulative`, the totals across the checkpoint when the run resumed an
            earlier one

Why `settings` when the report already holds the whole command line: `command` is only
good for eyeballing. Broken-out settings answer "how many tokens per scene does
--grid-format tokens cost versus ascii" without parsing anything, and they normalize
defaults -- a flag left off the command line still appears with the value actually used.

Usage
-----
    from util.token_tracking import log_run

    log_run(usage=run_usage, scenes=scenes_captioned, started_at=start_time,
            args=args, script="llm_ascii_to_caption", extra={"model": model})

`usage` is duck-typed rather than imported: anything exposing TokenUsage's fields works,
which keeps util/ from importing back out of the scripts that use it.

Inside a @track_energy run the section joins that run's report and is written when the run
ends. Outside one (caption_worker) it is written immediately, as a run of its own.
"""

import time
from datetime import datetime

from util import run_report

# Settings pulled straight off the caller's argparse Namespace, by identical name. A run
# whose parser has no such flag (a worker has no --game) leaves it out.
#
# --api-key-file is deliberately absent: a key path is credential-adjacent and says
# nothing about what the run cost. The report's `command` is the escape hatch for anyone
# who needs to see exactly what was typed.
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
    # reprompt policy -- these directly drive token spend, so they matter to a cost report
    "retry_on_empty",
    "retry_on_nonascii",
    "max_reprompts",
    "max_caption_retries",
    # distributed worker only
    "coordinator",
    "batch_size",
    "poll_interval",
]


def _rate(value, per, places=1):
    """value/per, or 0.0 when the denominator is zero (an empty or instant run)."""
    return round(value / per, places) if per else 0.0


def log_run(usage, scenes, started_at, args=None, script="", extra=None,
            cumulative=None, cumulative_scenes=0, cumulative_missing=0,
            checkpoint="", resumed=False, quiet=False):
    """
    Records a finished captioning run as the "tokens" section of its cost report.

    usage               object with TokenUsage's fields (input_tokens, output_tokens,
                        cached_input_tokens, reasoning_tokens, calls, unreported)
    scenes              scenes this run captioned, for the per-scene average
    started_at          run start as epoch seconds (time.time() at the top of the run);
                        the end time and duration are taken from now
    args                argparse Namespace; every ARG_COLUMNS name found on it is recorded
    script              which entry point produced the section ("llm_ascii_to_caption", ...)
    extra               values resolved at runtime that override/augment the ones read from
                        args, e.g. {"model": model} when --model defaulted to None
    cumulative          usage for the whole checkpoint including earlier resumed runs;
                        defaults to this run's usage when nothing was resumed
    cumulative_missing  scenes in the checkpoint from before token accounting existed, and
                        so absent from `cumulative` -- recorded so a low total is explicable
    quiet               suppress the one-line "saved" confirmation printed outside a
                        @track_energy run

    Never raises: a failure to write a bookkeeping file should not take down (or mask the
    real error of) a captioning run that has already finished its real work.
    """
    try:
        ended = time.time()
        duration = max(ended - started_at, 0.0)
        cumulative = cumulative if cumulative is not None else usage

        settings = {name: getattr(args, name) for name in ARG_COLUMNS if hasattr(args, name)}
        section = {
            "start_time": datetime.fromtimestamp(started_at).strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": datetime.fromtimestamp(ended).strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "script": script,
            "checkpoint": str(checkpoint),
            "resumed": bool(resumed),
            "settings": settings,
            # This run's spend.
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
            # The whole checkpoint's spend, including runs that came before this one.
            # Equal to this run's totals when nothing was resumed.
            "cumulative": {
                "scenes": cumulative_scenes or scenes,
                "input_tokens": cumulative.input_tokens,
                "output_tokens": cumulative.output_tokens,
                "total_tokens": cumulative.total,
                "scenes_missing_usage": cumulative_missing,
            },
        }

        # Resolved values win: --model may have been left off the command line and filled
        # in from DEFAULT_MODELS, and the report should say which model actually ran.
        # Anything that isn't a setting (worker_id) sits beside the totals instead.
        for name, value in (extra or {}).items():
            (settings if name in ARG_COLUMNS else section)[name] = value

        written = run_report.add_section("tokens", section)

        # Inside a @track_energy run nothing is written yet; its end-of-run line names the
        # report this section lands in.
        if written and not quiet:
            print(f"[tokens] Run saved to {written}\n")
    except Exception as exc:
        print(f"[tokens] Could not save token usage: {exc}\n")
