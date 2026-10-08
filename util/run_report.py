"""
One JSON cost report per artifact: the energy (util/energy_tracking.py) and token spend
(util/token_tracking.py) of every run that went into producing it.

Where the report goes
---------------------
The report lives beside whatever the run produced, so the cost travels with the artifact:

    dataset   Game_MMLV/DATA/MMLV_LevelsAndCaptions-llm.json
              -> Game_MMLV/DATA/MMLV_LevelsAndCaptions-llm.costs.json   (dataset_report_path)
    model     <output_dir>/energy_report.json                            (MODEL_REPORT_NAME)

A script calls set_path() once it knows its artifact. A run that never does (or that dies
before getting that far) falls back to DEFAULT_PATH in the working directory.

Shape
-----
    {
      "runs": [
        {"command": "...", "energy": {...}, "tokens": {...}},
        ...
      ],
      "totals": {"runs": 2, "energy": {...summed...}, "tokens": {...summed...}}
    }

Each run appends to `runs`, so resuming a captioning or training run adds to the history of
the artifact it resumed rather than overwriting it, and `totals` is recomputed over all of
them on every write. A run that starts its artifact over (set_path(..., append=False))
starts the report over too.

Sections are contributed independently: track_energy opens a run with open_run(), anything
inside it adds a section with add_section(), and track_energy writes the lot with
close_run(). add_section() outside an open run (a caption worker, which has no energy
tracking) writes a one-section run immediately.
"""

import json
import os
import sys

DEFAULT_PATH = "energy_report.json"
DATASET_SUFFIX = ".costs.json"
MODEL_REPORT_NAME = "energy_report.json"

# Per-run fields summed into `totals`, by section. Rates, percentages and cumulative
# figures are left out: summing them across runs means nothing.
_TOTALED = {
    "energy": [
        "duration_seconds",
        "cpu_energy_kwh",
        "ram_energy_kwh",
        "gpu_energy_kwh",
        "measured_energy_kwh",
        "total_energy_kwh",
        "co2eq_kg",
    ],
    "tokens": [
        "duration_seconds",
        "scenes",
        "calls",
        "unreported_calls",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    ],
}

# The run being assembled, as {section: data}, or None when no run is open.
_run = None
_path = None
_append = True


def dataset_report_path(dataset_path):
    """'X.json' or 'X.jsonl' -> 'X.costs.json', in the same directory."""
    stem, ext = os.path.splitext(dataset_path)
    return (stem if ext in (".json", ".jsonl") else dataset_path) + DATASET_SUFFIX


def set_path(path, append=True):
    """
    Where this run's report is written. append=False discards any runs already in the file,
    for a run that replaces its artifact from scratch.
    """
    global _path, _append
    _path, _append = path, append


def open_run():
    """
    Starts assembling a run. Call it before anything edits sys.argv (track_energy strips
    --energy_detail), so the recorded command is what was actually typed.
    """
    global _run, _path, _append
    _run, _path, _append = {"command": " ".join(sys.argv)}, None, True


def add_section(name, data):
    """
    Adds one section to the open run, or, with no run open, writes it as a run of its own.
    Returns the path written to in the latter case, else None.
    """
    if _run is None:
        return _write({name: data})
    _run[name] = data
    return None


def close_run():
    """Writes the open run's sections and closes it. Returns the path written to."""
    global _run
    sections, _run = _run or {}, None
    return _write(sections)


def _totals(runs):
    totals = {"runs": len(runs)}
    for section, fields in _TOTALED.items():
        present = [run[section] for run in runs if isinstance(run.get(section), dict)]
        if present:
            # Rounded to the finest precision any field is recorded at, so float error
            # accumulated over many runs doesn't surface as 0.30000000000000004.
            totals[section] = {
                field: round(sum(entry.get(field) or 0 for entry in present), 9)
                for field in fields
            }
    return totals


def _load_runs(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("runs", [])
    except Exception as exc:
        print(f"[report] Could not read existing {path} ({exc}); starting it over.")
        return []


def _write(sections):
    path = _path or DEFAULT_PATH
    runs = _load_runs(path) if _append else []
    # An open run already recorded its command; a standalone section takes the current one.
    runs.append({"command": " ".join(sys.argv), **sections})

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Written to a temp file and swapped in, so a crash mid-write can't truncate the history
    # of every earlier run.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "totals": _totals(runs)}, f, indent=2)
    os.replace(tmp, path)
    return path
