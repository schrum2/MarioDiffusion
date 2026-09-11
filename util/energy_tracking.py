"""
Whole-run energy tracking: codecarbon for CPU/RAM, plus a GPU estimate.

Why the GPU is estimated
------------------------
codecarbon reads GPU power through NVML (`nvmlDeviceGetPowerUsage` /
`nvmlDeviceGetTotalEnergyConsumption`). Recent NVIDIA drivers deprecated power
telemetry on consumer GeForce cards, so on such machines every NVML power call
returns NOT_SUPPORTED and codecarbon records the GPU as 0.000000 kWh -- which
badly understates GPU-bound runs. `nvidia-smi --query-gpu=power.draw` reports
`[N/A]` on those drivers too, so this is a driver limitation rather than a
codecarbon bug, and codecarbon exposes no `force_gpu_power` knob to patch around
it.

NVML *does* still expose per-GPU utilization (%) and the enforced power limit (W),
so this module samples utilization across the run and estimates:

    gpu_kwh = sum_over_gpus(mean_util_fraction * power_limit_w)
              * duration_hours / 1000

That is deliberately crude -- real draw is not linear in utilization, and an idle
GPU still pulls non-zero power that this scores as zero. Treat it as an
order-of-magnitude figure, good for comparing runs against each other, not as a
measurement. Every raw input (mean utilization, sample count, duration, power
limit) is written to the CSV so a better power model can be applied later without
re-running anything.

If NVML power telemetry *is* available (a datacenter GPU, or an older driver),
codecarbon's measured GPU energy is used instead and the heuristic is skipped --
the `gpu_source` column records which path was taken.

Usage
-----
    from util.energy_tracking import track_energy

    @track_energy(project_name="train_diffusion")
    def main():
        ...

On completion a single line is printed:

    [CodeCarbon] energy: CPU+RAM 0.000155 kWh + GPU 0.000165 kWh (est)
                 = 0.000320 kWh (0.000152 kg CO2e)

Pass --energy_detail on the command line for the full per-component breakdown.
The decorator consumes that flag before the wrapped function parses arguments, so
a script whose parse_args() runs *inside* the decorated function needs no changes.

A script that parses arguments before the decorated function is called (train_mlm.py
parses at module level, then calls the decorated train()) must register the flag in
its own parser, since argparse sees sys.argv before this decorator can strip it.

Either way one row is appended to energy_summary.csv, and codecarbon's own
emissions.csv is still written alongside it, unchanged, for full provenance.
"""

import csv
import os
import sys
import threading
import time
from datetime import datetime
from functools import wraps

# How often to poll NVML for utilization. Utilization queries are cheap and only a
# running sum is kept, so this stays inexpensive across multi-hour training runs.
SAMPLE_SECONDS = 1.0

# Written to the current working directory, beside codecarbon's emissions.csv.
OUTPUT_FILE = "energy_summary.csv"

# Opt-in switch for the full breakdown; the default is a single line.
DETAIL_FLAG = "--energy_detail"

FIELDNAMES = [
    "start_time",
    "end_time",
    "project_name",
    "run_id",
    "duration_seconds",
    "cpu_energy_kwh",
    "cpu_power_w",
    "cpu_utilization_percent",
    "ram_energy_kwh",
    "ram_power_w",
    "ram_utilization_percent",
    "gpu_energy_kwh",
    "gpu_power_w",
    "gpu_source",
    "gpu_mean_util_percent",
    "gpu_power_limit_w",
    "gpu_samples",
    "measured_energy_kwh",
    "total_energy_kwh",
    "co2eq_kg",
    "note",
    "command",
]

# Decimal places per column. Raw floats would otherwise land in the CSV at full
# repr precision ("3.8740779444601186e-05"), which is unreadable in a spreadsheet
# and implies far more precision than the underlying measurement has.
_PLACES = {
    "duration_seconds": 2,
    "cpu_energy_kwh": 9,
    "cpu_power_w": 2,
    "cpu_utilization_percent": 2,
    "ram_energy_kwh": 9,
    "ram_power_w": 2,
    "ram_utilization_percent": 2,
    "gpu_energy_kwh": 9,
    "gpu_power_w": 2,
    "measured_energy_kwh": 9,
    "total_energy_kwh": 9,
    "co2eq_kg": 9,
}


class _UtilizationSampler(threading.Thread):
    """Polls per-GPU utilization on a background daemon thread."""

    def __init__(self, pynvml, handles):
        super().__init__(daemon=True)
        self._pynvml = pynvml
        self._handles = handles
        self._util_sums = [0.0] * len(handles)
        self._samples = 0
        self._stop_event = threading.Event()

    def run(self):
        # wait() doubles as the sleep and the stop check, so shutdown is prompt.
        while not self._stop_event.wait(SAMPLE_SECONDS):
            for i, handle in enumerate(self._handles):
                try:
                    self._util_sums[i] += self._pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    # A transient NVML hiccup shouldn't take down training.
                    pass
            self._samples += 1

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5 * SAMPLE_SECONDS)

    @property
    def samples(self):
        return self._samples

    def mean_utilizations(self):
        """Mean utilization percentage per GPU over the run."""
        if self._samples == 0:
            return [0.0] * len(self._handles)
        return [total / self._samples for total in self._util_sums]


def _open_nvml():
    """Returns (pynvml, handles, power_limits_w), or None if NVML is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                   for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        return None

    if not handles:
        return None

    power_limits = []
    for handle in handles:
        try:
            # nvmlDeviceGetEnforcedPowerLimit returns milliwatts.
            power_limits.append(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
        except Exception:
            power_limits.append(0.0)

    return pynvml, handles, power_limits


def _mean_power_w(energy_kwh, duration_seconds):
    """Average watts implied by an energy total over a duration."""
    if not duration_seconds:
        return 0.0
    return energy_kwh * 1000.0 * 3600.0 / duration_seconds


def _format_utilization(percent):
    """
    codecarbon reports 0.0 for cpu_utilization_percent under the Windows EMI
    tracking method even though the energy figure itself is valid, so show the
    percentage only when it actually carries information.
    """
    if not percent:
        return "util n/a"
    return f"{percent:.1f}% util"


def _format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _csv_row(summary):
    """
    Projects the summary onto FIELDNAMES, rendering floats in plain decimal at a
    sensible precision rather than at full repr precision / in scientific notation.
    """
    row = {}
    for field in FIELDNAMES:
        value = summary[field]
        places = _PLACES.get(field)
        row[field] = f"{value:.{places}f}" if places is not None else value
    return row


def _write_row(row):
    # Append, writing the header only when creating the file, so repeated runs
    # accumulate in one CSV the way codecarbon's emissions.csv does.
    write_header = not os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Resolved once per process: the flag is stripped from sys.argv the first time it
# is read, so a second decorated call in the same process would otherwise never
# see it. Scripts here have a single entry point, but caching keeps the answer
# consistent regardless.
_detail_requested = None


def _pop_detail_flag():
    """
    Consumes DETAIL_FLAG from sys.argv, returning whether it was present.

    Removing it before the wrapped function runs means every script using this
    decorator gains the flag for free -- their argparse parsers never see it, so
    none of them need to register it (and none of them error on it as unknown).
    """
    global _detail_requested
    if _detail_requested is None:
        _detail_requested = DETAIL_FLAG in sys.argv
        if _detail_requested:
            sys.argv = [arg for arg in sys.argv if arg != DETAIL_FLAG]
    return _detail_requested


def _print_oneline(summary):
    """The default end-of-run message: energy, in one line."""
    cpu_ram = summary["cpu_energy_kwh"] + summary["ram_energy_kwh"]
    source = summary["gpu_source"]

    if source == "none":
        tail = "(no GPU detected)"
    else:
        estimate_marker = " (est)" if source == "heuristic" else ""
        tail = f"+ GPU {summary['gpu_energy_kwh']:.6f} kWh{estimate_marker}"

    print(f"[CodeCarbon] energy: CPU+RAM {cpu_ram:.6f} kWh {tail}"
          f" = {summary['total_energy_kwh']:.6f} kWh"
          f" ({summary['co2eq_kg']:.6f} kg CO2e)")


def _component_line(label, energy_kwh, power_w, detail):
    return f"  {label:<11}{energy_kwh:>11.6f} kWh {power_w:>7.1f} W   {detail}".rstrip()


def _print_summary(summary):
    """The single end-of-run message. Everything else is silenced."""
    width = 62
    rule = "-" * width
    source = summary["gpu_source"]

    lines = [
        "",
        "=" * width,
        f" Energy summary - {summary['project_name']}",
        f" Duration {_format_duration(summary['duration_seconds'])}"
        f" ({summary['duration_seconds']:.1f} s)",
        rule,
        _component_line("CPU", summary["cpu_energy_kwh"], summary["cpu_power_w"],
                        _format_utilization(summary["cpu_utilization_percent"])),
        _component_line("RAM", summary["ram_energy_kwh"], summary["ram_power_w"],
                        _format_utilization(summary["ram_utilization_percent"])),
    ]

    if source == "none":
        lines.append("  GPU        no NVIDIA GPU detected")
    elif source == "heuristic":
        lines.append(_component_line(
            "GPU (est.)", summary["gpu_energy_kwh"], summary["gpu_power_w"],
            f"{summary['gpu_mean_util_percent_mean']:.1f}% util",
        ))
    else:
        lines.append(_component_line(
            "GPU", summary["gpu_energy_kwh"], summary["gpu_power_w"], "measured",
        ))

    lines.append(rule)

    # Spell out how TOTAL was composed, so it's never ambiguous whether the
    # estimate is included in it -- and, just below, in the CO2 figure.
    if source == "heuristic":
        breakdown = (f"= {summary['measured_energy_kwh']:.6f} measured"
                     f" + {summary['gpu_energy_kwh']:.6f} est.")
    else:
        breakdown = "(all measured)"

    lines += [
        f"  {'TOTAL':<11}{summary['total_energy_kwh']:>11.6f} kWh   {breakdown}",
        f"  {'CO2e':<11}{summary['co2eq_kg']:>11.6f} kg    (grid intensity x TOTAL)",
        rule,
    ]

    if source == "heuristic":
        lines.append(f"  GPU is a heuristic: avg util x "
                     f"{summary['gpu_power_limit_total_w']:.0f} W limit x time.")

    lines += [f"  Raw values appended to {OUTPUT_FILE}", "=" * width, ""]
    print("\n".join(lines))


def track_energy(project_name):
    """
    Decorator measuring whole-run energy for the wrapped call.

    Runs codecarbon (CPU/RAM, and GPU where NVML supports it) alongside a GPU
    utilization sampler, then prints one summary and appends one row to
    energy_summary.csv. codecarbon's periodic logging is silenced so completion
    produces a single message rather than a running commentary.

    Both the summary and the CSV row are emitted from a finally block, so they
    still appear if the wrapped function raises or calls sys.exit().
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            import logging

            from codecarbon import EmissionsTracker

            # EmissionsTracker applies its own log_level partway through __init__,
            # after the "multiple instances" check has already logged, so that one
            # warning escapes. Setting the level on codecarbon's logger up front
            # catches it too, leaving the summary below as the only output.
            logging.getLogger("codecarbon").setLevel(logging.ERROR)

            # Record what was actually typed before the flag is stripped out.
            command = " ".join(sys.argv)
            show_detail = _pop_detail_flag()

            nvml = _open_nvml()
            sampler = None
            power_limits = []
            if nvml is not None:
                pynvml, handles, power_limits = nvml
                sampler = _UtilizationSampler(pynvml, handles)

            # log_level="error" suppresses codecarbon's setup banner and its
            # every-15-seconds progress lines (including the repeated NVML
            # NOT_SUPPORTED tracebacks) so only our summary reaches the console.
            tracker = EmissionsTracker(project_name=project_name, log_level="error")

            started = time.time()
            started_at = datetime.now()
            if sampler is not None:
                sampler.start()
            tracker.start()
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    tracker.stop()
                except Exception:
                    pass
                if sampler is not None:
                    sampler.stop()

                elapsed = time.time() - started
                data = getattr(tracker, "final_emissions_data", None)

                duration = getattr(data, "duration", None) or elapsed
                cpu_energy = getattr(data, "cpu_energy", 0.0) or 0.0
                ram_energy = getattr(data, "ram_energy", 0.0) or 0.0
                measured_gpu_energy = getattr(data, "gpu_energy", 0.0) or 0.0

                mean_utils = sampler.mean_utilizations() if sampler is not None else []
                samples = sampler.samples if sampler is not None else 0

                # Prefer codecarbon's real GPU measurement wherever the driver
                # supports it; fall back to the heuristic only when it reported
                # nothing, and report neither when there's no NVIDIA GPU at all.
                if measured_gpu_energy > 0:
                    gpu_source = "measured"
                    gpu_energy = measured_gpu_energy
                elif sampler is not None:
                    gpu_source = "heuristic"
                    estimated_watts = sum(
                        (util / 100.0) * limit
                        for util, limit in zip(mean_utils, power_limits)
                    )
                    gpu_energy = estimated_watts * (duration / 3600.0) / 1000.0
                else:
                    gpu_source = "none"
                    gpu_energy = 0.0

                measured_energy = cpu_energy + ram_energy + (
                    measured_gpu_energy if gpu_source == "measured" else 0.0
                )
                total_energy = cpu_energy + ram_energy + gpu_energy

                # Rescale CO2 to the adjusted total. codecarbon's own figure is
                # derived from the energy it measured, so with the GPU counted as
                # zero it understates emissions by the same proportion. Recover the
                # grid intensity it used and reapply it to the corrected total.
                emissions = getattr(data, "emissions", 0.0) or 0.0
                energy_consumed = getattr(data, "energy_consumed", 0.0) or 0.0
                intensity = emissions / energy_consumed if energy_consumed > 0 else 0.0
                co2eq = intensity * total_energy

                mean_util_overall = (sum(mean_utils) / len(mean_utils)) if mean_utils else 0.0
                note = {
                    "heuristic": "GPU heuristic: avg util x limit x time",
                    "measured": "all components measured",
                    "none": "no NVIDIA GPU; CPU+RAM only",
                }[gpu_source]

                summary = {
                    "start_time": started_at.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    # codecarbon's own run_id for this run, so a row here joins
                    # exactly to its emissions.csv row instead of by timestamp
                    # proximity (the two are written moments apart).
                    "run_id": str(getattr(data, "run_id", "")),
                    # Two runs of the same script share a project_name and differ
                    # only by timestamp, which makes rows hard to tell apart. The
                    # command line carries --output_dir, --json and the rest, so
                    # each row says which configuration produced it.
                    "command": command,
                    "project_name": project_name,
                    "duration_seconds": round(duration, 2),
                    "cpu_energy_kwh": cpu_energy,
                    "cpu_power_w": _mean_power_w(cpu_energy, duration),
                    "cpu_utilization_percent": getattr(data, "cpu_utilization_percent", 0.0) or 0.0,
                    "ram_energy_kwh": ram_energy,
                    "ram_power_w": _mean_power_w(ram_energy, duration),
                    "ram_utilization_percent": getattr(data, "ram_utilization_percent", 0.0) or 0.0,
                    "gpu_energy_kwh": gpu_energy,
                    "gpu_power_w": _mean_power_w(gpu_energy, duration),
                    "gpu_source": gpu_source,
                    "gpu_mean_util_percent": ";".join(f"{u:.2f}" for u in mean_utils),
                    "gpu_power_limit_w": ";".join(f"{p:.1f}" for p in power_limits),
                    "gpu_samples": samples,
                    "measured_energy_kwh": measured_energy,
                    "total_energy_kwh": total_energy,
                    "co2eq_kg": co2eq,
                    "note": note,
                }

                _write_row(_csv_row(summary))

                # Extra derived values the printed summary wants but the CSV
                # keeps in per-GPU form.
                summary["gpu_mean_util_percent_mean"] = mean_util_overall
                summary["gpu_power_limit_total_w"] = sum(power_limits)

                if show_detail:
                    _print_summary(summary)
                else:
                    _print_oneline(summary)
        return wrapper
    return decorator
