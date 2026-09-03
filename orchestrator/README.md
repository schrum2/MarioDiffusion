# Lab training orchestrator

Coordinates `train_diffusion.py` (and `train_mlm.py`) runs across every machine in the
lab from one dashboard: queue jobs, see which machine is running what, get notified
when a machine goes offline mid-run, request a checkpoint-and-stop on a running job,
resume it later, and pull the finished files back over the network on demand.

## How it fits together

```
                 ┌─────────────────────────┐
                 │   coordinator/server.py │   <- runs on your machine, has the dashboard
                 │   (FastAPI + web UI)    │
                 └────────────▲────────────┘
                       HTTP (poll every 5s)
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
│ worker/agent.py│   │ worker/agent.py│   │ worker/agent.py│   ... one per lab machine
│  (machine A)   │   │  (machine B)   │   │  (machine C)   │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │ subprocess               │ subprocess               │
  train_diffusion.py         train_diffusion.py         train_diffusion.py
```

Workers always dial *out* to the coordinator (register → poll → report), so there's
nothing to open or forward on the worker machines — this matches how you described
wanting it ("workers look to the coordinator's IP"), and it also means a worker that
gets rebooted or reconnects to a different network just re-registers itself next time
it can reach the coordinator.

By default, output stays on the worker's local disk exactly as `train_diffusion.py`
already writes it (under whatever `--output_dir` you queue the job with). Nothing is
copied anywhere unless you click **fetch files** on a job, which zips that job's
`output_dir` on the worker and uploads it to the coordinator for download.

## One-time setup

**1. Patch `train_diffusion.py` on every machine.** The orchestrator needs two small
additions that the stock script doesn't have:

- A way to ask a *running* process to save a checkpoint and exit, without OS signals
  (which behave inconsistently for subprocesses on Windows vs. Linux). The patched
  script polls for a `STOP_REQUEST` file inside `output_dir` once per training step; the
  worker agent creates that file when you click "checkpoint & stop", the script saves a
  checkpoint at the current epoch and exits with a distinct code (75) so the coordinator
  can tell "paused on request" apart from "crashed" or "finished".
- A `--auto_resume` flag. The stock script blocks on an interactive
  `Resume training from last checkpoint? (y/n)` prompt when `output_dir` already has
  checkpoints in it — that will hang forever with no terminal attached, which is exactly
  the situation a worker agent runs training in. `--auto_resume` skips the prompt and
  resumes automatically. The coordinator adds this flag itself whenever it requeues or
  resumes a job, so you never have to add it by hand.

  The patched file is at `patched_scripts/train_diffusion.py` in this bundle — diff it
  against your copy if you want to review the changes, then replace your copy on every
  machine (or just the one central checkout all your machines share, if that's how your
  lab is set up).

  `train_mlm.py` is left as-is. It has no resume support upstream at all (it just
  errors out if `output_dir` exists), so there's nothing safe to auto-restart into. The
  orchestrator will still queue, run, and monitor `train_mlm.py` jobs; a crash just
  won't auto-requeue itself the way a diffusion job does. You said this matters less to
  you, so this seemed like the right place to not over-build.

**2. Start the coordinator**, on whichever machine you want to run the dashboard from
(your desk, or one of the lab machines):

```bash
cd coordinator
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8000 --key-phrase "secret"
```

Open `http://<that machine's IP>:8000` in a browser. Leave it running — closing the
browser tab doesn't stop it, only Ctrl+C does. If you restart it, queued/running jobs
are automatically put back in the queue (see "What happens on a crash or reboot" below).

**3. Start a worker agent on every lab machine**, pointed at the coordinator:

```bash
cd worker
pip install -r requirements.txt
python agent.py --coordinator http://<coordinator-ip>:8000 --repo-path C:\path\to\your\repo --key-phrase "secret"
```

If a machine has more than one GPU you want to use independently, run one agent process
with a GPU list; it shows up as one dashboard entry per GPU and runs one job per GPU
concurrently:

```bash
python agent.py --coordinator http://<coordinator-ip>:8000 --repo-path . --gpu-ids 0,1,2,3
```

Set this to launch at login (Windows Task Scheduler, or just a shortcut in the Startup
folder) if you want it to come back on its own whenever someone turns the machine back
on — the agent has no state that needs to survive a restart beyond the small
`.worker_id_*` file it writes next to itself, which just lets it re-identify itself to
the coordinator as "the same machine" instead of registering as a duplicate.

## Using it day to day

- **Queue a job**: pick `train_diffusion.py` or `train_mlm.py`, paste the arguments you'd
  normally pass on the command line (must include `--output_dir`), set a priority if you
  want some jobs to jump the queue, and submit. It's picked up by the next machine that
  goes idle.
- **Morning check-in**: the dashboard shows every machine's status (idle/busy/offline)
  and every job's status and log tail — no more logging into each box individually.
- **A machine crashes or gets turned off**: the coordinator notices heartbeats stop
  within ~45s, marks its job `crashed`, and (for `train_diffusion.py` jobs, up to 3
  retries by default) automatically re-queues it with `--auto_resume` so it picks back
  up on the next available machine from the last saved checkpoint.
- **You want to free up a machine on purpose**: click **checkpoint & stop** on the
  running job. It saves within one training step and the job goes to `paused`; click
  **resume** whenever you want to send it back into the queue.
- **You want the results off the machine**: click **fetch files** on a finished or
  paused job. The worker zips that job's `output_dir` and uploads it; a **download**
  link appears on the dashboard once it's ready.

## What happens on a crash or reboot

- **Worker machine dies mid-job**: coordinator marks the job `crashed` and, for
  `train_diffusion.py`, auto-requeues it (with `--auto_resume`) up to `max_retries`
  times (default 3, configurable per job if you edit the request — exposed as a field
  in `/api/jobs` if you want to script job submission).
- **Coordinator machine restarts**: `state.json` next to `server.py` has the full queue
  and history, so nothing is lost. Any job that was `running`/`assigned` when it went
  down is put back to `queued` on restart (the coordinator can't know if that job is
  still actually running somewhere), so it may briefly double-run if the worker was in
  fact still alive — the worker's next poll will report its real state and reconcile
  this within one poll interval (~5s).
- **Worker agent restarts**: re-registers using its saved `worker_id`, so it's treated
  as the same machine, not a new one.

## A few things worth knowing before you rely on this

- **No authentication.** Anyone who can reach the coordinator's port can queue, cancel,
  or fetch files from any job. Fine on a closed lab network; don't expose this port
  outside it.
- **Checkpoint-and-stop resumes at the last completed epoch**, not the exact batch —
  the patched script only checkpoints at epoch granularity (matching how periodic
  checkpointing already worked), so a stop mid-epoch discards that partial epoch's
  progress on resume, same as if the machine had just crashed at that point.
- **All workers are assumed to share the same repo checkout structure** (same relative
  script paths, same environment). The coordinator doesn't inspect or validate a
  machine's Python environment or CUDA setup before assigning it a job — if a job
  crashes immediately, check that machine's log tail on the dashboard first.
- I didn't wire up `train-diffusion.bat`'s later stages (sample generation, caption
  evaluation, MLM/tile-embedding pretraining) — you mentioned focusing on
  `train_diffusion.py` (and optionally `train_mlm.py`) was enough, so the orchestrator
  runs those two scripts directly rather than the full pipeline `.bat`. If you do want a
  queued job to run the full pipeline later, the same `--auto_resume`-style approach
  would need to go into whichever script is the long pole in that pipeline.

## Files in this bundle

```
coordinator/
  server.py            FastAPI backend + job queue + state persistence
  static/               dashboard (index.html / style.css / app.js)
  requirements.txt
worker/
  agent.py              runs on each lab machine
  requirements.txt
patched_scripts/
  train_diffusion.py    your script + STOP_REQUEST/--auto_resume support (see above)
README.md               this file
```
