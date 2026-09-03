"""
Diffusion-lab worker agent.

Run this on every lab machine that has the training repo checked out and a GPU:

    pip install -r requirements.txt
    python agent.py --coordinator http://<coordinator-ip>:8000 --repo-path C:\\path\\to\\repo

For a machine with multiple GPUs you want to use independently, run one agent process
with a comma-separated --gpu-ids list; the agent runs one poll/launch loop per GPU, each
pinned via CUDA_VISIBLE_DEVICES, and each shows up as its own worker in the dashboard:

    python agent.py --coordinator http://<coordinator-ip>:8000 --repo-path . --gpu-ids 0,1,2,3

Each slot remembers its coordinator-assigned worker_id in a small local file
(.worker_id_<slot>) next to this script, so it re-registers as "the same" worker after
a restart instead of showing up as a brand new, duplicate machine.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

POLL_INTERVAL_SEC = 5
HEARTBEAT_WHILE_RUNNING_SEC = 5
LOG_TAIL_LINES = 40
STOP_REQUEST_FILENAME = "STOP_REQUEST"   # must match train_diffusion.py
PAUSED_EXIT_CODE = 75                    # must match train_diffusion.py's STOP_REQUEST_EXIT_CODE

SCRIPT_ENTRYPOINTS = {
    "train_diffusion": "train_diffusion.py",
    "train_mlm": "train_mlm.py",
}


def log(slot_label, msg):
    print(f"[{time.strftime('%H:%M:%S')}][{slot_label}] {msg}", flush=True)


class JobRunner:
    """Owns at most one running subprocess at a time, for one GPU slot."""

    def __init__(self, args, worker_id, slot_label, jobs_log_path):
        self.args = args
        self.worker_id = worker_id
        self.slot_label = slot_label
        self.jobs_log_path = jobs_log_path  # job_id -> output_dir, persisted so /fetch
        self.session = requests.Session()
        self.proc = None
        self.current_job_id = None
        self.current_output_dir = None
        self.log_buffer = []
        self.log_lock = threading.Lock()
        self.job_output_dirs = self._load_jobs_log()

    # ---- persistence of job_id -> output_dir, so fetch still works after agent restarts ----
    def _load_jobs_log(self):
        if self.jobs_log_path.exists():
            try:
                return json.loads(self.jobs_log_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_job_output_dir(self, job_id, output_dir):
        self.job_output_dirs[job_id] = output_dir
        try:
            self.jobs_log_path.write_text(json.dumps(self.job_output_dirs))
        except Exception as e:
            log(self.slot_label, f"warning: could not persist jobs log: {e}")

    # ---- coordinator HTTP calls ----
    def api(self, method, path, **kwargs):
        url = self.args.coordinator.rstrip("/") + path
        resp = self.session.request(method, url, timeout=15, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def register(self):
        worker_id_file = Path(f".worker_id_{self.slot_label}")
        saved_id = worker_id_file.read_text().strip() if worker_id_file.exists() else None
        if self.args.gpu_id is not None:
            gpu_info = self.args.gpu_label or f"GPU {self.args.gpu_id}"
        else:
            gpu_info = "CPU/unspecified"
        result = self.api("POST", "/api/workers/register", json={
            "worker_id": saved_id,
            "name": f"{socket.gethostname()}:{self.slot_label}",
            "repo_path": str(self.args.repo_path),
            "gpu_info": gpu_info,
            "slot": self.slot_label,
        })
        self.worker_id = result["worker_id"]
        worker_id_file.write_text(self.worker_id)
        log(self.slot_label, f"registered as worker_id={self.worker_id}")

    def get_log_tail(self):
        with self.log_lock:
            return list(self.log_buffer[-LOG_TAIL_LINES:])

    def append_log(self, line):
        with self.log_lock:
            self.log_buffer.append(line.rstrip("\n"))
            if len(self.log_buffer) > 2000:
                self.log_buffer = self.log_buffer[-1000:]

    # ---- running a job ----
    def launch(self, job):
        script = SCRIPT_ENTRYPOINTS.get(job["script"])
        if script is None:
            self.report(job["job_id"], "failed", error=f"unknown script {job['script']}")
            return
        entry = self.args.repo_path / script
        if not entry.exists():
            self.report(job["job_id"], "failed", error=f"{entry} not found on this machine")
            return

        cmd = [self.args.python, "-u", str(script)] + list(job["args"])
        env = os.environ.copy()
        if self.args.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_id)

        output_dir_arg = None
        for i, a in enumerate(job["args"]):
            if a == "--output_dir" and i + 1 < len(job["args"]):
                output_dir_arg = job["args"][i + 1]
        output_dir = (self.args.repo_path / output_dir_arg) if output_dir_arg else None

        self.current_job_id = job["job_id"]
        self.current_output_dir = output_dir
        if output_dir_arg:
            self._save_job_output_dir(job["job_id"], output_dir_arg)
        with self.log_lock:
            self.log_buffer = []

        log(self.slot_label, f"starting job {job['job_id']}: {' '.join(cmd)}")
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.args.repo_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )
        except Exception as e:
            self.report(job["job_id"], "failed", error=f"could not launch: {e}")
            self.current_job_id = None
            return

        self.report(job["job_id"], "started", pid=self.proc.pid)

        reader = threading.Thread(target=self._stream_output, args=(job["job_id"],), daemon=True)
        reader.start()

        exit_code = self.proc.wait()
        reader.join(timeout=5)

        job_id = self.current_job_id
        self.current_job_id = None
        self.current_output_dir = None

        if exit_code == PAUSED_EXIT_CODE and job["script"] == "train_diffusion":
            log(self.slot_label, f"job {job_id} paused on checkpoint-stop request (exit {exit_code})")
            self.report(job_id, "paused", exit_code=exit_code, log_tail=self.get_log_tail())
        elif exit_code == 0:
            log(self.slot_label, f"job {job_id} completed normally")
            self.report(job_id, "completed", exit_code=exit_code, log_tail=self.get_log_tail())
        else:
            log(self.slot_label, f"job {job_id} exited with code {exit_code} (treated as crashed)")
            self.report(job_id, "crashed", exit_code=exit_code, log_tail=self.get_log_tail())

        self.proc = None

    def _stream_output(self, job_id):
        last_report = 0.0
        for line in self.proc.stdout:
            # Print directly to the worker's terminal/console
            print(f"[{self.slot_label}][JOB-OUT] {line}", end="", flush=True)

            # Store line for local logs and coordinator reports
            self.append_log(line)

            now = time.time()
            if now - last_report > HEARTBEAT_WHILE_RUNNING_SEC:
                last_report = now
                try:
                    self.report(job_id, "progress", log_tail=self.get_log_tail())
                except Exception as e:
                    log(self.slot_label, f"progress report failed (will retry): {e}")

    def report(self, job_id, event, exit_code=None, log_tail=None, pid=None, error=None):
        try:
            self.api("POST", f"/api/jobs/{job_id}/report", json={
                "event": event, "exit_code": exit_code,
                "log_tail": log_tail, "pid": pid, "error": error,
            })
        except Exception as e:
            log(self.slot_label, f"could not report {event} for job {job_id}: {e}")

    # ---- control instructions from the coordinator ----
    def apply_control(self, control):
        if control == "checkpoint_stop" and self.proc and self.current_output_dir:
            stop_file = self.current_output_dir / STOP_REQUEST_FILENAME
            if not stop_file.exists():
                log(self.slot_label, f"writing {stop_file} to request graceful checkpoint+stop")
                try:
                    self.current_output_dir.mkdir(parents=True, exist_ok=True)
                    stop_file.write_text("requested by coordinator\n")
                except Exception as e:
                    log(self.slot_label, f"could not write stop-request file: {e}")
        elif control == "cancel" and self.proc:
            log(self.slot_label, "hard-cancel requested; terminating process")
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception as e:
                log(self.slot_label, f"error terminating process: {e}")

    # ---- fetching a job's output_dir back to the coordinator ----
    def handle_fetch_request(self, fetch_req):
        request_id = fetch_req["request_id"]
        output_dir_arg = fetch_req["output_dir"]
        job_id = self.current_job_id
        output_dir = self.args.repo_path / output_dir_arg
        if not output_dir.exists():
            log(self.slot_label, f"fetch requested but {output_dir} does not exist on this machine")
            return
        log(self.slot_label, f"zipping {output_dir} for fetch request {request_id}")
        tmp_zip = Path(f"_fetch_{request_id}")
        try:
            archive_path = shutil.make_archive(str(tmp_zip), "zip", root_dir=str(output_dir))
            with open(archive_path, "rb") as f:
                self.session.post(
                    self.args.coordinator.rstrip("/") + f"/api/fetch/{request_id}/upload",
                    data={"job_id": job_id or ""},
                    files={"file": (f"{output_dir.name}.zip", f, "application/zip")},
                    timeout=600,
                )
            log(self.slot_label, f"uploaded {archive_path} for fetch request {request_id}")
        except Exception as e:
            log(self.slot_label, f"fetch upload failed: {e}")
        finally:
            try:
                os.remove(archive_path)
            except Exception:
                pass

    # ---- main loop ----
    def run(self):
        self.register()
        while True:
            try:
                status = "busy" if self.proc is not None else "idle"
                resp = self.api("POST", f"/api/workers/{self.worker_id}/poll", json={
                    "status": status,
                    "current_job_id": self.current_job_id,
                    "log_tail": self.get_log_tail() if self.proc is not None else None,
                })
            except Exception as e:
                log(self.slot_label, f"poll failed (coordinator unreachable?): {e}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            if resp.get("control"):
                self.apply_control(resp["control"])
            if resp.get("fetch_request"):
                threading.Thread(target=self.handle_fetch_request, args=(resp["fetch_request"],), daemon=True).start()

            if resp.get("job") and self.proc is None:
                self.launch(resp["job"])  # blocks this slot's loop until the job exits
                continue  # immediately poll again rather than sleeping

            time.sleep(POLL_INTERVAL_SEC)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coordinator", required=True, help="e.g. http://192.168.1.50:8000")
    ap.add_argument("--repo-path", required=True, type=Path, help="path to the training repo checkout on this machine")
    ap.add_argument("--python", default=sys.executable, help="python executable to run training scripts with")
    ap.add_argument("--gpu-ids", default=None, help="comma-separated GPU ids to run one slot per GPU, e.g. 0,1,2,3")
    ap.add_argument("--gpu-label", default=None, help="override the GPU label shown in the dashboard (single-GPU machines)")
    return ap.parse_args()


def main():
    args = parse_args()
    args.repo_path = args.repo_path.resolve()

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",")] if args.gpu_ids else [None]

    threads = []
    for gid in gpu_ids:
        slot_args = argparse.Namespace(**vars(args))
        slot_args.gpu_id = gid
        slot_label = f"gpu{gid}" if gid is not None else "default"
        jobs_log_path = Path(f".jobs_log_{slot_label}.json")
        runner = JobRunner(slot_args, worker_id=None, slot_label=slot_label, jobs_log_path=jobs_log_path)
        t = threading.Thread(target=runner.run, daemon=True, name=f"slot-{slot_label}")
        t.start()
        threads.append(t)
        time.sleep(0.5)  # stagger registration calls slightly

    print(f"Worker agent running with {len(threads)} slot(s). Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down (in-flight jobs on this machine keep running as orphan "
              "processes; the coordinator will mark them crashed once heartbeats stop "
              "and can auto-requeue them).")


if __name__ == "__main__":
    main()
