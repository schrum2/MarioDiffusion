"""
Diffusion-lab coordinator.

Run this on the machine you want to orchestrate training from:

    pip install -r requirements.txt
    python server.py --host 0.0.0.0 --port 8000

Then open http://<this machine's IP>:8000 in a browser, and point each
worker's agent.py at that same IP/port (see ../worker/agent.py).

State (workers, jobs, queue) is kept in memory and mirrored to a JSON file
(state.json next to this script, override with --state-file) after every
change, so restarting the coordinator does not lose the job queue or
history. Workers re-register on their own; nothing on the worker side is
lost either.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import shlex
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --------------------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------------------
HEARTBEAT_TIMEOUT_SEC = 45       # worker considered offline if no poll in this long
LOG_TAIL_MAX_LINES = 200
VALID_SCRIPTS = {"train_diffusion", "train_mlm"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "crashed", "cancelled", "paused"}

STATE_LOCK = threading.RLock()
BASE_DIR = Path(__file__).parent
DOWNLOAD_DIR = BASE_DIR / "fetched_files"
DOWNLOAD_DIR.mkdir(exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------
# State: plain dicts, persisted to JSON. Kept intentionally simple (no ORM/DB) since a
# lab of a few dozen machines and a modest job queue does not need one, and a flat JSON
# file is trivial to inspect or hand-edit if something ever needs a manual fix.
# --------------------------------------------------------------------------------------
class State:
    def __init__(self, path: Path):
        self.path = path
        self.workers = {}   # worker_id -> dict
        self.jobs = {}       # job_id -> dict
        self.job_order = []  # job ids in creation order, for stable queue FIFO
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.workers = data.get("workers", {})
                self.jobs = data.get("jobs", {})
                self.job_order = data.get("job_order", list(self.jobs.keys()))
                # Any worker that was mid-job when the coordinator went down is stale;
                # any job that was running/assigned is now orphaned. Reconcile on load.
                for j in self.jobs.values():
                    if j["status"] in ("assigned", "running", "checkpoint_stop_requested", "cancel_requested"):
                        j["status"] = "queued"
                        j["worker_id"] = None
                        j["pending_control"] = None
                        j["log_tail"].append("[coordinator] state reloaded after restart; job re-queued.")
                for w in self.workers.values():
                    w["status"] = "offline"
                    w["current_job_id"] = None
            except Exception as e:
                print(f"WARNING: could not load state file {self.path}: {e}. Starting fresh.")

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"workers": self.workers, "jobs": self.jobs, "job_order": self.job_order},
            indent=2, default=str,
        ))
        tmp.replace(self.path)


STATE: Optional[State] = None
SHARED_SECRET_PHRASE = None


def derive_secret_key(key_phrase: str) -> bytes:
    return hashlib.sha256(key_phrase.encode("utf-8")).digest()


def xor_stream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    out = bytearray()
    nonce_len = len(nonce)
    for i, b in enumerate(data):
        k = key[(i + nonce_len) % len(key)] ^ nonce[i % nonce_len]
        out.append(b ^ k)
    return bytes(out)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _from_b64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def build_auth_headers(secret_phrase: str, method: str, path: str, body: bytes = b"", status_code: Optional[int] = None):
    if not secret_phrase:
        return {}
    nonce = os.urandom(16)
    key = derive_secret_key(secret_phrase)
    payload = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }
    cipher = xor_stream(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"), key, nonce)
    tag = hmac.new(key, nonce + cipher + method.encode("utf-8") + path.encode("utf-8"), hashlib.sha256).digest()
    return {
        "x-auth-nonce": _b64(nonce),
        "x-auth-cipher": _b64(cipher),
        "x-auth-tag": _b64(tag),
    }


def verify_auth_headers(secret_phrase: str, method: str, path: str, body: bytes = b"", status_code: Optional[int] = None,
                       nonce: Optional[str] = None, cipher: Optional[str] = None, tag: Optional[str] = None):
    if not secret_phrase:
        return True
    if not nonce or not cipher or not tag:
        return False
    try:
        key = derive_secret_key(secret_phrase)
        nonce_bytes = _from_b64(nonce)
        cipher_bytes = _from_b64(cipher)
        expected_tag = hmac.new(key, nonce_bytes + cipher_bytes + method.encode("utf-8") + path.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected_tag, _from_b64(tag)):
            return False
        payload = json.loads(xor_stream(cipher_bytes, key, nonce_bytes).decode("utf-8"))
        if payload.get("method") != method or payload.get("path") != path:
            return False
        if status_code is not None and payload.get("status_code") is not None and payload.get("status_code") != status_code:
            return False
        if body and hashlib.sha256(body).hexdigest() != payload.get("body_sha256", ""):
            return False
        return True
    except Exception:
        return False


def parse_output_dir(args_list):
    """Pull --output_dir <value> out of a job's argument list, for display/dedupe."""
    for i, a in enumerate(args_list):
        if a == "--output_dir" and i + 1 < len(args_list):
            return args_list[i + 1]
        if a.startswith("--output_dir="):
            return a.split("=", 1)[1]
    return None


def worker_identity(name, slot=None):
    return (name, slot or "default")


def mark_worker_job_offline(worker):
    """Any active job on a dead/stale worker is no longer safe to consider running."""
    job_id = worker.get("current_job_id")
    if job_id and job_id in STATE.jobs:
        job = STATE.jobs[job_id]
        if job["status"] in ("assigned", "running", "checkpoint_stop_requested"):
            job["status"] = "crashed"
            job["finished_at"] = now_iso()
            job["exit_code"] = -1
            job["worker_id"] = None
            job["pending_control"] = None
            job["log_tail"].append(
                f"[coordinator] worker {worker['name']} stopped sending heartbeats; job marked crashed."
            )
            maybe_requeue(job)
    worker["current_job_id"] = None
    worker["status"] = "offline"


def next_queued_job_for(worker):
    """Lowest-priority-number, oldest, queued job. No matching on worker beyond
    'idle and script supported' -- every worker runs the same repo checkout so any
    worker can run any job."""
    with STATE_LOCK:
        candidates = [
            STATE.jobs[jid] for jid in STATE.job_order
            if STATE.jobs[jid]["status"] == "queued"
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda j: (j["priority"], j["created_at"]))
        return candidates[0]


def mark_offline_workers():
    with STATE_LOCK:
        cutoff = time.time() - HEARTBEAT_TIMEOUT_SEC
        for w in STATE.workers.values():
            if w["status"] != "offline" and w.get("_last_seen_epoch", 0) < cutoff:
                mark_worker_job_offline(w)

        # Also catch any stale workers that were already marked offline but still carry
        # an active job assignment from a crashed or restarted agent.
        for w in STATE.workers.values():
            if w.get("current_job_id") and w["current_job_id"] in STATE.jobs:
                job = STATE.jobs[w["current_job_id"]]
                if job["status"] in ("assigned", "running", "checkpoint_stop_requested"):
                    job["status"] = "crashed"
                    job["finished_at"] = now_iso()
                    job["exit_code"] = -1
                    job["worker_id"] = None
                    job["pending_control"] = None
                    job["log_tail"].append(
                        f"[coordinator] worker {w['name']} still had an active assignment while offline; job marked crashed."
                    )
                    maybe_requeue(job)
                    w["current_job_id"] = None


def maybe_requeue(job):
    """After a crash, requeue automatically if the job allows it and hasn't exceeded
    its retry budget. Requeued runs get --auto_resume added so they pick back up from
    the last checkpoint instead of erroring out on the existing output_dir."""
    if job["script"] != "train_diffusion":
        return  # train_mlm has no resume support upstream; do not auto-restart it blindly
    if not job.get("requeue_on_crash", True):
        return
    if job.get("retries", 0) >= job.get("max_retries", 3):
        job["log_tail"].append("[coordinator] retry budget exhausted; not auto-requeuing.")
        return
    job["retries"] = job.get("retries", 0) + 1
    job["status"] = "queued"
    job["worker_id"] = None
    job["pending_control"] = None
    if "--auto_resume" not in job["args"]:
        job["args"].append("--auto_resume")
    job["log_tail"].append(f"[coordinator] auto-requeued (attempt {job['retries']}/{job['max_retries']}).")


# --------------------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    worker_id: Optional[str] = None
    name: str
    repo_path: str
    gpu_info: Optional[str] = None
    slot: Optional[str] = None  # e.g. "gpu0" label for multi-GPU machines running several agents


class PollRequest(BaseModel):
    status: str  # "idle" | "busy"
    current_job_id: Optional[str] = None
    log_tail: Optional[list] = None
    gpu_info: Optional[str] = None


class JobCreateRequest(BaseModel):
    script: str
    args: list
    priority: int = 100
    requeue_on_crash: bool = True
    max_retries: int = 3


class JobReportRequest(BaseModel):
    event: str  # "started" | "progress" | "completed" | "failed" | "crashed" | "paused"
    exit_code: Optional[int] = None
    log_tail: Optional[list] = None
    pid: Optional[int] = None
    error: Optional[str] = None


class FetchUploadNotify(BaseModel):
    request_id: str


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
app = FastAPI(title="Diffusion Lab Coordinator")


@app.middleware("http")
async def auth_middleware(request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    if not SHARED_SECRET_PHRASE:
        response = await call_next(request)
        return response
    if request.method == "GET" and (
        request.url.path == "/api/state" or request.url.path.startswith("/api/fetch/")
    ):
        response = await call_next(request)
        for key, value in build_auth_headers(SHARED_SECRET_PHRASE, request.method, request.url.path, status_code=response.status_code).items():
            response.headers[key] = value
        return response

    body = await request.body()
    headers = request.headers
    if not verify_auth_headers(
        SHARED_SECRET_PHRASE,
        request.method,
        request.url.path,
        body=body,
        nonce=headers.get("x-auth-nonce"),
        cipher=headers.get("x-auth-cipher"),
        tag=headers.get("x-auth-tag"),
    ):
        return JSONResponse({"detail": "unauthorized: bad or missing shared key"}, status_code=401)

    response = await call_next(request)
    for key, value in build_auth_headers(
        SHARED_SECRET_PHRASE,
        request.method,
        request.url.path,
        status_code=response.status_code,
    ).items():
        response.headers[key] = value
    return response


@app.post("/api/workers/register")
def register_worker(req: RegisterRequest):
    with STATE_LOCK:
        identity = worker_identity(req.name, req.slot)
        existing = STATE.workers.get(req.worker_id) if req.worker_id else None
        if existing is None:
            for wid, w in list(STATE.workers.items()):
                if worker_identity(w.get("name"), w.get("slot")) == identity:
                    existing = w
                    req.worker_id = wid
                    break

        if existing is None:
            wid = str(uuid.uuid4())
        else:
            wid = existing["id"]

        # A restarted agent should replace earlier stale records for the same machine/slot.
        for stale_wid, stale in list(STATE.workers.items()):
            if stale_wid != wid and worker_identity(stale.get("name"), stale.get("slot")) == identity:
                del STATE.workers[stale_wid]

        state = {
            "id": wid,
            "name": req.name,
            "repo_path": req.repo_path,
            "gpu_info": req.gpu_info,
            "slot": req.slot,
            "status": "idle",
            "current_job_id": existing.get("current_job_id") if existing else None,
            "last_seen": now_iso(),
            "_last_seen_epoch": time.time(),
            "registered_at": existing.get("registered_at", now_iso()) if existing else now_iso(),
        }
        STATE.workers[wid] = state
        STATE.save()
        return {"worker_id": wid}


@app.post("/api/workers/{worker_id}/poll")
def poll(worker_id: str, req: PollRequest):
    with STATE_LOCK:
        if worker_id not in STATE.workers:
            raise HTTPException(404, "unknown worker_id; call /register again")
        w = STATE.workers[worker_id]
        w["status"] = req.status
        w["last_seen"] = now_iso()
        w["_last_seen_epoch"] = time.time()
        w["current_job_id"] = req.current_job_id
        if req.gpu_info:
            w["gpu_info"] = req.gpu_info

        response = {"job": None, "control": None, "fetch_request": None}

        # Relay any pending control instruction for the job this worker is running.
        if req.current_job_id and req.current_job_id in STATE.jobs:
            job = STATE.jobs[req.current_job_id]
            if req.log_tail:
                job["log_tail"] = (job.get("log_tail", []) + req.log_tail)[-LOG_TAIL_MAX_LINES:]
            if job.get("pending_control"):
                response["control"] = job["pending_control"]
            if job.get("pending_fetch"):
                response["fetch_request"] = job["pending_fetch"]

        # Hand out a new job only if the worker says it's idle (i.e. not mid-job).
        if req.status == "idle":
            job = next_queued_job_for(w)
            if job:
                job["status"] = "assigned"
                job["worker_id"] = worker_id
                job["assigned_at"] = now_iso()
                response["job"] = {
                    "job_id": job["id"],
                    "script": job["script"],
                    "args": job["args"],
                }

        STATE.save()
        return response


@app.post("/api/jobs")
def create_job(req: JobCreateRequest):
    if req.script not in VALID_SCRIPTS:
        raise HTTPException(400, f"script must be one of {sorted(VALID_SCRIPTS)}")
    output_dir = parse_output_dir(req.args)
    if not output_dir:
        raise HTTPException(400, "args must include --output_dir <path>")
    with STATE_LOCK:
        jid = str(uuid.uuid4())
        STATE.jobs[jid] = {
            "id": jid,
            "script": req.script,
            "args": list(req.args),
            "output_dir": output_dir,
            "status": "queued",
            "worker_id": None,
            "priority": req.priority,
            "requeue_on_crash": req.requeue_on_crash,
            "max_retries": req.max_retries,
            "retries": 0,
            "created_at": now_iso(),
            "assigned_at": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "pending_control": None,
            "pending_fetch": None,
            "log_tail": [],
            "resumed_from": None,
        }
        STATE.job_order.append(jid)
        STATE.save()
        return {"job_id": jid}


@app.post("/api/jobs/{job_id}/report")
def report_job(job_id: str, req: JobReportRequest):
    with STATE_LOCK:
        if job_id not in STATE.jobs:
            raise HTTPException(404, "unknown job")
        job = STATE.jobs[job_id]
        if req.log_tail:
            job["log_tail"] = (job.get("log_tail", []) + req.log_tail)[-LOG_TAIL_MAX_LINES:]

        if req.event == "started":
            job["status"] = "running"
            job["started_at"] = now_iso()
            job["pending_control"] = None
        elif req.event == "progress":
            pass
        elif req.event == "completed":
            job["status"] = "completed"
            job["finished_at"] = now_iso()
            job["exit_code"] = req.exit_code
            job["pending_control"] = None
            job["worker_id"] = None
        elif req.event == "paused":
            # Exit code STOP_REQUEST_EXIT_CODE (75): checkpoint saved on request, ready to resume.
            job["status"] = "paused"
            job["finished_at"] = now_iso()
            job["exit_code"] = req.exit_code
            job["pending_control"] = None
            job["worker_id"] = None
        elif req.event == "failed":
            job["status"] = "failed"
            job["finished_at"] = now_iso()
            job["exit_code"] = req.exit_code
            if req.error:
                job["log_tail"].append(f"[worker] error: {req.error}")
            job["pending_control"] = None
            job["worker_id"] = None
        elif req.event == "crashed":
            job["status"] = "crashed"
            job["finished_at"] = now_iso()
            job["exit_code"] = req.exit_code
            job["pending_control"] = None
            job["worker_id"] = None
            maybe_requeue(job)
        else:
            raise HTTPException(400, f"unknown event {req.event}")
        STATE.save()
        return {"ok": True}


@app.post("/api/jobs/{job_id}/checkpoint_stop")
def checkpoint_stop(job_id: str):
    with STATE_LOCK:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["script"] != "train_diffusion":
            raise HTTPException(400, "checkpoint-and-stop is only supported for train_diffusion.py jobs")
        if job["status"] not in ("running", "assigned"):
            raise HTTPException(400, f"job is not running (status={job['status']})")
        job["pending_control"] = "checkpoint_stop"
        job["status"] = "checkpoint_stop_requested"
        job["log_tail"].append("[coordinator] checkpoint-and-stop requested; will take effect within one training step.")
        STATE.save()
        return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with STATE_LOCK:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = now_iso()
            STATE.save()
            return {"ok": True}
        if job["status"] in TERMINAL_JOB_STATUSES:
            raise HTTPException(400, f"job already finished (status={job['status']})")
        job["pending_control"] = "cancel"
        job["log_tail"].append("[coordinator] hard cancel requested (no checkpoint will be saved).")
        STATE.save()
        return {"ok": True}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Clone a paused/crashed/failed job into a fresh queued job with --auto_resume set,
    so it lands on whichever worker is free next and continues from its last checkpoint."""
    with STATE_LOCK:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["script"] != "train_diffusion":
            raise HTTPException(400, "resume is only supported for train_diffusion.py jobs")
        if job["status"] not in ("paused", "crashed", "failed", "cancelled"):
            raise HTTPException(400, f"job is not in a resumable state (status={job['status']})")
        new_args = list(job["args"])
        if "--auto_resume" not in new_args:
            new_args.append("--auto_resume")
        jid = str(uuid.uuid4())
        STATE.jobs[jid] = {
            "id": jid,
            "script": job["script"],
            "args": new_args,
            "output_dir": job["output_dir"],
            "status": "queued",
            "worker_id": None,
            "priority": job["priority"],
            "requeue_on_crash": job.get("requeue_on_crash", True),
            "max_retries": job.get("max_retries", 3),
            "retries": 0,
            "created_at": now_iso(),
            "assigned_at": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "pending_control": None,
            "pending_fetch": None,
            "log_tail": [f"[coordinator] resumed from job {job_id}"],
            "resumed_from": job_id,
        }
        STATE.job_order.append(jid)
        STATE.save()
        return {"job_id": jid}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with STATE_LOCK:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] != "queued":
            raise HTTPException(400, "can only delete a job that is still queued; cancel it instead")
        job["status"] = "cancelled"
        job["finished_at"] = now_iso()
        STATE.save()
        return {"ok": True}


# ---- File fetch: pull a finished/paused job's output_dir back from the worker ---------
@app.post("/api/jobs/{job_id}/fetch")
def request_fetch(job_id: str):
    with STATE_LOCK:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if not job.get("worker_id"):
            raise HTTPException(400, "job has no known worker to fetch from")
        request_id = str(uuid.uuid4())
        job["pending_fetch"] = {
            "request_id": request_id,
            "output_dir": job["output_dir"],
            "status": "requested",
        }
        STATE.save()
        return {"request_id": request_id}


@app.post("/api/fetch/{request_id}/upload")
async def upload_fetch(request_id: str, job_id: str = Form(...), file: UploadFile = File(...)):
    with STATE_LOCK:
        job = STATE.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        dest = DOWNLOAD_DIR / f"{request_id}.zip"
        with open(dest, "wb") as f:
            f.write(await file.read())
        if job.get("pending_fetch", {}).get("request_id") == request_id:
            job["pending_fetch"]["status"] = "ready"
            job["pending_fetch"]["filename"] = dest.name
        STATE.save()
        return {"ok": True}


@app.get("/api/fetch/{request_id}/download")
def download_fetch(request_id: str):
    dest = DOWNLOAD_DIR / f"{request_id}.zip"
    if not dest.exists():
        raise HTTPException(404, "not ready yet")
    return FileResponse(dest, filename=dest.name, media_type="application/zip")


@app.get("/api/state")
def get_state():
    mark_offline_workers()
    with STATE_LOCK:
        return JSONResponse({
            "workers": list(STATE.workers.values()),
            "jobs": [STATE.jobs[jid] for jid in STATE.job_order],
            "server_time": now_iso(),
        })


# Serve the dashboard
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--state-file", default=str(BASE_DIR / "state.json"))
    ap.add_argument("--key-phrase", required=True, help="shared secret phrase that coordinator and worker agents must both know")
    args = ap.parse_args()

    global SHARED_SECRET_PHRASE
    SHARED_SECRET_PHRASE = args.key_phrase
    STATE = State(Path(args.state_file))

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
