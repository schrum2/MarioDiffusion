const POLL_MS = 3000;

function timeAgo(iso) {
  if (!iso) return "—";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function splitArgs(text) {
  // Accept newline- or space-separated args; keep quoted strings together.
  const flat = text.replace(/\n/g, " ").trim();
  if (!flat) return [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  const out = [];
  let m;
  while ((m = re.exec(flat)) !== null) {
    out.push(m[1] ?? m[2] ?? m[3]);
  }
  return out;
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : null;
}

function renderWorkers(workers) {
  const grid = document.getElementById("workersGrid");
  document.getElementById("workerCount").textContent = `${workers.length} registered`;
  if (workers.length === 0) {
    grid.innerHTML = `<div class="empty-note">no machines registered yet — run worker/agent.py on a lab machine and point it at this coordinator.</div>`;
    return;
  }
  grid.innerHTML = workers.map(w => `
    <div class="worker-card">
      <div class="wname">${w.name}</div>
      <div class="wgpu">${w.gpu_info || "—"}</div>
      <div class="status-pill status-${w.status}">${w.status}</div>
      <div class="wseen">seen ${timeAgo(w.last_seen)}${w.repo_path ? " · " + w.repo_path : ""}</div>
    </div>
  `).join("");
}

function jobActions(job) {
  const actions = [];
  if (job.status === "queued") {
    actions.push(`<button class="small" data-action="delete" data-job="${job.id}">remove</button>`);
  }
  if (["running", "assigned", "checkpoint_stop_requested"].includes(job.status)) {
    if (job.script === "train_diffusion" && job.status !== "checkpoint_stop_requested") {
      actions.push(`<button class="small warn" data-action="checkpoint_stop" data-job="${job.id}">checkpoint &amp; stop</button>`);
    }
    actions.push(`<button class="small danger" data-action="cancel" data-job="${job.id}">cancel</button>`);
  }
  if (["paused", "crashed", "failed", "cancelled"].includes(job.status) && job.script === "train_diffusion") {
    actions.push(`<button class="small" data-action="resume" data-job="${job.id}">resume</button>`);
  }
  if (job.worker_id) {
    if (job.pending_fetch && job.pending_fetch.status === "ready") {
      actions.push(`<a class="small" style="border:1px solid var(--phosphor-dim);color:var(--phosphor);padding:4px 9px;border-radius:3px;text-decoration:none;font-family:var(--mono);font-size:11px" href="/api/fetch/${job.pending_fetch.request_id}/download">download</a>`);
    } else if (job.pending_fetch) {
      actions.push(`<span class="small" style="color:var(--text-dim)">zipping…</span>`);
    } else {
      actions.push(`<button class="small" data-action="fetch" data-job="${job.id}">fetch files</button>`);
    }
  }
  actions.push(`<button class="small" data-action="log" data-job="${job.id}">log</button>`);
  return actions.join("");
}

function renderJobs(jobs) {
  const list = document.getElementById("jobsList");
  document.getElementById("jobCount").textContent = `${jobs.length} total`;
  if (jobs.length === 0) {
    list.innerHTML = `<div class="empty-note">no jobs yet — queue one below.</div>`;
    return;
  }
  const byRecency = [...jobs].reverse();
  list.innerHTML = byRecency.map(job => {
    const worker = job.worker_id ? job.worker_id.slice(0, 8) : "—";
    return `
    <div class="job-row" data-job-row="${job.id}">
      <div class="jscript">${job.script}<br><span style="color:var(--text)">${job.output_dir}</span></div>
      <div class="jout">${(job.args || []).join(" ")}</div>
      <div><span class="status-pill status-${job.status}">${job.status}</span>${job.retries ? `<div style="color:var(--text-dim);font-size:10px;margin-top:4px">retry ${job.retries}/${job.max_retries}</div>` : ""}</div>
      <div class="jworker">worker ${worker}<br>${timeAgo(job.started_at || job.created_at)}</div>
      <div class="job-actions">${jobActions(job)}</div>
    </div>`;
  }).join("");
}

function openLog(job) {
  document.getElementById("logBoxTitle").textContent = `${job.script} — ${job.output_dir}`;
  document.getElementById("logBoxContent").textContent = (job.log_tail || []).join("\n") || "(no output yet)";
  document.getElementById("logOverlay").hidden = false;
}

let latestState = { workers: [], jobs: [] };

async function refresh() {
  try {
    latestState = await api("GET", "/api/state");
    renderWorkers(latestState.workers);
    renderJobs(latestState.jobs);
    const running = latestState.jobs.filter(j => j.status === "running").length;
    const online = latestState.workers.filter(w => w.status !== "offline").length;
    document.getElementById("topbarStats").textContent =
      `${online}/${latestState.workers.length} machines online · ${running} job(s) running`;
  } catch (e) {
    document.getElementById("topbarStats").textContent = "coordinator unreachable";
    console.error(e);
  }
}

document.getElementById("jobForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const script = form.script.value;
  const priority = parseInt(form.priority.value || "100", 10);
  const requeue = form.requeue_on_crash.checked;
  const args = splitArgs(form.args.value);
  const statusEl = document.getElementById("submitStatus");
  if (!args.includes("--output_dir")) {
    statusEl.textContent = "args must include --output_dir <path>";
    statusEl.style.color = "var(--red)";
    return;
  }
  try {
    const res = await api("POST", "/api/jobs", { script, args, priority, requeue_on_crash: requeue });
    statusEl.style.color = "var(--phosphor)";
    statusEl.textContent = `queued (${res.job_id.slice(0, 8)})`;
    form.reset();
    form.script.value = script;
    refresh();
  } catch (err) {
    statusEl.style.color = "var(--red)";
    statusEl.textContent = "error: " + err.message;
  }
});

document.getElementById("jobsList").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-action]");
  if (!btn) return;
  const action = btn.dataset.action;
  const jobId = btn.dataset.job;
  const job = latestState.jobs.find(j => j.id === jobId);

  if (action === "log") { openLog(job); return; }

  const confirmMsgs = {
    cancel: "Hard-cancel this job? No checkpoint will be saved for whatever progress hasn't already been saved.",
    delete: "Remove this queued job?",
  };
  if (confirmMsgs[action] && !confirm(confirmMsgs[action])) return;

  try {
    if (action === "checkpoint_stop") await api("POST", `/api/jobs/${jobId}/checkpoint_stop`);
    if (action === "cancel") await api("POST", `/api/jobs/${jobId}/cancel`);
    if (action === "resume") await api("POST", `/api/jobs/${jobId}/resume`);
    if (action === "fetch") await api("POST", `/api/jobs/${jobId}/fetch`);
    if (action === "delete") await api("DELETE", `/api/jobs/${jobId}`);
    refresh();
  } catch (err) {
    alert("action failed: " + err.message);
  }
});

document.getElementById("logBoxClose").addEventListener("click", () => {
  document.getElementById("logOverlay").hidden = true;
});
document.getElementById("logOverlay").addEventListener("click", (e) => {
  if (e.target.id === "logOverlay") document.getElementById("logOverlay").hidden = true;
});

refresh();
setInterval(refresh, POLL_MS);
