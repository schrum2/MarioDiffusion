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

function quoteArg(value) {
  const s = String(value);
  if (/^[A-Za-z0-9_./:-]+$/.test(s)) return s;
  return `"${s.replace(/"/g, '\\"')}"`;
}

function getGameDir(game) {
  if (game === "MM-Simple" || game === "MM-Full") return "Game_MM";
  return `Game_${game}`;
}

function getModelName(model) {
  return {
    MiniLM: "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    GTE: "Alibaba-NLP/gte-large-en-v1.5",
    CLIP: "sentence-transformers/clip-ViT-L-14",
    T5: "google/t5-v1_1-base"
  }[model];
}

function generateDiffusionCommand() {
  const game = document.getElementById("gameSelect").value;
  const data = document.getElementById("dataInput").value.trim();
  const type = document.getElementById("typeSelect").value;
  const model = document.getElementById("modelSelect").value;
  const split = document.getElementById("splitSelect").value;
  const tileMethod = document.getElementById("tileMethodSelect").value;
  const tileDim = document.getElementById("tileDimInput").value || "16";
  const seed = document.getElementById("seedInput").value || "0";
  const epochs = document.getElementById("epochsInput").value || "500";
  const numCaptions = document.getElementById("numCaptionsInput").value.trim();
  const captionKeys = document.getElementById("captionKeysInput").value.trim();

  if (!data) return "";

  const gameDir = getGameDir(game);
  const unconditional = type === "none";
  const useMLM = !unconditional && model === "MLM";

  let modelDir;

  const captionLimitTag =
    numCaptions && !unconditional
      ? `-captions${numCaptions}`
      : "";

  const tileTag =
    tileMethod !== "none"
      ? `${tileMethod}${tileDim}`
      : "";

  if (unconditional) {
    modelDir = tileMethod !== "none"
      ? `${game}-${data}-unconditional-${tileTag}-seed${seed}`
      : `${game}-${data}-unconditional-seed${seed}`;
  } else if (useMLM) {
    modelDir = tileMethod !== "none"
      ? `${game}-${data}-conditional-${tileTag}-${type}${captionLimitTag}-seed${seed}`
      : `${game}-${data}-conditional-${type}${captionLimitTag}-seed${seed}`;
  } else {
    const modelTag = `${model}-${split}`;

    modelDir = tileMethod !== "none"
      ? `${game}-${data}-conditional-${modelTag}-${tileTag}-${type}${captionLimitTag}-seed${seed}`
      : `${game}-${data}-conditional-${modelTag}-${type}${captionLimitTag}-seed${seed}`;
  }

  // Keep the output field synchronized with the generated default.
  const outputDir = document.getElementById("outputDirInput");

  if (!outputDir.dataset.manuallyEdited) {
    outputDir.value = modelDir;
  }

  const finalOutputDir = outputDir.value.trim() || modelDir;

  const trainDataType = unconditional ? "regular" : type;
  const dataPath =
    `${gameDir}/DATA/${data}_LevelsAndCaptions-${trainDataType}`;

  const trainData = `${dataPath}-train.json`;
  const valData = `${dataPath}-validate.json`;

  const args = [
    "--save_image_epochs", epochs,
    "--augment",
    "--output_dir", quoteArg(finalOutputDir),
    "--num_epochs", epochs,
    "--json", trainData,
    "--val_json", valData,
    "--seed", seed,
    "--game", game
  ];

  if (!unconditional) {
    args.push("--text_conditional");
  }

  // Tile embeddings.
  if (tileMethod !== "none") {
    const embeddingDir =
      `${game}-${data}-${tileMethod}${tileDim}-embeddings-seed${seed}`;

    args.push(
      "--block_embedding_model_path",
      quoteArg(embeddingDir)
    );
  }

  // 128 datasets require the smaller batch size.
  if (data.includes("128")) {
    args.push("--batch_size", "16");
  }

  // Conditional-only options.
  if (!unconditional) {
    if (numCaptions) {
      args.push("--captions_per_key", numCaptions);
    }

    if (captionKeys) {
      const keys = captionKeys
        .split(/[,\s]+/)
        .map(s => s.trim())
        .filter(Boolean);

      if (keys.length > 0) {
        args.push("--caption_source_keys", ...keys);
      }
    }

    if (type === "negative") {
      args.push("--negative_prompt_training");
    }

    if (type === "absence") {
      args.push("--describe_absence");
    }

    // The batch file suppresses the validation caption plot when
    // caption_source_keys are being used.
    if (!captionKeys) {
      args.push("--plot_validation_caption_score");
    }

    args.push("--plot_clip_score");

    if (useMLM) {
      const mlmOutput =
        `${game}-${data}-MLM-${type}-seed${seed}`;

      const tokenizer =
        `${gameDir}/DATA/${data}_Tokenizer-${type}.pkl`;

      args.push(
        "--pkl", tokenizer,
        "--mlm_model_dir", quoteArg(mlmOutput)
      );
    } else {
      const modelName = getModelName(model);

      args.push(
        "--pretrained_language_model",
        quoteArg(modelName)
      );

      if (split === "multiple") {
        args.push("--split_pretrained_sentences");
      }
    }
  }

  return args.join(" ");
}

function generateMlmCommand() {
  const game = document.getElementById("mlmGameSelect").value;
  const data = document.getElementById("mlmDataInput").value.trim();
  const type = document.getElementById("mlmTypeSelect").value;
  const seed = document.getElementById("mlmSeedInput").value || "0";
  const epochs = document.getElementById("mlmEpochsInput").value || "300";
  const checkpoint =
    document.getElementById("mlmCheckpointInput").value || "20";

  const gameDir = getGameDir(game);

  const trainData =
    `${gameDir}/DATA/${data}_LevelsAndCaptions-${type}-train.json`;
  const valData =
    `${gameDir}/DATA/${data}_LevelsAndCaptions-${type}-validate.json`;
  const testData =
    `${gameDir}/DATA/${data}_LevelsAndCaptions-${type}-test.json`;

  const tokenizer =
    `${gameDir}/DATA/${data}_Tokenizer-${type}.pkl`;

  const output =
    `${game}-${data}-MLM-${type}-seed${seed}`;

  const outputInput = document.getElementById("mlmOutputDirInput");

  if (!outputInput.dataset.manuallyEdited) {
    outputInput.value = output;
  }

  const finalOutput =
    outputInput.value.trim() || output;

  const args = [
    "--epochs", epochs,
    "--checkpoint_freq", checkpoint,
    "--save_checkpoints",
    "--json", trainData,
    "--val_json", valData,
    "--test_json", testData,
    "--pkl", tokenizer,
    "--output_dir", quoteArg(finalOutput),
    "--seed", seed
  ];

  if (game === "MM2") {
    args.push("--max_seq_length", "200");
  }

  return args.join(" ");
}

function generateCommand() {
  const script = document.getElementById("scriptSelect").value;

  if (script === "train_mlm") {
    return generateMlmCommand();
  }

  return generateDiffusionCommand();
}

function updateGeneratedCommand() {
  document.getElementById("argsInput").value = generateCommand();
}

function updateConfigVisibility() {
  const script = document.getElementById("scriptSelect").value;
  const diffusion = script === "train_diffusion";

  document.getElementById("diffusionConfig").hidden = !diffusion;
  document.getElementById("mlmConfig").hidden = diffusion;

  if (diffusion) {
    updateDiffusionVisibility();
  }

  updateGeneratedCommand();
}

function updateDiffusionVisibility() {
  const type = document.getElementById("typeSelect").value;
  const model = document.getElementById("modelSelect").value;
  const unconditional = type === "none";
  const useMLM = !unconditional && model === "MLM";

  document.getElementById("modelSelect").disabled = unconditional;
  document.getElementById("splitSelect").disabled =
    unconditional || useMLM;

  document.getElementById("splitField").style.opacity =
    unconditional || useMLM ? "0.45" : "1";

  document.getElementById("numCaptionsInput").disabled =
    unconditional;

  document.getElementById("captionKeysInput").disabled =
    unconditional || useMLM;

  document.getElementById("tileDimInput").disabled =
    document.getElementById("tileMethodSelect").value === "none";

  document.getElementById("tileDimField").style.opacity =
    document.getElementById("tileMethodSelect").value === "none"
      ? "0.45"
      : "1";
}

function bytesToBase64(bytes) {
  let binary = "";
  bytes.forEach(byte => { binary += String.fromCharCode(byte); });
  return btoa(binary);
}

function textBytes(text) {
  return new TextEncoder().encode(text);
}

function concatBytes(...parts) {
  const result = new Uint8Array(parts.reduce((length, part) => length + part.length, 0));
  let offset = 0;
  parts.forEach(part => {
    result.set(part, offset);
    offset += part.length;
  });
  return result;
}

async function buildAuthHeaders(method, path, bodyBytes) {
  const phrase = document.getElementById("keyPhraseInput").value;
  if (!phrase) return {};

  const key = new Uint8Array(await crypto.subtle.digest("SHA-256", textBytes(phrase)));
  const bodyHash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bodyBytes))]
    .map(byte => byte.toString(16).padStart(2, "0"))
    .join("");
  const nonce = crypto.getRandomValues(new Uint8Array(16));
  const payload = JSON.stringify({
    body_sha256: bodyHash,
    method,
    path,
    status_code: null
  });
  const payloadBytes = textBytes(payload);
  const cipher = new Uint8Array(payloadBytes.length);
  for (let i = 0; i < payloadBytes.length; i++) {
    cipher[i] = payloadBytes[i] ^ key[(i + nonce.length) % key.length] ^ nonce[i % nonce.length];
  }
  const hmacKey = await crypto.subtle.importKey(
    "raw", key, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const signingData = concatBytes(nonce, cipher, textBytes(method), textBytes(path));
  const tag = new Uint8Array(await crypto.subtle.sign("HMAC", hmacKey, signingData));

  return {
    "x-auth-nonce": bytesToBase64(nonce),
    "x-auth-cipher": bytesToBase64(cipher),
    "x-auth-tag": bytesToBase64(tag)
  };
}

async function api(method, path, body) {
  const opts = { method, headers: {} };

  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }

  Object.assign(
    opts.headers,
    await buildAuthHeaders(method, path, textBytes(opts.body || ""))
  );

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

  document.getElementById("workerCount").textContent =
    `${workers.length} registered`;

  if (workers.length === 0) {
    grid.innerHTML =
      `<div class="empty-note">no machines registered yet — run worker/agent.py on a lab machine and point it at this coordinator.</div>`;
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
    actions.push(
      `<button class="small" data-action="delete" data-job="${job.id}">remove</button>`
    );
  }

  if (["running", "assigned", "checkpoint_stop_requested"].includes(job.status)) {
    if (
      job.script === "train_diffusion" &&
      job.status !== "checkpoint_stop_requested"
    ) {
      actions.push(
        `<button class="small warn" data-action="checkpoint_stop" data-job="${job.id}">checkpoint &amp; stop</button>`
      );
    }

    actions.push(
      `<button class="small danger" data-action="cancel" data-job="${job.id}">cancel</button>`
    );
  }

  if (
    ["paused", "crashed", "failed", "cancelled"].includes(job.status) &&
    job.script === "train_diffusion"
  ) {
    actions.push(
      `<button class="small" data-action="resume" data-job="${job.id}">resume</button>`
    );
  }

  if (job.worker_id) {
    if (job.pending_fetch && job.pending_fetch.status === "ready") {
      actions.push(
        `<a class="small" style="border:1px solid var(--phosphor-dim);color:var(--phosphor);padding:4px 9px;border-radius:3px;text-decoration:none;font-family:var(--mono);font-size:11px" href="/api/fetch/${job.pending_fetch.request_id}/download">download</a>`
      );
    } else if (job.pending_fetch) {
      actions.push(
        `<span class="small" style="color:var(--text-dim)">zipping…</span>`
      );
    } else {
      actions.push(
        `<button class="small" data-action="fetch" data-job="${job.id}">fetch files</button>`
      );
    }
  }

  actions.push(
    `<button class="small" data-action="log" data-job="${job.id}">log</button>`
  );

  return actions.join("");
}

function renderJobs(jobs) {
  const list = document.getElementById("jobsList");

  document.getElementById("jobCount").textContent =
    `${jobs.length} total`;

  if (jobs.length === 0) {
    list.innerHTML =
      `<div class="empty-note">no jobs yet — queue one below.</div>`;
    return;
  }

  const byRecency = [...jobs].reverse();

  list.innerHTML = byRecency.map(job => {
    const worker = job.worker_id
      ? job.worker_id.slice(0, 8)
      : "—";

    return `
    <div class="job-row" data-job-row="${job.id}">
      <div class="jscript">
        ${job.script}<br>
        <span style="color:var(--text)">${job.output_dir}</span>
      </div>
      <div class="jout">${(job.args || []).join(" ")}</div>
      <div>
        <span class="status-pill status-${job.status}">
          ${job.status}
        </span>
        ${job.retries
          ? `<div style="color:var(--text-dim);font-size:10px;margin-top:4px">
               retry ${job.retries}/${job.max_retries}
             </div>`
          : ""}
      </div>
      <div class="jworker">
        worker ${worker}<br>
        ${timeAgo(job.started_at || job.created_at)}
      </div>
      <div class="job-actions">${jobActions(job)}</div>
    </div>`;
  }).join("");
}

function openLog(job) {
  document.getElementById("logBoxTitle").textContent =
    `${job.script} — ${job.output_dir}`;

  document.getElementById("logBoxContent").textContent =
    (job.log_tail || []).join("\n") || "(no output yet)";

  document.getElementById("logOverlay").hidden = false;
}

let latestState = { workers: [], jobs: [] };

async function refresh() {
  try {
    latestState = await api("GET", "/api/state");

    renderWorkers(latestState.workers);
    renderJobs(latestState.jobs);

    const running =
      latestState.jobs.filter(j => j.status === "running").length;

    const online =
      latestState.workers.filter(w => w.status !== "offline").length;

    document.getElementById("topbarStats").textContent =
      `${online}/${latestState.workers.length} machines online · ${running} job(s) running`;
  } catch (e) {
    document.getElementById("topbarStats").textContent =
      "coordinator unreachable";

    console.error(e);
  }
}


// ---------------------------------------------------------------------------
// Form handling
// ---------------------------------------------------------------------------

document.getElementById("scriptSelect").addEventListener(
  "change",
  updateConfigVisibility
);

[
  "gameSelect",
  "dataInput",
  "typeSelect",
  "modelSelect",
  "splitSelect",
  "tileMethodSelect",
  "tileDimInput",
  "seedInput",
  "epochsInput",
  "numCaptionsInput",
  "captionKeysInput",
  "mlmGameSelect",
  "mlmDataInput",
  "mlmTypeSelect",
  "mlmSeedInput",
  "mlmEpochsInput",
  "mlmCheckpointInput"
].forEach(id => {
  document.getElementById(id).addEventListener("input", () => {
    updateDiffusionVisibility();

    const output = document.getElementById("outputDirInput");
    const mlmOutput = document.getElementById("mlmOutputDirInput");

    // Only update generated output directories until the user explicitly
    // edits one of them.
    if (output && !output.matches(":focus")) {
      output.dataset.manuallyEdited = "";
    }

    if (mlmOutput && !mlmOutput.matches(":focus")) {
      mlmOutput.dataset.manuallyEdited = "";
    }

    updateGeneratedCommand();
  });

  document.getElementById(id).addEventListener("change", () => {
    updateDiffusionVisibility();
    updateGeneratedCommand();
  });
});

document.getElementById("outputDirInput").addEventListener("input", e => {
  e.target.dataset.manuallyEdited = "true";
  updateGeneratedCommand();
});

document.getElementById("mlmOutputDirInput").addEventListener("input", e => {
  e.target.dataset.manuallyEdited = "true";
  updateGeneratedCommand();
});

document.getElementById("regenerateButton").addEventListener(
  "click",
  () => {
    document.getElementById("outputDirInput").dataset.manuallyEdited = "";
    document.getElementById("mlmOutputDirInput").dataset.manuallyEdited = "";
    updateGeneratedCommand();
  }
);

document.getElementById("regenerateButtonBottom").addEventListener(
  "click",
  () => {
    document.getElementById("outputDirInput").dataset.manuallyEdited = "";
    document.getElementById("mlmOutputDirInput").dataset.manuallyEdited = "";
    updateGeneratedCommand();
  }
);

document.getElementById("jobForm").addEventListener(
  "submit",
  async (e) => {
    e.preventDefault();

    const form = e.target;
    const script = form.script.value;
    const priority = parseInt(
      form.priority.value || "100",
      10
    );
    const requeue = form.requeue_on_crash.checked;

    // IMPORTANT: argsInput is authoritative here.
    // The user can manually edit the generated command.
    const args = splitArgs(form.args.value);

    const statusEl =
      document.getElementById("submitStatus");

    if (!args.includes("--output_dir")) {
      statusEl.textContent =
        "command must include --output_dir <path>";

      statusEl.style.color = "var(--red)";
      return;
    }

    try {
      const res = await api(
        "POST",
        "/api/jobs",
        {
          script,
          args,
          priority,
          requeue_on_crash: requeue
        }
      );

      statusEl.style.color = "var(--phosphor)";
      statusEl.textContent =
        `queued (${res.job_id.slice(0, 8)})`;

      form.reset();

      // Restore useful defaults after reset.
      form.script.value = script;
      form.priority.value = "100";
      form.requeue_on_crash.checked = true;

      document.getElementById("gameSelect").value = "Mario";
      document.getElementById("dataInput").value = "regular";
      document.getElementById("typeSelect").value = "regular";
      document.getElementById("modelSelect").value = "MLM";
      document.getElementById("splitSelect").value = "single";
      document.getElementById("tileMethodSelect").value = "none";
      document.getElementById("tileDimInput").value = "16";
      document.getElementById("seedInput").value = "0";
      document.getElementById("epochsInput").value = "500";

      updateConfigVisibility();
      refresh();
    } catch (err) {
      statusEl.style.color = "var(--red)";
      statusEl.textContent =
        "error: " + err.message;
    }
  }
);


// ---------------------------------------------------------------------------
// Job actions
// ---------------------------------------------------------------------------

document.getElementById("jobsList").addEventListener(
  "click",
  async (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;

    const action = btn.dataset.action;
    const jobId = btn.dataset.job;
    const job = latestState.jobs.find(
      j => j.id === jobId
    );

    if (action === "log") {
      openLog(job);
      return;
    }

    const confirmMsgs = {
      cancel:
        "Hard-cancel this job? No checkpoint will be saved for whatever progress hasn't already been saved.",
      delete:
        "Remove this queued job?",
    };

    if (
      confirmMsgs[action] &&
      !confirm(confirmMsgs[action])
    ) {
      return;
    }

    try {
      if (action === "checkpoint_stop")
        await api(
          "POST",
          `/api/jobs/${jobId}/checkpoint_stop`
        );

      if (action === "cancel")
        await api(
          "POST",
          `/api/jobs/${jobId}/cancel`
        );

      if (action === "resume")
        await api(
          "POST",
          `/api/jobs/${jobId}/resume`
        );

      if (action === "fetch")
        await api(
          "POST",
          `/api/jobs/${jobId}/fetch`
        );

      if (action === "delete")
        await api(
          "DELETE",
          `/api/jobs/${jobId}`
        );

      refresh();
    } catch (err) {
      alert("action failed: " + err.message);
    }
  }
);


// ---------------------------------------------------------------------------
// Log overlay
// ---------------------------------------------------------------------------

document.getElementById("logBoxClose").addEventListener(
  "click",
  () => {
    document.getElementById("logOverlay").hidden = true;
  }
);

document.getElementById("logOverlay").addEventListener(
  "click",
  (e) => {
    if (e.target.id === "logOverlay")
      document.getElementById("logOverlay").hidden = true;
  }
);


// Initial state.
updateConfigVisibility();
refresh();
setInterval(refresh, POLL_MS);