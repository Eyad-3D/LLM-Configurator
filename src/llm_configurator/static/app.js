"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="session-token"]').content;
const GiB = (n) => (n / 1073741824).toFixed(1);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let report = null;
async function api(path, body) {
  const options = { headers: { "X-Session-Token": token } };
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok)
    throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}
function message(text, error = false) {
  $("message").textContent = text;
  $("message").className = error ? "error" : "";
}
function metric(label, value, sub) {
  return `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="sub" title="${esc(sub)}">${esc(sub)}</div></div>`;
}
function hardware(hw) {
  const gpu = hw.gpus[0];
  $("hardware").innerHTML =
    metric(
      "Available system RAM",
      `${GiB(hw.ram_available)} GiB`,
      `of ${GiB(hw.ram_total)} GiB installed`,
    ) +
    metric(
      "Available GPU memory",
      gpu ? `${GiB(gpu.available)} GiB` : "Unknown",
      gpu
        ? `${gpu.name} · ${GiB(gpu.total)} GiB total`
        : "No supported NVIDIA telemetry",
    ) +
    metric(
      "CPU activity",
      `${Math.round(hw.cpu_percent)}%`,
      `${hw.cores ?? "?"} cores · ${hw.threads ?? "?"} threads · ${hw.cpu}`,
    ) +
    metric(
      "Free disk space",
      `${GiB(hw.disk_free)} GiB`,
      "Home drive · checked again before downloading",
    );
  const previousGPU = $("gpu_index").value;
  $("gpu_index").innerHTML = hw.gpus.length
    ? hw.gpus
        .map(
          (g) =>
            `<option value="${g.index}">${esc(g.name)} (${g.index})</option>`,
        )
        .join("")
    : '<option value="0">CPU only — GPU telemetry unavailable</option>';
  if ([...$("gpu_index").options].some((o) => o.value === previousGPU))
    $("gpu_index").value = previousGPU;
  $("processes").innerHTML = hw.processes.length
    ? hw.processes
        .map(
          (p) =>
            `<label class="process-row" title="PID ${p.pid}; RSS ${GiB(p.rss)} GiB"><input type="checkbox" value="${p.pid}" ${p.reclaimable == null ? "disabled" : ""}><span class="process-name">${esc(p.name)}</span><span class="process-size">${p.reclaimable == null ? "unknown" : "~" + GiB(p.reclaimable) + " GiB"}</span></label>`,
        )
        .join("")
    : '<p class="hint">No sizeable accessible processes found.</p>';
}
async function loadState() {
  const state = await api("/api/state");
  hardware(state.hardware);
  $("demo").hidden = !state.demo;
  $("catalogue_status").textContent = state.status
    ? `${state.status.variants} variants cached · Last refresh ${new Date(state.status.timestamp).toLocaleString()}. ${(state.status.warnings || []).join(" · ")}`
    : "No metadata cached yet. Refresh to retrieve available GGUF variants.";
  $("mappings").innerHTML = state.definitions
    .map((entry, index) => {
      const options = state.scores
        .map(
          (score) =>
            `<option value="${esc(score.slug)}" ${entry.aa_slug === score.slug ? "selected" : ""}>${esc(score.name)} · ${esc(score.slug)}</option>`,
        )
        .join("");
      return `<div class="mapping"><label for="map-${index}">${esc(entry.base_repo)}</label><select id="map-${index}"><option value="">Unmapped — quality unknown</option>${options}</select><button class="secondary" data-map="${index}">Save match</button></div>`;
    })
    .join("");
  document.querySelectorAll("[data-map]").forEach((button) =>
    button.addEventListener("click", async () => {
      const index = Number(button.dataset.map);
      try {
        await api("/api/map", {
          base_repo: state.definitions[index].base_repo,
          slug: $(`map-${index}`).value || null,
        });
        message(
          "Benchmark mapping saved. Compare again to update the ranking.",
        );
      } catch (error) {
        message(error.message, true);
      }
    }),
  );
  return state;
}
function requirements() {
  return {
    workload: $("workload").value,
    context: Number($("context").value),
    users: Number($("users").value),
    min_tps: Number($("min_tps").value),
    reserve_gib: Number($("reserve_gib").value),
    gpu_reserve_gib: Number($("gpu_reserve_gib").value),
    gpu_index: Number($("gpu_index").value),
    strict_speed: $("strict_speed").checked,
    reclaim_pids: [...$("processes").querySelectorAll("input:checked")].map(
      (input) => Number(input.value),
    ),
  };
}
function renderResults() {
  if (!report) return;
  const all = $("show_all").checked;
  const shortlist = new Set(report.shortlist);
  const items = all
    ? report.candidates
    : report.candidates.filter((c) => shortlist.has(c.id));
  $("count").textContent =
    `${report.candidates.length} estimated memory fits · ${items.length} shown`;
  $("cards").innerHTML = items.length
    ? items
        .map((c) => {
          const score =
            c.quality_score == null
              ? "No quality score"
              : `Base-model ${c.quality_metric} index: ${c.quality_score.toFixed(1)}`;
          return `<article class="card"><div class="card-top"><div class="card-title"><h3>${esc(c.name)}</h3><span class="quant">${esc(c.quant)}</span></div><span class="tag ${c.speed_meets_target ? "" : "unknown"}">${c.speed_meets_target ? "Speed verified locally" : "Speed unverified"}</span></div>
      <p class="mode">${esc({ gpu: "GPU-resident estimate", split: "GPU + CPU split", cpu: "CPU execution" }[c.mode])} · ${c.gpu_layers}/${c.total_layers} layers on GPU · ${c.scenario === "now" ? "Current resources" : "After closing selected apps"}</p>
      <div class="card-metrics"><div><strong>${c.context.toLocaleString()}</strong><span>context tokens / user</span></div><div><strong>${GiB(c.ram_bytes)} GiB</strong><span>estimated system RAM</span></div><div><strong>${GiB(c.vram_bytes)} GiB</strong><span>estimated VRAM</span></div><div><strong>${c.tps == null ? "Not tested" : c.tps.toFixed(1) + " tok/s"}</strong><span>${c.tps == null ? "local benchmark needed" : "synthetic generation speed"}</span></div></div>
      <div class="card-bottom"><p>${esc(score)}${c.quality_score == null ? ". Memory-only comparison." : ". Exact quantisation quality unknown."}<br>${GiB(c.ram_headroom_bytes)} GiB RAM left beyond your reserve.</p><button class="text-button" data-detail="${esc(c.id)}">View configuration ↗</button></div></article>`;
        })
        .join("")
    : `<div class="empty"><h3>No qualifying configurations yet.</h3><p>${report.demo ? "Try reducing context or active users, or include unverified speed options." : "Refresh model metadata first. If models are cached, try a shorter context, fewer active users, or include unverified speed options."}</p><p class="hint">Rejections: ${report.rejected.context} context · ${report.rejected.memory} memory · ${report.rejected.speed} speed</p></div>`;
  $("notes").innerHTML = report.notes
    .map((note) => `<p>· ${esc(note)}</p>`)
    .join("");
  document
    .querySelectorAll("[data-detail]")
    .forEach((button) =>
      button.addEventListener("click", () =>
        detail(report.candidates.find((c) => c.id === button.dataset.detail)),
      ),
    );
  $("export").disabled = false;
}
function detail(c) {
  const settings = {
    model_repository: c.repo,
    model_revision: c.revision,
    model_file: c.filename,
    runtime: "llama.cpp",
    gpu_layers: c.runtime_gpu_layers,
    gpu_index: c.gpu_index,
    context_per_user: c.context,
    parallel_sequences: c.users,
    kv_cache: "f16",
    threads: c.threads,
  };
  const variantArgument = `'${c.variant_id.replace(/'/g, "'\\''")}'`;
  $("detail_content").innerHTML =
    `<p class="eyebrow">DEPLOYMENT DETAILS</p><h3>${esc(c.name)} · ${esc(c.quant)}</h3><p>${esc(c.explanation)}</p>
    <dl><dt>Memory-only context ceiling</dt><dd>${c.memory_max_context.toLocaleString()} tokens / user</dd><dt>FP16 KV cache, all users</dt><dd>${GiB(c.kv_bytes)} GiB</dd><dt>Weight file size</dt><dd>${GiB(c.file_bytes)} GiB</dd><dt>Quality evidence</dt><dd>${esc(c.quality_evidence)}</dd><dt>Evaluation entry</dt><dd>${esc(c.score_settings || "Unmapped")}</dd><dt>Benchmark version</dt><dd>${esc(c.score_version || "Unavailable")}</dd><dt>Metadata retrieved</dt><dd>${esc(c.metadata_date)}</dd></dl>
    ${c.score_source ? `<p>Source: <a href="${esc(c.score_source)}" target="_blank" rel="noreferrer">Artificial Analysis</a>. Scores apply to the evaluation entry above.</p>` : ""}
    <h3>Runtime settings</h3><pre>${esc(JSON.stringify(settings, null, 2))}</pre><p>Use these settings in your llama.cpp installation. The memory ceiling is an estimate; it is not a validated context or speed guarantee.</p>
    ${c.demo ? "<p>Demo fixtures cannot be downloaded or benchmarked. Start without --demo and refresh metadata for real models.</p>" : `<h3>Download & measure</h3><p>Run these commands locally after installing llama.cpp. Downloads require confirmation. Benchmarking generates 128 tokens near the selected context limit.</p><pre>${esc(`llm-config download ${variantArgument} --directory ./models\n\nllm-config bench ${variantArgument} --model './models/${c.filename.split("/").pop()}' --context ${c.context} --gpu-layers ${c.gpu_layers} --gpu-index ${c.gpu_index ?? 0}`)}</pre><p>Commands use shell quoting compatible with Bash and PowerShell for these model IDs. ${c.users > 1 ? "This benchmark measures one active user only; it will not verify your concurrent speed target." : "Compare again after the benchmark to apply the measured result."}</p>`}`;
  $("detail").showModal();
}
$("close_detail").addEventListener("click", () => $("detail").close());
$("requirements").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("compare").disabled = true;
  message("Scanning current resources and comparing configurations…");
  try {
    report = await api("/api/recommend", requirements());
    renderResults();
    message(
      `Comparison updated at ${new Date(report.hardware.timestamp).toLocaleTimeString()}.`,
    );
  } catch (error) {
    message(error.message, true);
  } finally {
    $("compare").disabled = false;
  }
});
$("scan").addEventListener("click", async () => {
  try {
    await loadState();
    report = null;
    $("export").disabled = true;
    $("cards").innerHTML =
      '<div class="empty"><h3>Hardware rescanned.</h3><p>Compare again to update your configurations.</p></div>';
    $("count").textContent = "Hardware updated";
    $("notes").textContent = "";
    message(
      "Hardware updated. Application selections reset to avoid stale process IDs.",
    );
  } catch (error) {
    message(error.message, true);
  }
});
$("show_all").addEventListener("change", renderResults);
$("refresh").addEventListener("click", async () => {
  $("refresh").disabled = true;
  try {
    await api("/api/refresh", {});
    message(
      "Refreshing metadata from Hugging Face and Artificial Analysis. No weights are downloaded.",
    );
    async function poll() {
      try {
        const state = await api("/api/refresh");
        if (state.running) {
          setTimeout(poll, 1200);
          return;
        }
        await loadState();
        message(
          `Metadata refresh finished. ${(state.result?.warnings || []).join(" · ")}`,
        );
        $("refresh").disabled = false;
      } catch (error) {
        message(error.message, true);
        $("refresh").disabled = false;
      }
    }
    setTimeout(poll, 1000);
  } catch (error) {
    message(error.message, true);
    $("refresh").disabled = false;
  }
});
$("export").addEventListener("click", () => {
  if (!report) return;
  // Exclude process identities from the user-facing export by default.
  const exportReport = JSON.parse(JSON.stringify(report));
  delete exportReport.hardware.processes;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(exportReport, null, 2)], {
      type: "application/json",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "llm-configurations.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
loadState().catch((error) => message(error.message, true));

async function credentialStatus() {
  const state = await api("/api/credentials");
  const labels = {
    saved: "Key saved securely in your OS credential store.",
    session:
      "Using a session-only key. It will be forgotten when the app closes.",
    environment:
      "Using AA_API_KEY from the environment. Remove it there to disable it.",
    none: "No API key configured.",
  };
  $("key-status").textContent =
    labels[state.source] || "Credential status unavailable.";
}
async function credentialAction(action) {
  const buttons = ["test-key", "save-key", "remove-key"];
  buttons.forEach((id) => ($(id).disabled = true));
  $("key-status").textContent =
    action === "test" ? "Testing connection…" : "Updating credentials…";
  try {
    const body =
      action === "remove"
        ? {}
        : { key: $("aa-api-key").value, remember: $("remember-key").checked };
    const result = await api("/api/credentials/" + action, body);
    if (action === "test") {
      $("key-status").textContent =
        result.message +
        ($("aa-api-key").value ? " Click Save key to apply this key." : "");
    } else {
      $("aa-api-key").value = "";
      await credentialStatus();
      message(
        action === "save"
          ? "API key applied. Refresh model metadata to retrieve rankings."
          : "App credentials cleared. An existing AA_API_KEY environment variable remains active.",
      );
    }
  } catch (error) {
    $("key-status").textContent = error.message;
  } finally {
    buttons.forEach((id) => ($(id).disabled = false));
  }
}
$("api-key-form").addEventListener("submit", (event) => {
  event.preventDefault();
  credentialAction("save");
});
$("test-key").addEventListener("click", () => credentialAction("test"));
$("remove-key").addEventListener("click", () => credentialAction("remove"));
credentialStatus().catch(() => {
  $("key-status").textContent =
    "Could not read credential status. Try reloading the app.";
});
