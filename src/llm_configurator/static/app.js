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
// Model refreshes can return one warning per model (e.g. 40+ when offline). Show one
// plain sentence for the "couldn't reach" group and at most a few others.
function summarizeWarnings(list) {
  const items = (Array.isArray(list) ? list : []).map(String).filter(Boolean);
  const unreachable = items.filter((w) => /Metadata unavailable/i.test(w));
  const others = items.filter((w) => !/Metadata unavailable/i.test(w));
  const parts = [];
  if (unreachable.length)
    parts.push(
      `Couldn’t get the latest details for ${unreachable.length} model${unreachable.length === 1 ? "" : "s"} (is the internet connected?). Using what’s already saved on this computer.`,
    );
  parts.push(...others.slice(0, 3));
  if (others.length > 3) parts.push(`and ${others.length - 3} more notes`);
  return parts.join(" · ");
}
let report = null;
let appState = null;
let includeRankings = false;
async function api(path, body) {
  const options = { headers: { "X-Session-Token": token } };
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  // A crash page or proxy error may not be JSON; keep the status so callers can explain it.
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(
      payload.error || `Request failed: ${response.status}`,
    );
    error.status = response.status;
    throw error;
  }
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
  const selected = new Set(
    [...$("processes").querySelectorAll("input:checked")].map(
      (input) => `${input.value}:${input.dataset.created}`,
    ),
  );
  const gpus = Array.isArray(hw.gpus) ? hw.gpus : [];
  const others = Array.isArray(hw.other_gpus) ? hw.other_gpus : [];
  // Show the graphics chip the comparison will use (the one picked in Advanced settings).
  const wanted = Number($("gpu_index").value);
  const gpu = gpus.find((g) => g.index === wanted) || gpus[0];
  const known = (n) => n != null && Number.isFinite(Number(n));
  const gpuMetric = gpu
    ? metric(
        gpu.unified ? "Graphics memory (shared)" : "Available GPU memory",
        known(gpu.available) ? `${GiB(gpu.available)} GiB` : "Unknown",
        gpu.unified
          ? `${gpu.name} · Shared with system RAM, not extra memory`
          : `${gpu.name}${known(gpu.total) ? ` · ${GiB(gpu.total)} GiB total` : ""}`,
      )
    : metric(
        "Available GPU memory",
        others.length ? "Can’t be read" : "None found",
        others.length
          ? `${others[0].name || "A graphics chip"} found — free memory can’t be read, so estimates use RAM`
          : "No graphics card found — estimates use RAM",
      );
  // Graphics chips we can name but can't measure: say which, and why, so "uses RAM" isn't a mystery.
  const unread = others
    .map(
      (g) =>
        `<p class="hint">${gpus.length ? "Also found " : ""}${esc(g?.name || (gpus.length ? "a graphics chip" : "A graphics chip"))} (not counted in memory estimates): ${esc(g?.reason || "its free memory can’t be read.")}</p>`,
    )
    .join("");
  const warnings = (Array.isArray(hw.warnings) ? hw.warnings : [])
    .map((w) => `<p class="hint">${esc(w)}</p>`)
    .join("");
  $("hardware").innerHTML =
    metric(
      "Available system RAM",
      `${GiB(hw.ram_available)} GiB`,
      `of ${GiB(hw.ram_total)} GiB installed`,
    ) +
    gpuMetric +
    metric(
      "CPU activity",
      `${Math.round(hw.cpu_percent)}%`,
      `${hw.cores ?? "?"} cores · ${hw.threads ?? "?"} threads · ${hw.cpu_name || hw.cpu || "Processor"}`,
    ) +
    metric(
      "Free disk space",
      `${GiB(hw.disk_free)} GiB`,
      "Home drive · checked again before downloading",
    ) +
    unread +
    warnings;
  const previousGPU = $("gpu_index").value;
  $("gpu_index").innerHTML = gpus.length
    ? gpus
        .map(
          (g) =>
            `<option value="${esc(g.index)}">${esc(g.name)} (${esc(g.index)})</option>`,
        )
        .join("")
    : '<option value="0">Processor only — no usable graphics card found</option>';
  if ([...$("gpu_index").options].some((o) => o.value === previousGPU))
    $("gpu_index").value = previousGPU;
  $("processes").innerHTML = hw.processes.length
    ? [...hw.processes]
        .sort((a, b) => (b.rss ?? -1) - (a.rss ?? -1))
        .map(
          (p) =>
            `<label class="process-row" title="PID ${esc(p.pid)}; RSS ${esc(GiB(p.rss))} GiB"><input type="checkbox" value="${esc(p.pid)}" data-created="${esc(p.created)}" ${selected.has(`${p.pid}:${p.created}`) ? "checked" : ""} ${p.reclaimable == null ? "disabled" : ""}><span class="process-name">${esc(p.name)}</span><span class="process-size"><strong>${p.rss == null ? "Usage unknown" : GiB(p.rss) + " GiB used"}</strong><small>${p.reclaimable == null ? "Freeable memory unknown" : "~" + GiB(p.reclaimable) + " GiB could be freed"}</small></span></label>`,
        )
        .join("")
    : '<p class="hint">No sizeable accessible processes found.</p>';
}
async function loadState() {
  const state = await api("/api/state");
  appState = state;
  hardware(state.hardware);
  $("demo").hidden = !state.demo;
  $("catalogue_status").textContent = state.status
    ? `${state.status.variants} variants cached · Last refresh ${new Date(state.status.timestamp).toLocaleString()}. ${summarizeWarnings(state.status.warnings)}`
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
    priority: $("priority").value,
    include_rankings: includeRankings,
    context: Number($("context").value),
    users: Number($("users").value),
    min_tps: Number($("min_tps").value),
    reserve_gib: Number($("reserve_gib").value),
    gpu_reserve_gib: Number($("gpu_reserve_gib").value),
    gpu_index: Number($("gpu_index").value),
    strict_speed: $("strict_speed").checked,
    kv_cache_type: $("kv_cache_type")?.value || "f16",
    reclaim_pids:
      $("resource-mode").value === "free"
        ? [...$("processes").querySelectorAll("input:checked")].map((input) =>
            Number(input.value),
          )
        : [],
  };
}
function qualityPanel(c) {
  if (!report.requirements.include_rankings)
    return '<p class="hint">Rankings skipped · Comparing resources and available local speed evidence.</p>';
  const q = c.quality_comparison;
  const metric =
    {
      general: "General intelligence",
      coding: "Coding",
      agentic: "Agentic work",
    }[c.quality_metric] || c.quality_metric;
  if (!q || q.rank == null) {
    return `<section class="quality-panel quality-missing"><strong>Not ranked</strong><p>${q?.reason === "incomparable" ? "Benchmark versions or reference scores differ. Refresh and check the model mappings before comparing." : "No comparable benchmark score for this workload. Add your API key, refresh metadata and match the benchmark entry."}</p><button type="button" class="text-button" data-open-rank>Set up benchmark comparison →</button></section>`;
  }
  return `<section class="quality-panel"><div class="quality-heading"><div><span class="quality-label">${esc(metric)}</span><strong>${Number(c.quality_score).toFixed(1)} <small>index points</small></strong></div><div class="quality-rank"><strong>${q.tied ? "Joint " : ""}#${esc(q.rank)} <small>of ${esc(q.rated_models)} rated models</small></strong></div></div><p class="hint">Base-model benchmark · Exact quantisation quality is unmeasured.</p></section>`;
}
// The engine appends the verdict sentence to the explanation; the card already shows it above.
const when = (value) => {
  const date = value ? new Date(value) : null;
  return date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : value || "Unknown";
};
function explanationOf(c) {
  const text = String(c.explanation ?? "");
  return c.verdict_text && text.endsWith(c.verdict_text)
    ? text.slice(0, -c.verdict_text.length).trim()
    : text;
}
// Where the model runs. MoE models keep their "experts" (big, rarely used parts) in main memory.
function placementLabel(c) {
  if (c.mode === "split" && c.n_cpu_moe > 0) return "Experts on CPU";
  return { cpu: "CPU", gpu: "GPU", split: "GPU + CPU" }[c.mode] || "";
}
function speedPresentation(c) {
  if (c.tps != null)
    return {
      value: `${c.tps.toFixed(1)} tok/s`,
      label: c.evidence === "tuned" ? "Measured after tuning" : "Measured local generation speed",
      tag: "Speed verified locally",
    };
  // A short tune ran near the start of a conversation only, so it is not this length's speed.
  const quick = c.tuned?.tps;
  if (c.evidence === "tuned" && Number.isFinite(quick))
    return {
      value: `~${quick.toFixed(1)} tok/s`,
      label: "Tuned with a short test · not checked at this length yet",
      tag: "Speed not tested at this length",
    };
  const nearby = c.speed_interpolated?.tps;
  if (c.evidence === "interpolated" && Number.isFinite(nearby))
    return {
      value: `~${nearby.toFixed(1)} tok/s`,
      label: "Estimated from your tests at other lengths",
      tag: "Speed not tested at this length",
    };
  const crowd = c.community?.median_tps;
  if (c.evidence === "community" && Number.isFinite(crowd))
    return {
      value: `~${crowd.toFixed(1)} tok/s`,
      label: `Reported by ${c.community.n ?? "other"} similar computer${c.community.n === 1 ? "" : "s"}`,
      tag: "Speed from other people",
    };
  const e = c.speed_estimate;
  if (e?.available)
    return {
      value: `${e.low_tps.toFixed(1)}–${e.high_tps.toFixed(1)} tok/s`,
      label: "Estimated generation · low confidence",
      tag: {
        likely_meets: "Likely meets speed target",
        borderline: "Speed target borderline",
        likely_below: "Likely below speed target",
      }[e.target_status],
    };
  return {
    value: "Unavailable",
    label: e?.reason || "Hardware calibration or local benchmark needed",
    tag: "Speed unverified",
  };
}
// Every speed number we have for this card, each with where it came from, so a test, a tune,
// a guess from other lengths and other people's results never read the same.
function speedEvidence(c) {
  const n = (v) => Number.isFinite(v);
  const rows = [];
  if (n(c.tps))
    rows.push([
      c.evidence === "tuned" ? "Measured after tuning, on this computer" : "Measured on this computer",
      `${c.tps.toFixed(1)} tok/s`,
    ]);
  if (n(c.tuned?.tps) && !(c.evidence === "tuned" && n(c.tps)))
    rows.push([
      c.tuned.verified
        ? "Tuning run on this computer"
        : "Tuning run with a short test (near the start of a conversation)",
      `${c.tuned.tps.toFixed(1)} tok/s`,
    ]);
  if (n(c.speed_interpolated?.tps))
    rows.push(["Estimated from your tests at other lengths (not a test at this length)", `~${c.speed_interpolated.tps.toFixed(1)} tok/s`]);
  if (n(c.community?.median_tps)) {
    const count = Number.isInteger(c.community.n) ? c.community.n : null;
    rows.push([
      `Other people’s results: middle value from ${count ?? "several"} similar computer${count === 1 ? "" : "s"} (not this one)`,
      `~${c.community.median_tps.toFixed(1)} tok/s`,
    ]);
  }
  if (rows.length < 2) return "";
  return `<dl>${rows.map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join("")}</dl>`;
}
// Plain verdicts from the engine; older reports fall back to measured speed only.
const VERDICTS = {
  runs_well: ["good", "Runs well"],
  runs_slowly: ["slow", "Runs slowly"],
  too_slow: ["bad", "Too slow"],
  unknown: ["unknown", "Not tested yet"],
};
const EVIDENCE = {
  measured: "measured on this computer",
  tuned: "measured after tuning",
  short_tune: "tuned with a short test",
  interpolated: "estimated from nearby tests",
  community: "based on other people’s results",
  estimated: "estimate",
};
function verdictOf(c) {
  if (c.verdict && VERDICTS[c.verdict]) return c.verdict;
  if (c.tps != null) return c.speed_meets_target ? "runs_well" : "runs_slowly";
  return "unknown";
}
function verdictBadge(c) {
  const verdict = verdictOf(c);
  const [tone, label] = VERDICTS[verdict];
  // Anything not measured on this computer is labelled, so estimates never read as tests.
  const source =
    verdict === "unknown"
      ? ["interpolated", "community"].includes(c.evidence)
        ? c.evidence
        : c.evidence === "tuned" && c.tps == null
          ? "short_tune"
          : null
      : c.verdict
        ? c.evidence
        : "measured";
  const origin = source ? EVIDENCE[source] || EVIDENCE.estimated : null;
  const qualifier = origin && !["measured", "tuned"].includes(source) ? origin : null;
  const title = c.verdict_text || (origin ? `${label} (${origin})` : label);
  return `<span class="verdict verdict-${tone}" title="${esc(title)}">${esc(label)}${qualifier ? `<small> · ${esc(qualifier)}</small>` : ""}</span>`;
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
  const comparison = report.quality_comparison;
  $("comparison-summary").textContent = !report.requirements.include_rankings
    ? "Rankings are off for this comparison."
    : comparison
      ? `${comparison.rated_models} of ${comparison.eligible_models} distinct eligible models have workload scores. ${comparison.comparable ? "Ranks compare rated models across all fits, not just the shortlist. Card order follows your selected priority." : "Reference scores are not comparable; quality ranks are withheld."}`
      : "";
  $("cards").innerHTML = items.length
    ? items
        .map((c) => {
          const speed = speedPresentation(c);
          const score =
            c.quality_score == null
              ? "No quality score"
              : `Base-model ${c.quality_metric} index: ${c.quality_score.toFixed(1)}`;
          return `<article class="card"><div class="card-top"><div class="card-title"><h3>${esc(c.name)}</h3><span class="quant">${esc(c.quant)}</span></div><div class="card-tags">${verdictBadge(c)}${c.verdict ? "" : `<span class="tag ${c.speed_meets_target ? "" : "unknown"}">${esc(speed.tag)}</span>`}</div></div>${c.verdict_text ? `<p class="verdict-text">${esc(c.verdict_text)}</p>` : ""}
      ${qualityPanel(c)}
      <div class="card-metrics concise"><div><strong>${c.context.toLocaleString()}</strong><span>context tokens per session</span></div><div><strong>${esc(speed.value)}</strong><span>${esc(speed.label)}</span></div><div><strong>${esc(placementLabel(c))}</strong><span>${c.scenario === "now" ? "Fits current resources (estimated)" : "May fit after closing apps"}</span></div></div>
      <div class="card-bottom"><p>${esc(explanationOf(c))}</p><div class="card-actions"><button class="text-button" data-detail="${esc(c.id)}">View details ↗</button><button type="button" data-run="${esc(c.id)}">Get it running →</button></div></div></article>`;
        })
        .join("")
    : `<div class="empty"><h3>No qualifying configurations yet.</h3><p>${report.demo ? "Try reducing context or active users, or include unverified speed options." : "Refresh model metadata first. If models are cached, try a shorter context, fewer active users, or include unverified speed options."}</p><p class="hint">Rejections: ${esc(report.rejected?.context)} context · ${esc(report.rejected?.memory)} memory · ${esc(report.rejected?.speed)} speed</p></div>`;
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
  document.querySelectorAll("[data-run]").forEach((button) =>
    button.addEventListener("click", () =>
      openRunPanel(report.candidates.find((c) => c.id === button.dataset.run)),
    ),
  );
  document
    .querySelectorAll("[data-open-rank]")
    .forEach((button) => button.addEventListener("click", () => openRanking()));
  $("export").disabled = false;
}
function openRunPanel(c) {
  if (!c) return;
  if (!window.RunPanel) {
    message("The setup panel did not load. Reload the page and try again.", true);
    return;
  }
  window.RunPanel.open(c, {
    candidates: report.candidates,
    workload: report.requirements?.workload,
    demo: !!(report.demo || appState?.demo || c.demo),
  });
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
    kv_cache: c.kv_cache_type || "f16",
    experts_on_cpu_layers: c.n_cpu_moe || 0,
    threads: c.threads,
  };
  const variantArgument = `'${c.variant_id.replace(/'/g, "'\\''")}'`;
  $("detail_content").innerHTML =
    `<p class="eyebrow">DEPLOYMENT DETAILS</p><h3>${esc(c.name)} · ${esc(c.quant)}</h3><p>${esc(c.explanation)}</p>
    <dl><dt>Quality rank among rated eligible models</dt><dd>${c.quality_comparison?.rank == null ? "Not ranked" : "#" + esc(c.quality_comparison.rank) + " of " + esc(c.quality_comparison.rated_models)}</dd><dt>Gap to highest reference score</dt><dd>${c.quality_comparison?.points_behind_best == null ? "Unavailable" : esc(c.quality_comparison.points_behind_best) + " index points"}</dd>${c.memory_total_bytes != null ? `<dt>Estimated memory in total</dt><dd>${GiB(c.memory_total_bytes)} GiB</dd>` : ""}<dt>Estimated system RAM</dt><dd>${GiB(c.ram_bytes)} GiB</dd><dt>${c.unified_memory ? "Of which the graphics chip uses (shared with RAM)" : "Estimated VRAM"}</dt><dd>${GiB(c.vram_bytes)} GiB</dd><dt>RAM headroom after reserve</dt><dd>${GiB(c.ram_headroom_bytes)} GiB</dd><dt>Memory-only context ceiling</dt><dd>${c.memory_max_context.toLocaleString()} tokens / session</dd><dt>Conversation memory (${esc({ f16: "full size", q8_0: "half size", q4_0: "quarter size" }[c.kv_cache_type] || "full size")}), all users</dt><dd>${GiB(c.kv_bytes)} GiB</dd><dt>Weight file size</dt><dd>${GiB(c.file_bytes)} GiB</dd><dt>Quality evidence</dt><dd>${esc(c.quality_evidence)}</dd><dt>Evaluation entry</dt><dd>${esc(c.score_settings || "Unmapped")}</dd><dt>Benchmark version</dt><dd>${esc(c.score_version || "Unavailable")}</dd><dt>Model details downloaded</dt><dd>${esc(when(c.metadata_date))}</dd></dl>
    ${/^https:\/\//.test(c.score_source || "") ? `<p>Source: <a href="${esc(c.score_source)}" target="_blank" rel="noreferrer">Artificial Analysis</a>. Scores apply to the evaluation entry above.</p>` : ""}
    <h3>Speed evidence</h3><p>${esc(speedPresentation(c).value)} · ${esc(speedPresentation(c).label)}</p>${speedEvidence(c)}
    ${c.speed_estimate?.available ? `<p>${esc(c.speed_estimate.method)}. ${esc(c.speed_estimate.scope)}</p><p>${esc(c.speed_estimate.caveat)}</p><p>Calibrated: ${esc(c.speed_estimate.calibrated_at)}</p>` : ""}
    <h3>Runtime settings</h3><pre>${esc(JSON.stringify(settings, null, 2))}</pre><p>Use these settings in your llama.cpp installation. The memory ceiling is an estimate; it is not a validated context or speed guarantee.</p>
    ${c.demo ? "<p>Demo fixtures cannot be downloaded or benchmarked. Start without --demo and refresh metadata for real models.</p>" : `<h3>Download & measure</h3><p>Run these commands locally after installing llama.cpp. Downloads require confirmation. Benchmarking generates 128 tokens near the selected context limit.</p><pre>${esc(`llm-config download ${variantArgument} --directory ./models\n\nllm-config bench ${variantArgument} --model './models/${c.filename.split("/").pop()}' --context ${c.context} --gpu-layers ${c.gpu_layers} --gpu-index ${c.gpu_index ?? 0}`)}</pre><p>Commands use shell quoting compatible with Bash and PowerShell for these model IDs. ${c.users > 1 ? "This benchmark measures one active session only; it will not verify your concurrent speed target." : "Compare again after the benchmark to apply the measured result."}</p>`}`;
  $("detail").showModal();
}
$("close_detail").addEventListener("click", () => $("detail").close());
$("scan").addEventListener("click", async () => {
  try {
    await loadState();
    report = null;
    $("export").disabled = true;
    $("cards").innerHTML =
      '<div class="empty"><h3>Hardware rescanned.</h3><p>Compare again to update your configurations.</p></div>';
    $("count").textContent = "Hardware updated";
    $("comparison-summary").textContent = "";
    $("notes").textContent = "";
    message(
      "Hardware updated. Application selections reset to avoid stale process IDs.",
    );
  } catch (error) {
    message(error.message, true);
  }
});
$("show_all").addEventListener("change", renderResults);
function renderPreparationProgress(progress) {
  if (!progress) return;
  $("model-progress").textContent = progress.models_total
    ? `Model sources checked: ${progress.models_done} of ${progress.models_total}${progress.models_failed ? ` · ${progress.models_failed} unavailable` : ""}`
    : "Using prepared model data";
  $("score-progress").textContent =
    {
      running: "Benchmark rankings: retrieving…",
      complete: "Benchmark rankings: retrieved",
      failed:
        "Benchmark rankings: update unavailable; using cached data if available",
      skipped: "Benchmark rankings: skipped",
    }[progress.scores] || "Benchmark rankings: pending";
}
async function refreshMetadata(withScores, withModels = true) {
  // A different tab or manual refresh may own the server worker. Wait, then
  // submit our own scope so a model-only job cannot satisfy a ranking request.
  for (;;) {
    try {
      await api("/api/refresh", {
        include_scores: withScores,
        include_models: withModels,
      });
      break;
    } catch (error) {
      if (error.status !== 409) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const state = await api("/api/refresh");
    renderPreparationProgress(state.progress);
    if (!state.running) {
      await loadState();
      return state.result;
    }
  }
}
$("refresh").addEventListener("click", async () => {
  $("refresh").disabled = true;
  try {
    message("Refreshing model and benchmark metadata…");
    const result = await refreshMetadata(true);
    message(
      summarizeWarnings(result.warnings) ||
        "Metadata updated. You can now match benchmark entries.",
    );
  } catch (error) {
    message(error.message, true);
  } finally {
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
      document.dispatchEvent(new Event("credentials-changed"));
      message(
        action === "save"
          ? "API key applied. Continue with rankings to start background preparation."
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

// ---- community speed results (public list; downloading sends nothing about this computer) ----
function communitySummary(info) {
  const count = Array.isArray(info?.records) ? info.records.length : 0;
  if (!count) return "No shared results downloaded yet.";
  const when = info.fetched_at ? new Date(info.fetched_at) : null;
  return `${count.toLocaleString()} shared result${count === 1 ? "" : "s"} on this computer${when && !Number.isNaN(when.getTime()) ? `, downloaded ${when.toLocaleString()}` : ""}.`;
}
async function communityStatus() {
  try {
    $("community-status").textContent = communitySummary(
      await api("/api/community"),
    );
  } catch {
    $("community-status").textContent = "Couldn’t check for shared results.";
  }
}
async function communityImport() {
  const buttonNode = $("community-import");
  buttonNode.disabled = true;
  $("community-status").textContent = "Downloading shared results…";
  const finish = (job) => {
    buttonNode.disabled = false;
    if (job.state === "done") {
      const n = job.result?.count ?? 0;
      const skipped = job.result?.rejected;
      $("community-status").textContent =
        `Got ${Number(n).toLocaleString()} shared result${n === 1 ? "" : "s"}${skipped ? ` (${skipped} skipped because they looked wrong)` : ""}. Press Recalculate to use them.`;
    } else if (job.state === "cancelled")
      $("community-status").textContent = "Stopped. Nothing was changed.";
    else
      $("community-status").textContent =
        window.Jobs?.describe(job).text || "The download didn’t work. Try again later.";
  };
  try {
    const job = await api("/api/community/import", {});
    if (!window.Jobs || !job?.id) return finish({ state: "done", result: job });
    window.Jobs.track(job);
    window.Jobs.watch(job.id, (latest) => {
      if (["done", "failed", "cancelled"].includes(latest.state)) finish(latest);
      else $("community-status").textContent = window.Jobs.describe(latest).text;
    });
  } catch (error) {
    buttonNode.disabled = false;
    $("community-status").textContent = error.status
      ? error.message
      : "The app didn’t respond. Is it still running?";
  }
}
if ($("community-import")) {
  $("community-import").addEventListener("click", communityImport);
  $("community-box").addEventListener(
    "toggle",
    () => $("community-box").open && communityStatus(),
  );
}
$("gpu_index").addEventListener("change", () => {
  if (appState?.hardware) hardware(appState.hardware);
});
// Files already on disk (mounted on first open so the page doesn't scan early).
if ($("local-files-box")) {
  $("local-files-box").addEventListener("toggle", () => {
    if (!$("local-files-box").open || $("local-files").dataset.mounted) return;
    if (!window.LocalModels?.mount) {
      $("local-files").textContent = "This list isn’t available in this version.";
      return;
    }
    $("local-files").dataset.mounted = "1";
    window.LocalModels.mount($("local-files"), {
      onJob: (job) => job && window.Jobs?.track(job),
    });
  });
}
