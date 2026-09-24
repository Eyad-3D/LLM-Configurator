/* "Get it running": one panel per recommendation that walks from engine install to a
   running local server. Numbers shown here come from the app's measurements only;
   the page never shows file paths, and demo mode explains why actions are unavailable. */
(() => {
  "use strict";
  const dialog = document.getElementById("run-panel");
  if (!dialog) return;
  const call = (path, body) => window.api(path, body);
  const fmt = () => window.Jobs?.format || { bytes: String, duration: String };
  const DEMO_TEXT =
    "Not available in demo mode. The demo models are made up, so they can’t be downloaded, tested or run. Start the app without --demo to use real models.";
  const STEPS = [
    {
      id: "runtime",
      title: "Get the engine",
      help: "llama.cpp is the free engine that runs AI models on your computer — like a media player, but for models. We install it once and reuse it.",
    },
    {
      id: "download",
      title: "Download the model",
      help: "The model is one large file. Downloads can pause and pick up where they stopped. If you already have it, we reuse your copy.",
    },
    {
      id: "test",
      title: "Test it",
      help: "Loads the model, asks a simple question and measures speed and memory on your computer.",
    },
    {
      id: "tune",
      title: "Tune it",
      help: "Tries different engine settings on your computer and keeps the fastest. Your computer stays busy while it runs.",
    },
    {
      id: "quality",
      title: "Check quality",
      help: "Quick quizzes and side-by-side answers help you judge if it is good enough for your work.",
    },
    {
      id: "use",
      title: "Use it",
      help: "Copy a ready-made setup for your favourite app, or start a local server that other apps can talk to.",
    },
  ];
  const FALLBACK_FORMATS = [
    { id: "llama-server", label: "Start script" },
    { id: "ollama", label: "Ollama" },
    { id: "lmstudio", label: "LM Studio" },
    { id: "open-webui", label: "Open WebUI" },
    { id: "continue", label: "Continue" },
    { id: "openai-python", label: "Python" },
    { id: "docker-compose", label: "Docker" },
  ];
  const SETTING_LABELS = {
    threads: "CPU threads",
    batch: "Reading batch size",
    ubatch: "Reading micro-batch size",
    flash_attn: "Flash attention (a faster way to do the maths)",
    cache_type_k: "Compressed notes (keys)",
    cache_type_v: "Compressed notes (values)",
    n_cpu_moe: "Expert layers kept in main memory",
    gpu_layers: "Layers on the graphics card",
    context: "Context size",
    parallel: "Chats at once",
  };
  const STATE_LABELS = {
    checking: "Checking…",
    ready: "Ready",
    blocked: "Waiting",
    working: "Working…",
    paused: "Paused",
    done: "Done",
    failed: "Needs attention",
    demo: "Demo",
  };

  const memory = new Map(); // candidate id -> per-candidate step state
  const shared = {
    runtime: { status: "checking" },
    serve: { status: "checking" },
    formats: null,
  };
  let ctx = null;
  let mem = null;
  let opener = null;
  let changed = false;
  let expanded = null;

  // ---- small DOM helpers (textContent only, so model names can't inject markup) ----
  function el(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key === "value" || key === "checked") node[key] = value;
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    node.append(
      ...kids.flat(Infinity).filter((kid) => kid != null && kid !== false),
    );
    return node;
  }
  const button = (text, onclick, cls = "", extra = {}) =>
    el("button", { type: "button", class: cls, onclick, ...extra }, text);
  const hint = (text) => el("p", { class: "hint", text });
  const bytes = (n) => fmt().bytes(n);
  const number = (n, digits = 1) =>
    n == null || !Number.isFinite(Number(n)) ? null : Number(n).toFixed(digits);
  function announce(text) {
    const live = document.getElementById("run-live");
    if (!live) return;
    live.textContent = "";
    setTimeout(() => (live.textContent = text), 20);
  }
  function detectPlatform() {
    const text =
      navigator.userAgentData?.platform ||
      navigator.platform ||
      navigator.userAgent ||
      "";
    return /win/i.test(text) ? "windows" : "posix";
  }
  function problem(error) {
    if (error?.status === 409 && ctx?.demo)
      return { status: "demo", reason: DEMO_TEXT };
    return {
      status: "failed",
      reason: error?.message || "Something went wrong.",
    };
  }
  function fresh() {
    return {
      download: { status: "checking" },
      test: { status: "idle" },
      tune: { status: "idle", budget: 300, goal: "generation" },
      quality: { mounted: false },
      use: { format: null, platform: detectPlatform(), exports: {} },
    };
  }

  // ---- step status ----
  function prerequisite() {
    if (ctx.demo) return DEMO_TEXT;
    if (shared.runtime.status !== "done") return "Get the engine first (step 1).";
    if (mem.download.status !== "done")
      return "Download the model first (step 2).";
    return null;
  }
  function status(id) {
    if (id === "runtime" || id === "download") {
      const s = id === "runtime" ? shared.runtime : mem.download;
      return {
        state: s.status,
        reason: ["blocked", "failed", "demo"].includes(s.status)
          ? s.reason
          : null,
      };
    }
    if (id === "test" || id === "tune") {
      const s = mem[id];
      if (["working", "done", "failed"].includes(s.status))
        return { state: s.status, reason: s.status === "failed" ? s.reason : null };
      const reason = prerequisite();
      if (reason) return { state: ctx.demo ? "demo" : "blocked", reason };
      return { state: "ready", reason: null };
    }
    if (id === "quality") {
      const reason = prerequisite();
      return reason
        ? { state: ctx.demo ? "demo" : "blocked", reason }
        : { state: "ready", reason: null };
    }
    const serve = shared.serve;
    if (serve.status === "working") return { state: "working", reason: null };
    if (runningHere()) return { state: "done", reason: null };
    return { state: "ready", reason: null };
  }

  // ---- panel frame ----
  function summary(c) {
    const parts = [
      `${Number(c.context || 0).toLocaleString()} tokens per chat`,
      { cpu: "Runs on the processor", gpu: "Runs on the graphics card", split: "Graphics card + processor" }[
        c.mode
      ],
    ].filter(Boolean);
    const notes = [];
    if (c.kv_cache_type && c.kv_cache_type !== "f16")
      notes.push(
        "Uses compressed notes (KV cache): stores the conversation memory in less space — lets you fit longer chats, with a small quality cost.",
      );
    if (c.n_cpu_moe > 0)
      notes.push(
        "Keeps some expert parts in main memory (MoE offload): lets a bigger model fit, but it runs a little slower.",
      );
    return el(
      "div",
      { class: "run-summary" },
      el("p", { class: "hint", text: parts.join(" · ") }),
      notes.map((text) => hint(text)),
    );
  }
  function frame() {
    const c = ctx.candidate;
    const steps = el("ol", { class: "run-steps", id: "run-steps" });
    STEPS.forEach((step, index) => {
      const toggle = el(
        "button",
        {
          type: "button",
          class: "step-toggle",
          id: `run-step-${step.id}-toggle`,
          "aria-controls": `run-step-${step.id}-body`,
          "aria-expanded": "false",
          "data-step-toggle": step.id,
        },
        el("span", { class: "step-num", "aria-hidden": "true", text: String(index + 1) }),
        el("span", { class: "step-name", text: step.title }),
        el("span", { class: "step-state" }),
      );
      steps.append(
        el(
          "li",
          { class: "run-step", "data-step": step.id },
          el("h3", {}, toggle),
          el("p", { class: "step-reason", hidden: true }),
          el("div", {
            class: "step-body",
            id: `run-step-${step.id}-body`,
            role: "region",
            "aria-labelledby": toggle.id,
            hidden: true,
          }),
        ),
      );
    });
    steps.addEventListener("click", (event) => {
      const toggle = event.target.closest("[data-step-toggle]");
      if (toggle) expand(toggle.dataset.stepToggle, true);
    });
    steps.addEventListener("keydown", (event) => {
      const toggle = event.target.closest?.("[data-step-toggle]");
      if (!toggle) return;
      const toggles = [...steps.querySelectorAll("[data-step-toggle]")];
      const index = toggles.indexOf(toggle);
      const target = {
        ArrowDown: toggles[(index + 1) % toggles.length],
        ArrowUp: toggles[(index - 1 + toggles.length) % toggles.length],
        Home: toggles[0],
        End: toggles.at(-1),
      }[event.key];
      if (target) {
        event.preventDefault();
        target.focus();
      }
    });
    const content = document.getElementById("run-content");
    content.replaceChildren(
      el(
        "div",
        { class: "run-head" },
        el(
          "div",
          {},
          el("p", { class: "eyebrow", text: "GET IT RUNNING" }),
          el(
            "h2",
            { id: "run-title", tabindex: "-1" },
            `${c.name} `,
            el("span", { class: "quant", text: c.quant || "" }),
          ),
          window.verdictBadge ? verdictNode(c) : null,
          summary(c),
        ),
      ),
      steps,
    );
  }
  function verdictNode(c) {
    const holder = el("div", { class: "run-verdict" });
    holder.innerHTML = window.verdictBadge(c); // escaped by app.js
    return holder;
  }
  function stateLabel(id, state) {
    if (id === "use" && state === "done") return "Server running";
    if (id === "runtime" && state === "ready") return "Not installed";
    if (id === "download" && state === "ready")
      return mem.download.plan?.local_copy ? "Found on your disk" : "Not downloaded";
    return STATE_LABELS[state];
  }
  function expand(id, toggleOpen = false) {
    const same = expanded === id;
    expanded = toggleOpen && same ? null : id;
    for (const step of STEPS) {
      const open = step.id === expanded;
      const toggle = document.getElementById(`run-step-${step.id}-toggle`);
      toggle?.setAttribute("aria-expanded", String(open));
      const body = document.getElementById(`run-step-${step.id}-body`);
      if (body) body.hidden = !open;
    }
    if (expanded) render(expanded);
    if (expanded === "use") ensureFormats();
  }
  function render(id) {
    if (!ctx || !dialog.contains(document.getElementById(`run-step-${id}-body`)))
      return;
    const item = dialog.querySelector(`[data-step="${id}"]`);
    const { state, reason } = status(id);
    item.className = `run-step state-${state}`;
    item.querySelector(".step-state").textContent = stateLabel(id, state);
    const why = item.querySelector(".step-reason");
    why.hidden = !reason || id === expanded;
    why.textContent = reason || "";
    const body = item.querySelector(".step-body");
    if (body.hidden) return;
    const hadFocus = body.contains(document.activeElement);
    const help = STEPS.find((s) => s.id === id).help;
    if (id === "quality") {
      // The quality panel owns its own DOM; only mount it once per open.
      if (!mem.quality.mounted || !body.querySelector(".quality-mount"))
        mountQuality(body, reason);
      return;
    }
    body.replaceChildren(hint(help), ...views[id](reason));
    if (hadFocus)
      (body.querySelector("button:not([disabled]), [tabindex='0']") ||
        document.getElementById(`run-step-${id}-toggle`))?.focus();
  }
  function renderAll() {
    STEPS.forEach((step) => render(step.id));
  }
  const reasonView = (reason) =>
    reason ? el("p", { class: "step-note", text: reason }) : null;
  function progressView(job, onCancel, extra = []) {
    const info = window.Jobs?.describe(job) || { fraction: null, text: "Working…" };
    const bar = el("progress", { class: "job-bar", max: "1", "aria-label": "Progress" });
    if (info.fraction != null) bar.value = info.fraction;
    return el(
      "div",
      { class: "job-progress" },
      bar,
      el("p", { class: "job-text", role: "status", text: info.text }),
      el(
        "div",
        { class: "step-actions" },
        extra,
        onCancel ? button("Cancel", onCancel, "secondary") : null,
      ),
    );
  }
  function updateProgress(id, job) {
    const box = dialog.querySelector(`[data-step="${id}"] .job-progress`);
    if (!box) return render(id);
    const info = window.Jobs.describe(job);
    const bar = box.querySelector("progress");
    if (info.fraction != null) bar.value = info.fraction;
    else bar.removeAttribute("value");
    box.querySelector(".job-text").textContent = info.text;
  }
  const current = (candidateId) => ctx && ctx.candidate.id === candidateId;

  /** Start following a job for a step; `finish(job)` updates state when it ends. */
  function follow(id, state, job, finish, candidateId = ctx.candidate.id) {
    state.status = "working";
    state.job = job;
    window.Jobs.track(job);
    const refresh = () => {
      if (!current(candidateId) && id !== "runtime") return;
      if (id === "serve") return render("use");
      render(id);
    };
    refresh();
    state.stop = window.Jobs.watch(job.id, (latest) => {
      state.job = latest;
      if (["done", "failed", "cancelled"].includes(latest.state)) {
        state.stop = null;
        finish(latest);
        if (id === "runtime") renderAll();
        else if (current(candidateId)) renderAll();
        return;
      }
      if (current(candidateId) || id === "runtime")
        updateProgress(id === "serve" ? "use" : id, latest);
    });
  }
  async function cancelJob(state) {
    if (!state.job) return;
    try {
      await window.Jobs.cancel(state.job.id);
    } catch (error) {
      announce(error.message);
    }
  }

  // ---- 1. runtime ----
  async function checkRuntime() {
    if (shared.runtime.status === "working") return;
    try {
      const info = await call("/api/runtime");
      shared.runtime = {
        status: info.installed ? "done" : "ready",
        info,
      };
    } catch (error) {
      shared.runtime = problem(error);
    }
    renderAll();
  }
  async function installRuntime() {
    const s = shared.runtime;
    try {
      const job = await call("/api/runtime/install", {});
      follow("runtime", s, job, (done) => {
        if (done.state === "done" && done.result?.installed) {
          shared.runtime = { status: "done", info: done.result };
          announce("The engine is installed.");
        } else if (done.state === "cancelled") {
          shared.runtime = { status: "ready", info: s.info };
        } else {
          shared.runtime = {
            status: "failed",
            info: s.info,
            reason:
              done.error ||
              "The engine was installed but did not start. Try again, or install llama.cpp yourself.",
          };
        }
      });
    } catch (error) {
      Object.assign(s, problem(error));
      render("runtime");
    }
  }
  function runtimeView() {
    const s = shared.runtime;
    const info = s.info || {};
    if (s.status === "checking") return [hint("Checking for the engine…")];
    if (s.status === "working")
      return [progressView(s.job, () => cancelJob(s))];
    if (s.status === "done") {
      const where = {
        managed: "installed by this app",
        configured: "the copy you chose",
        path: "found on your computer",
      }[info.source];
      const backend = {
        cuda: "NVIDIA graphics card",
        metal: "Apple graphics",
        rocm: "AMD graphics card",
        vulkan: "graphics card (Vulkan)",
        cpu: "processor only",
      }[info.backend];
      return [
        el(
          "p",
          { class: "step-done" },
          `Engine ready${info.version ? ` · version ${info.version}` : ""}${where ? ` · ${where}` : ""}.`,
        ),
        backend ? hint(`Uses your ${backend}.`) : null,
        (info.warnings || []).map((w) => el("p", { class: "step-note", text: w })),
      ];
    }
    return [
      reasonView(s.reason),
      s.status === "demo"
        ? null
        : el(
            "div",
            { class: "step-actions" },
            button(
              s.status === "failed" ? "Try again" : "Install the engine",
              installRuntime,
            ),
            s.status === "failed" ? button("Check again", checkRuntime, "secondary") : null,
          ),
      s.status === "demo"
        ? null
        : hint(
            "Downloads the official llama.cpp release for your system and checks it is genuine before using it.",
          ),
    ];
  }

  // ---- 2. download ----
  async function planDownload(candidate = ctx.candidate) {
    const state = memory.get(candidate.id).download;
    if (state.status === "working") return;
    try {
      const plan = await call("/api/downloads/plan", {
        variant_id: candidate.variant_id,
      });
      state.plan = plan;
      if (plan.remaining_bytes === 0) state.status = "done";
      else if (plan.local_copy) state.status = "ready";
      else if (plan.enough_space === false) {
        state.status = "blocked";
        state.reason = `Not enough free disk space. It needs ${bytes(plan.remaining_bytes)} and you have ${bytes(plan.disk_free)} free (we keep 1 GiB spare). Free some space, then check again.`;
      } else state.status = plan.remaining_bytes < plan.total_bytes ? "paused" : "ready";
      if (state.status !== "blocked") state.reason = null;
    } catch (error) {
      Object.assign(state, problem(error));
    }
    if (current(candidate.id)) renderAll();
  }
  async function startDownload() {
    const state = mem.download;
    const candidate = ctx.candidate;
    state.intent = null;
    try {
      const job = await call("/api/downloads", { variant_id: candidate.variant_id });
      follow("download", state, job, (done) => {
        if (done.state === "done") {
          state.status = "done";
          state.reused = !!done.result?.reused;
          state.bytes = done.result?.bytes ?? state.plan?.total_bytes;
          changed = true;
          if (current(candidate.id))
            announce(state.reused ? "Using the copy on your disk." : "Download finished.");
        } else if (done.state === "cancelled") {
          if (state.intent === "cancel") void removeDownload(candidate, true);
          else {
            state.status = "paused";
            void planDownload(candidate);
          }
        } else {
          state.status = "failed";
          state.reason = done.error || "The download stopped.";
        }
      });
    } catch (error) {
      Object.assign(state, problem(error));
      render("download");
    }
  }
  async function removeDownload(candidate = ctx.candidate, afterCancel = false) {
    const state = memory.get(candidate.id).download;
    try {
      const result = await call("/api/downloads/remove", {
        variant_id: candidate.variant_id,
      });
      state.status = "checking";
      if (current(candidate.id))
        announce(
          afterCancel
            ? "Download cancelled. Partial files removed."
            : `Removed. Freed ${bytes(result.freed_bytes)}.`,
        );
      state.message = afterCancel
        ? "Download cancelled. Partial files removed."
        : `Removed from disk. Freed ${bytes(result.freed_bytes)}.`;
      await planDownload(candidate);
    } catch (error) {
      Object.assign(state, problem(error));
      if (current(candidate.id)) renderAll();
    }
  }
  function downloadView() {
    const s = mem.download;
    const plan = s.plan || {};
    const size = plan.total_bytes != null ? bytes(plan.total_bytes) : null;
    const facts = hint(
      [
        size && `Size: ${size}`,
        plan.disk_free != null && `Free disk space: ${bytes(plan.disk_free)}`,
        plan.files?.length > 1 && `${plan.files.length} parts`,
      ]
        .filter(Boolean)
        .join(" · "),
    );
    if (s.status === "checking") return [hint("Checking your disk…")];
    if (s.status === "working")
      return [
        facts,
        progressView(s.job, null, [
          button("Pause", () => {
            s.intent = "pause";
            cancelJob(s);
          }, "secondary"),
          button("Cancel download", () => {
            s.intent = "cancel";
            cancelJob(s);
          }, "secondary"),
        ]),
        hint("Pause keeps what is downloaded so far. Cancel deletes it."),
      ];
    if (s.status === "done")
      return [
        el(
          "p",
          { class: "step-done" },
          s.reused || (plan.local_copy && !s.bytes)
            ? "Using the copy already on your disk."
            : `Downloaded${size ? ` (${size})` : ""}. Checked and ready.`,
        ),
        el(
          "div",
          { class: "step-actions" },
          button("Remove from disk", () => removeDownload(), "text-button"),
        ),
      ];
    if (s.status === "demo" || s.status === "failed" || s.status === "blocked")
      return [
        reasonView(s.reason),
        s.status === "demo"
          ? null
          : el(
              "div",
              { class: "step-actions" },
              s.status === "failed" ? button("Try again", startDownload) : null,
              button("Check again", () => planDownload(), "secondary"),
            ),
      ];
    const partial =
      plan.total_bytes && plan.remaining_bytes < plan.total_bytes
        ? plan.total_bytes - plan.remaining_bytes
        : 0;
    return [
      s.message ? el("p", { class: "step-note", text: s.message }) : null,
      facts,
      plan.local_copy
        ? el(
            "p",
            { class: "step-done" },
            "Found on your disk — nothing to download.",
          )
        : null,
      partial
        ? hint(`Paused with ${bytes(partial)} of ${size} saved.`)
        : null,
      el(
        "div",
        { class: "step-actions" },
        button(
          plan.local_copy
            ? "Use the copy on my disk"
            : partial
              ? "Resume download"
              : `Download${size ? ` (${size})` : ""}`,
          startDownload,
        ),
        partial
          ? button("Delete partial download", () => removeDownload(), "secondary")
          : null,
      ),
    ];
  }

  // ---- 3. test ----
  async function startTest(kind) {
    const state = mem.test;
    const candidate = ctx.candidate;
    try {
      const job = await call("/api/test", { candidate_id: candidate.id, kind });
      follow("test", state, job, (done) => {
        if (done.state === "done") {
          state.status = "done";
          state.result = done.result;
          changed = true;
        } else if (done.state === "cancelled") state.status = state.result ? "done" : "idle";
        else {
          state.status = "failed";
          state.reason = done.error || "The test did not finish.";
        }
      });
    } catch (error) {
      Object.assign(state, problem(error));
      render("test");
    }
  }
  function metricBox(label, value, sub) {
    return el(
      "div",
      { class: "run-metric" },
      el("span", { class: "run-metric-label", text: label }),
      el("strong", { text: value ?? "Not measured" }),
      sub ? el("small", { text: sub }) : null,
    );
  }
  function memoryText(peak, estimate) {
    if (peak == null) return null;
    return `${bytes(peak)} used${estimate != null ? ` · estimate ${bytes(estimate)}` : ""}`;
  }
  function testResult(result) {
    const verdict = {
      works: ["good", "It works"],
      works_slowly: ["slow", "It works, but slowly"],
      failed: ["bad", "It didn’t work"],
    }[result.verdict] || ["unknown", "Test finished"];
    const smoke = result.smoke;
    const speed = result.speed;
    const summaryData = speed?.summary || {};
    const mem = speed?.memory || {};
    const nodes = [
      el(
        "div",
        { class: `test-verdict verdict-${verdict[0]}` },
        el("strong", { text: verdict[1] }),
        result.verdict_text ? el("span", { text: result.verdict_text }) : null,
      ),
    ];
    if (smoke)
      nodes.push(
        el(
          "ul",
          { class: "checks", "aria-label": "Checks" },
          (smoke.checks || []).map((check) =>
            el(
              "li",
              { class: check.ok ? "ok" : "bad" },
              el("span", { "aria-hidden": "true", text: check.ok ? "✓" : "✗" }),
              el("span", { class: "visually-hidden", text: check.ok ? "Passed: " : "Failed: " }),
              `${check.name}${check.detail ? ` — ${check.detail}` : ""}`,
            ),
          ),
        ),
        smoke.ok === false && smoke.message
          ? el("p", { class: "step-note", text: smoke.message })
          : null,
        smoke.load_seconds != null
          ? hint(`Loaded in ${number(smoke.load_seconds, 0)} s.`)
          : null,
      );
    if (speed) {
      const ramLine = memoryText(mem.peak_ram_bytes, mem.estimated_ram_bytes);
      const vramLine = memoryText(mem.peak_vram_bytes, mem.estimated_vram_bytes);
      nodes.push(
        el(
          "div",
          { class: "run-metrics" },
          metricBox(
            "Reading speed",
            number(summaryData.pp_tps) && `${number(summaryData.pp_tps)} tokens/s`,
            "How fast it reads what you send",
          ),
          metricBox(
            "First-word delay",
            number(summaryData.ttft_s, 2) && `${number(summaryData.ttft_s, 2)} s`,
            "Wait before the answer starts",
          ),
          metricBox(
            "Writing speed",
            number(summaryData.tps) && `${number(summaryData.tps)} tokens/s`,
            "How fast the answer appears",
          ),
          metricBox(
            "Memory",
            ramLine ? `RAM: ${ramLine}` : null,
            [
              vramLine && `Graphics memory: ${vramLine}`,
              mem.within_estimate === true
                ? "Within our estimate"
                : mem.within_estimate === false
                  ? "More than we estimated"
                  : null,
            ]
              .filter(Boolean)
              .join(" · ") || null,
          ),
        ),
        mem.note ? hint(mem.note) : null,
        summaryData.depth
          ? hint(
              `Speeds measured with ${Number(summaryData.depth).toLocaleString()} tokens already in the chat, close to your chosen size.`,
            )
          : null,
        hint("A token is a piece of a word — roughly ¾ of a word on average."),
      );
    }
    return nodes;
  }
  function testView(reason) {
    const s = mem.test;
    if (s.status === "working") return [progressView(s.job, () => cancelJob(s))];
    return [
      s.status === "failed" ? reasonView(s.reason) : null,
      s.result
        ? testResult(s.result)
        : reason
          ? null
          : hint("Quick check: does it load and answer? Full test: also measures speed and memory."),
      reason
        ? reasonView(reason)
        : el(
            "div",
            { class: "step-actions" },
            button(
              s.result ? "Run the full test again" : "Run the full test",
              () => startTest("full"),
              s.result ? "secondary" : "",
            ),
            button("Quick check only", () => startTest("smoke"), "secondary"),
          ),
    ];
  }

  // ---- 4. tune ----
  async function startTune() {
    const state = mem.tune;
    const candidate = ctx.candidate;
    try {
      const job = await call("/api/tune", {
        candidate_id: candidate.id,
        budget_seconds: state.budget,
        goal: state.goal,
      });
      follow("tune", state, job, (done) => {
        if (done.state === "done") {
          state.status = "done";
          state.result = done.result;
          changed = true;
        } else if (done.state === "cancelled") state.status = state.result ? "done" : "idle";
        else {
          state.status = "failed";
          state.reason = done.error || "Tuning did not finish.";
        }
      });
    } catch (error) {
      Object.assign(state, problem(error));
      render("tune");
    }
  }
  function speedChange(label, before, after) {
    if (before == null && after == null) return null;
    const b = number(before);
    const a = number(after);
    const pct =
      before > 0 && after != null
        ? Math.round((after / before - 1) * 100)
        : null;
    return el(
      "div",
      { class: "run-metric" },
      el("span", { class: "run-metric-label", text: label }),
      el("strong", { text: `${b ?? "?"} → ${a ?? "?"} tokens/s` }),
      pct != null
        ? el("small", { text: pct > 0 ? `${pct}% faster` : pct < 0 ? `${-pct}% slower` : "No change" })
        : null,
    );
  }
  function settingValue(key, value) {
    if (value == null) return "default";
    if (key.startsWith("cache_type"))
      return { f16: "off (full size)", q8_0: "on (half size)", q4_0: "on (quarter size)" }[value] || value;
    if (key === "flash_attn") return { on: "on", off: "off", auto: "automatic" }[value] || String(value);
    return String(value);
  }
  function tuneResult(result) {
    const base = ctx.candidate.launch || {};
    const best = result.best || {};
    const changes = Object.keys(SETTING_LABELS).filter(
      (key) => key in best && base[key] !== undefined && best[key] !== base[key],
    );
    const ratio = Number(result.improvement);
    const headline =
      Number.isFinite(ratio) && ratio > 1.02
        ? `Found settings about ${Math.round((ratio - 1) * 100)}% faster.`
        : "Your starting settings were already about as fast as it gets.";
    const tried = (result.trials || []).filter((t) => t.status === "ok").length;
    const stopped = {
      budget: "Stopped when the time ran out.",
      converged: "Stopped early: nothing faster left to try.",
      cancelled: "Stopped because you cancelled.",
    }[result.stopped];
    return [
      el("p", { class: "step-done", text: headline }),
      el(
        "div",
        { class: "run-metrics" },
        speedChange("Writing speed", result.baseline?.tps, result.best_result?.tps),
        speedChange("Reading speed", result.baseline?.pp_tps, result.best_result?.pp_tps),
      ),
      changes.length
        ? el(
            "ul",
            { class: "changes", "aria-label": "Settings that changed" },
            changes.map((key) =>
              el("li", {
                text: `${SETTING_LABELS[key]}: ${settingValue(key, base[key])} → ${settingValue(key, best[key])}`,
              }),
            ),
          )
        : null,
      hint([`Tried ${tried} setting${tried === 1 ? "" : "s"}.`, stopped].filter(Boolean).join(" ")),
      (result.notes || []).map((note) => hint(note)),
      hint("The best settings are saved on this computer for this model."),
    ];
  }
  function tuneView(reason) {
    const s = mem.tune;
    if (s.status === "working") return [progressView(s.job, () => cancelJob(s))];
    const name = `run-budget-${ctx.candidate.id}`.replace(/[^\w-]/g, "_");
    const budgets = el(
      "fieldset",
      { class: "budget" },
      el("legend", { text: "How long can it take?" }),
      [
        [60, "1 minute"],
        [300, "5 minutes"],
        [900, "15 minutes"],
      ].map(([seconds, label]) =>
        el(
          "label",
          { class: "check" },
          el("input", {
            type: "radio",
            name,
            value: String(seconds),
            checked: s.budget === seconds,
            onchange: () => (s.budget = seconds),
          }),
          label,
        ),
      ),
    );
    const goalId = `run-goal-${Math.random().toString(36).slice(2, 8)}`;
    const goal = el(
      "select",
      { id: goalId, onchange: (e) => (s.goal = e.target.value) },
      [
        ["generation", "Faster writing (chat, coding)"],
        ["balanced", "A balance of reading and writing"],
        ["prompt", "Faster reading (long documents)"],
      ].map(([value, label]) =>
        el("option", { value, selected: s.goal === value }, label),
      ),
    );
    const controls = el(
      "div",
      { class: "tune-controls" },
      budgets,
      el("div", {}, el("label", { for: goalId, text: "What should get faster?" }), goal),
    );
    return [
      s.status === "failed" ? reasonView(s.reason) : null,
      s.result ? tuneResult(s.result) : null,
      reason
        ? reasonView(reason)
        : [
            controls,
            el(
              "div",
              { class: "step-actions" },
              button(s.result ? "Tune again" : "Start tuning", startTune, s.result ? "secondary" : ""),
            ),
          ],
    ];
  }

  // ---- 5. quality ----
  function qualityApi(path, options) {
    // Accept both api(path, body) and fetch-style api(path, {method, body}).
    if (
      options &&
      typeof options === "object" &&
      ("method" in options || "body" in options)
    ) {
      const method = String(options.method || (options.body != null ? "POST" : "GET")).toUpperCase();
      if (method === "GET") return call(path);
      const body =
        typeof options.body === "string" ? JSON.parse(options.body) : options.body ?? {};
      return call(path, body);
    }
    return call(path, options);
  }
  function mountQuality(body, reason) {
    const help = hint(STEPS.find((s) => s.id === "quality").help);
    const mount = el("div", { class: "quality-mount" });
    body.replaceChildren(help, reason ? reasonView(reason) : null, mount);
    mem.quality.mounted = true;
    if (!window.QualityPanel?.mount) {
      mount.append(
        el("p", {
          class: "step-note",
          text: "Quality checks aren’t available in this version yet. You can still test speed and use the model.",
        }),
      );
      return;
    }
    try {
      window.QualityPanel.mount(mount, {
        api: qualityApi,
        candidate: ctx.candidate,
        candidates: ctx.candidates,
        workload: ctx.workload,
        onJob: (job) => job && window.Jobs?.track(job),
      });
    } catch (error) {
      mount.append(el("p", { class: "step-note", text: `Quality checks could not open: ${error.message}` }));
    }
  }

  // ---- 6. use: export + serve ----
  async function ensureFormats() {
    if (shared.formats) return;
    shared.formats = FALLBACK_FORMATS;
    try {
      const listed = await call("/api/export");
      if (Array.isArray(listed?.formats) && listed.formats.length) {
        shared.formats = listed.formats;
        render("use");
      }
    } catch {
      // Older servers only support POST /api/export; the built-in list matches export.formats().
    }
  }
  async function loadExport(format) {
    const state = mem.use;
    state.format = format;
    const key = `${format}|${state.platform}`;
    const candidate = ctx.candidate;
    if (!state.exports[key]) {
      state.exports[key] = { loading: true };
      render("use");
      try {
        state.exports[key] = await call("/api/export", {
          candidate_id: candidate.id,
          format,
          platform: state.platform,
        });
      } catch (error) {
        state.exports[key] = { error: problem(error).reason };
      }
    }
    if (current(candidate.id)) {
      render("use");
    }
  }
  async function copy(text, statusNode) {
    let ok = false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch {}
    if (!ok) {
      const area = el("textarea", { class: "copy-buffer", readonly: true, "aria-hidden": "true" });
      area.value = text;
      document.getElementById("run-content").append(area);
      area.select();
      try {
        ok = document.execCommand?.("copy") === true;
      } catch {}
      area.remove();
    }
    const message = ok ? "Copied." : "Couldn’t copy automatically. Select the text and press Ctrl+C (⌘C on Mac).";
    if (statusNode) statusNode.textContent = message;
    announce(message);
    return ok;
  }
  function exportView() {
    const state = mem.use;
    const formats = shared.formats || FALLBACK_FORMATS;
    const selected = formats.some((f) => f.id === state.format) ? state.format : null;
    const tabs = el("div", {
      class: "export-tabs",
      role: "tablist",
      "aria-label": "Setup formats",
    });
    formats.forEach((format) => {
      const active = format.id === selected;
      tabs.append(
        el(
          "button",
          {
            type: "button",
            role: "tab",
            class: "export-tab",
            id: `run-tab-${format.id}`,
            "data-format": format.id,
            "aria-selected": String(active),
            "aria-controls": "run-export-panel",
            tabindex: active || (!selected && format === formats[0]) ? "0" : "-1",
            title: format.description || null,
            onclick: () => loadExport(format.id),
          },
          format.label,
        ),
      );
    });
    tabs.addEventListener("keydown", (event) => {
      const list = [...tabs.querySelectorAll("[role=tab]")];
      const index = list.indexOf(event.target);
      if (index < 0) return;
      const target = {
        ArrowRight: list[(index + 1) % list.length],
        ArrowLeft: list[(index - 1 + list.length) % list.length],
        Home: list[0],
        End: list.at(-1),
      }[event.key];
      if (!target) return;
      event.preventDefault();
      list.forEach((t) => t.setAttribute("tabindex", t === target ? "0" : "-1"));
      target.focus();
      target.click();
    });
    const platform = el(
      "div",
      { class: "platform-toggle", role: "group", "aria-label": "Your system" },
      [
        ["posix", "Mac or Linux"],
        ["windows", "Windows"],
      ].map(([value, label]) =>
        button(
          label,
          () => {
            state.platform = value;
            if (state.format) loadExport(state.format);
            else render("use");
          },
          "secondary small",
          { "aria-pressed": String(state.platform === value) },
        ),
      ),
    );
    const panel = el("div", {
      class: "export-panel",
      id: "run-export-panel",
      role: "tabpanel",
      "aria-labelledby": selected ? `run-tab-${selected}` : null,
    });
    const result = selected && state.exports[`${selected}|${state.platform}`];
    if (!selected) panel.append(hint("Pick where you want to use the model."));
    else if (!result || result.loading) panel.append(hint("Preparing…"));
    else if (result.error) panel.append(el("p", { class: "step-note", text: result.error }));
    else {
      const copyStatus = el("span", { class: "copy-status", role: "status" });
      panel.append(
        result.instructions?.length
          ? el("ol", { class: "instructions" }, result.instructions.map((text) => el("li", { text })))
          : null,
        el(
          "div",
          { class: "code-head" },
          el("code", { text: result.filename || "" }),
          el("span", {}, copyStatus, button("Copy", () => copy(result.content || "", copyStatus), "secondary small", { "data-copy": "export" })),
        ),
        el("pre", { class: "export-content", tabindex: "0" }, el("code", { text: result.content || "" })),
        (result.notes || []).map((note) => hint(note)),
      );
    }
    return [
      el("h4", { text: "Copy a setup" }),
      el("div", { class: "export-bar" }, tabs, platform),
      panel,
    ];
  }
  function runningHere() {
    const s = shared.serve.info;
    if (!s?.running || !ctx) return false;
    const c = ctx.candidate;
    return [s.candidate_id, s.variant_id, s.model, s.config?.alias].some(
      (v) => v != null && [c.id, c.variant_id, c.name, c.filename].includes(v),
    );
  }
  async function checkServe() {
    if (shared.serve.status === "working") return;
    try {
      const info = await call("/api/serve");
      shared.serve = { status: "ready", info };
    } catch {
      shared.serve = { status: "ready", info: null }; // Unknown: offer Start; errors show then.
    }
    render("use");
  }
  async function startServe() {
    const state = shared.serve;
    state.reason = null;
    try {
      const job = await call("/api/serve/start", { candidate_id: ctx.candidate.id });
      follow("serve", state, job, (done) => {
        if (done.state === "done") {
          shared.serve = { status: "ready", info: done.result };
          announce("Server started.");
        } else {
          shared.serve = {
            status: "ready",
            info: state.info,
            reason: done.state === "cancelled" ? null : done.error || "The server did not start.",
          };
        }
      });
    } catch (error) {
      const p = problem(error);
      state.status = "ready";
      state.reason = p.reason;
      render("use");
    }
  }
  async function stopServe() {
    try {
      shared.serve = { status: "ready", info: await call("/api/serve/stop", {}) };
      announce("Server stopped.");
    } catch (error) {
      shared.serve.reason = problem(error).reason;
    }
    render("use");
  }
  function serveView(reason) {
    const s = shared.serve;
    const info = s.info || {};
    const nodes = [el("h4", { text: "Run it as a local server" })];
    if (s.status === "working") {
      nodes.push(hint("Loading the model. Large models can take a minute."), progressView(s.job, () => cancelJob(s)));
      return nodes;
    }
    if (s.status === "checking") return [...nodes, hint("Checking…")];
    if (info.running) {
      const address = info.openai_base_url || (info.base_url ? `${info.base_url}/v1` : "");
      const copyStatus = el("span", { class: "copy-status", role: "status" });
      nodes.push(
        el(
          "p",
          { class: runningHere() ? "step-done" : "step-note" },
          runningHere()
            ? "Running. Apps on this computer can now use the model."
            : `Another model is running${info.model ? ` (${info.model})` : ""}. Starting this one replaces it.`,
        ),
        el(
          "div",
          { class: "code-head" },
          el("span", {}, "OpenAI address: ", el("code", { class: "base-url", text: address })),
          el("span", {}, copyStatus, button("Copy", () => copy(address, copyStatus), "secondary small", { "data-copy": "url" })),
        ),
        hint("Paste this address into any app that works with OpenAI (use any text as the API key). It only listens on this computer."),
      );
    } else nodes.push(hint("Starts the model in the background with the settings above. Other apps on this computer can then chat with it."));
    if (s.reason) nodes.push(reasonView(s.reason));
    if (reason) nodes.push(reasonView(reason));
    else
      nodes.push(
        el(
          "div",
          { class: "step-actions" },
          runningHere() ? null : button(info.running ? "Start this model instead" : "Start server", startServe),
          info.running ? button("Stop server", stopServe, "secondary") : null,
        ),
      );
    return nodes;
  }
  function useView() {
    return [...exportView(), ...serveView(prerequisite())];
  }

  const views = {
    runtime: runtimeView,
    download: downloadView,
    test: testView,
    tune: tuneView,
    use: useView,
  };

  // ---- open / close ----
  function reattach() {
    // Pick up server jobs started earlier (for example before a page reload).
    window.Jobs?.list().then(
      (jobs) => {
        if (!ctx) return;
        const running = jobs.filter((j) => ["queued", "running"].includes(j.state));
        const mine = (j) =>
          j.subject?.candidate_id === ctx.candidate.id ||
          j.subject?.variant_id === ctx.candidate.variant_id;
        for (const job of running) {
          const kind = String(job.kind || "");
          if (/runtime/.test(kind) && shared.runtime.status !== "working")
            follow("runtime", shared.runtime, job, () => checkRuntime());
          else if (/download/.test(kind) && mine(job) && mem.download.status !== "working")
            follow("download", mem.download, job, () => planDownload());
          else if (/serve/.test(kind) && shared.serve.status !== "working")
            follow("serve", shared.serve, job, () => checkServe());
        }
      },
      () => {},
    );
  }
  function firstOpenStep() {
    return (
      STEPS.find((step) => {
        const { state } = status(step.id);
        return !["done", "blocked"].includes(state) && step.id !== "quality";
      })?.id || "use"
    );
  }
  async function open(candidate, options = {}) {
    if (!candidate) return;
    opener = document.activeElement;
    changed = false;
    if (!memory.has(candidate.id)) memory.set(candidate.id, fresh());
    ctx = {
      candidate,
      candidates: options.candidates || [candidate],
      workload: options.workload || null,
      demo: !!options.demo,
    };
    mem = memory.get(candidate.id);
    mem.quality.mounted = false;
    expanded = null;
    frame();
    if (typeof dialog.showModal === "function") {
      if (!dialog.open) dialog.showModal();
    } else dialog.setAttribute("open", "");
    renderAll();
    document.getElementById("run-title")?.focus();
    const checks = [checkRuntime(), checkServe()];
    if (mem.download.status !== "working") checks.push(planDownload(candidate));
    reattach();
    await Promise.all(checks);
    if (ctx?.candidate.id === candidate.id && !expanded) expand(firstOpenStep());
  }
  function cleanup() {
    if (!ctx) return;
    const refresh = changed;
    ctx = null;
    document.getElementById("run-content")?.replaceChildren();
    if (refresh && typeof window.refreshVisibleSpeeds === "function")
      window.refreshVisibleSpeeds().catch?.(() => {});
    if (opener?.isConnected) opener.focus();
  }
  function close() {
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else {
      dialog.removeAttribute("open");
      cleanup();
    }
  }
  dialog.addEventListener("close", cleanup);
  dialog.addEventListener("keydown", (event) => {
    // Native dialogs handle Escape themselves; this covers the fallback path.
    if (event.key === "Escape" && typeof dialog.showModal !== "function") {
      event.preventDefault();
      close();
    }
  });
  document.getElementById("run-close")?.addEventListener("click", close);

  window.RunPanel = { open, close };
})();
