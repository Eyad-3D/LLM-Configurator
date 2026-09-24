const { test } = require("node:test");
const assert = require("node:assert/strict");
const { JSDOM } = require("jsdom");
const fs = require("node:fs");
const vm = require("node:vm");
const root = "src/llm_configurator/static/";
const GiB = 1073741824;
const tick = () => new Promise((resolve) => setTimeout(resolve, 20));
async function until(check, label = "condition", timeout = 3000) {
  const start = Date.now();
  for (;;) {
    try {
      if (check()) return;
    } catch {}
    if (Date.now() - start > timeout) throw new Error("Timed out: " + label);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function candidate(overrides = {}) {
  return {
    id: "v1|now|8192|0",
    variant_id: "v1",
    name: "Model A",
    quant: "Q4_K_M",
    filename: "model-a.gguf",
    context: 8192,
    users: 1,
    mode: "cpu",
    gpu_layers: 0,
    total_layers: 32,
    scenario: "now",
    quality_score: null,
    quality_metric: "general",
    tps: null,
    speed_estimate: { available: false, reason: "Calibration needed" },
    ram_bytes: 5 * GiB,
    vram_bytes: 0,
    explanation: "Estimated fit.",
    quality_comparison: { rank: null, reason: "missing" },
    verdict: "runs_well",
    verdict_text: "Should write about 20 words a second.",
    evidence: "estimated",
    kv_cache_type: "f16",
    n_cpu_moe: 0,
    launch: {
      threads: 4,
      batch: 512,
      ubatch: 512,
      flash_attn: "auto",
      cache_type_k: "f16",
      cache_type_v: "f16",
    },
    ...overrides,
  };
}

/**
 * A page with the results screen showing `candidates`, and a fake API.
 * `routes["METHOD /path"]` may be an object or (body, s) => value; throw
 * {status, error} from a function to return an error response.
 */
async function setup({
  demo = false,
  candidates = [candidate()],
  routes = {},
  platform = "Linux x86_64",
  clipboard = true,
} = {}) {
  const dom = new JSDOM(fs.readFileSync(root + "index.html", "utf8"), {
    url: "http://127.0.0.1:8765",
    runScripts: "outside-only",
    pretendToBeVisual: true,
  });
  const w = dom.window;
  w.scrollTo = () => {};
  Object.defineProperty(w.navigator, "platform", { value: platform });
  const copied = [];
  if (clipboard)
    Object.defineProperty(w.navigator, "clipboard", {
      value: { writeText: async (text) => copied.push(text) },
    });
  const calls = [];
  const jobs = new Map();
  let nextJob = 1;
  const s = { w, calls, jobs, copied };
  s.job = (kind, title, extra = {}) => {
    const job = {
      id: `job-${nextJob++}`,
      kind,
      title,
      state: "running",
      progress: null,
      result: null,
      error: null,
      subject: {},
      ...extra,
    };
    jobs.set(job.id, job);
    return job;
  };
  s.update = (id, changes) => Object.assign(jobs.get(id), changes);
  const defaults = {
    "GET /api/state": () => ({
      hardware: {
        ram_available: 16e9,
        ram_total: 32e9,
        cpu_percent: 1,
        cores: 8,
        threads: 16,
        cpu: "Test CPU",
        gpus: [],
        disk_free: 100e9,
        processes: [],
      },
      demo,
      status: { variants: 1, timestamp: new Date().toISOString(), warnings: [] },
      definitions: [],
      scores: [],
    }),
    "POST /api/calibrate": { running: false, cached: true, result: { cpu: {}, gpus: {} } },
    "GET /api/jobs": () => ({ jobs: [...jobs.values()] }),
    "GET /api/runtime": {
      installed: false,
      source: null,
      directory: null,
      version: null,
      backend: null,
      binaries: {},
      warnings: [],
    },
    "POST /api/downloads/plan": {
      files: [{ filename: "model-a.gguf", size_bytes: 4 * GiB, present: false, partial_bytes: 0 }],
      total_bytes: 4 * GiB,
      remaining_bytes: 4 * GiB,
      disk_free: 100 * GiB,
      enough_space: true,
      local_copy: false,
    },
    "GET /api/serve": { running: false, base_url: null, openai_base_url: null },
    "GET /api/export": () => {
      throw { status: 404, error: "Not found" };
    },
  };
  w.fetch = async (path, options = {}) => {
    const method = options.method || "GET";
    const body = options.body ? JSON.parse(options.body) : null;
    assert.equal(options.headers["X-Session-Token"], "__SESSION_TOKEN__");
    calls.push({ method, path, body });
    const key = `${method} ${path}`;
    let handler = routes[key] ?? defaults[key];
    const jobMatch = path.match(/^\/api\/jobs\/([^/]+)(\/cancel)?$/);
    if (handler === undefined && jobMatch) {
      handler = () => {
        const job = jobs.get(jobMatch[1]);
        if (!job) throw { status: 404, error: "Unknown job" };
        if (jobMatch[2]) job.state = "cancelled";
        return { ...job };
      };
    }
    if (handler === undefined) throw new Error("Unexpected endpoint " + key);
    let status = 200;
    let result;
    try {
      result = typeof handler === "function" ? await handler(body, s) : handler;
    } catch (error) {
      if (!error.status) throw error;
      status = error.status;
      result = { error: error.error };
    }
    return { ok: status < 400, status, json: async () => structuredClone(result) };
  };
  const context = dom.getInternalVMContext();
  for (const file of ["app.js", "wizard.js", "selects.js", "jobs.js", "run.js"])
    vm.runInContext(fs.readFileSync(root + file, "utf8"), context);
  w.Jobs.config.interval = 5;
  w.Jobs.config.listInterval = 20;
  await tick();
  const report = {
    requirements: { workload: "coding", include_rankings: false, context: 8192 },
    candidates,
    shortlist: candidates.map((c) => c.id),
    notes: [],
    rejected: {},
    demo,
    hardware: {},
  };
  vm.runInContext(
    `report = ${JSON.stringify(report)}; renderResults();`,
    context,
  );
  const $ = (id) => w.document.getElementById(id);
  s.$ = $;
  s.run = (code) => vm.runInContext(code, context);
  s.panel = () => $("run-panel");
  s.step = (id) => w.document.querySelector(`[data-step="${id}"]`);
  s.stepText = (id) => s.step(id).textContent;
  s.buttonIn = (id, label) =>
    [...s.step(id).querySelectorAll("button")].find((b) =>
      label instanceof RegExp ? label.test(b.textContent) : b.textContent === label,
    );
  s.open = async (index = 0) => {
    w.document.querySelectorAll("[data-run]")[index].focus();
    w.document.querySelectorAll("[data-run]")[index].click();
    await until(() => s.panel().hasAttribute("open"), "panel opens");
    await until(
      () => w.document.querySelector('[data-step-toggle][aria-expanded="true"]'),
      "a step expands",
    );
  };
  s.expand = (id) => $(`run-step-${id}-toggle`).getAttribute("aria-expanded") === "true" || $(`run-step-${id}-toggle`).click();
  s.close = async () => {
    await tick();
    w.close();
  };
  return s;
}
const installed = {
  installed: true,
  source: "managed",
  directory: "/home/alice/.local/share/llm-config/runtime/b5000-cpu",
  version: "b5000",
  build: 5000,
  backend: "cpu",
  binaries: {},
  warnings: [],
};
const downloadedPlan = {
  files: [{ filename: "model-a.gguf", size_bytes: 4 * GiB, present: true, partial_bytes: 0 }],
  total_bytes: 4 * GiB,
  remaining_bytes: 0,
  disk_free: 100 * GiB,
  enough_space: true,
  local_copy: false,
};

test("cards show plain verdict badges and fall back for older reports", async () => {
  const s = await setup({
    candidates: [
      candidate({ id: "a", verdict: "runs_well", evidence: "measured" }),
      candidate({ id: "b", verdict: "runs_slowly", evidence: "estimated" }),
      candidate({ id: "c", verdict: "too_slow", evidence: "interpolated" }),
      candidate({ id: "d", verdict: "unknown", evidence: "none" }),
      candidate({ id: "e", verdict: undefined, evidence: undefined, verdict_text: undefined, tps: 30, speed_meets_target: true }),
      candidate({ id: "f", verdict: undefined, evidence: undefined, verdict_text: undefined }),
    ],
  });
  const badges = [...s.w.document.querySelectorAll(".card .verdict")].map((b) => b.textContent);
  assert.deepEqual(badges, [
    "Runs well",
    "Runs slowly · estimate",
    "Too slow · estimated from your tests at other lengths",
    "Not tested yet",
    "Runs well",
    "Not tested yet",
  ]);
  assert.match(s.$("cards").textContent, /Should write about 20 words a second/);
  assert.equal(s.w.document.querySelectorAll("[data-run]").length, 6);
  assert.match(s.w.document.querySelector("[data-run]").textContent, /Get it running/);
  await s.close();
});

test("Get it running opens a six-step panel with states and never shows file paths", async () => {
  const s = await setup({
    candidates: [candidate({ kv_cache_type: "q8_0" })],
    routes: { "GET /api/runtime": { ...installed, installed: false, directory: "/home/alice/secret" } },
  });
  await s.open();
  const steps = [...s.w.document.querySelectorAll(".run-step .step-name")].map((n) => n.textContent);
  assert.deepEqual(steps, ["Get the engine", "Download the model", "Test it", "Tune it", "Check quality", "Use it"]);
  assert.match(s.$("run-title").textContent, /Model A/);
  assert.match(s.panel().textContent, /compressed notes \(KV cache\)/i);
  await until(() => /Not installed/.test(s.stepText("runtime")), "runtime state");
  assert.match(s.stepText("download"), /Not downloaded/);
  assert.match(s.stepText("test"), /Not yet/);
  assert.match(s.stepText("test"), /Get the engine first/);
  assert.equal(s.$("run-step-runtime-toggle").getAttribute("aria-expanded"), "true");
  assert.doesNotMatch(s.panel().textContent, /\/home\/alice/);
  await s.close();
});

test("runtime install runs as a job with progress and shows in the tray", async () => {
  let installJob;
  const s = await setup({
    routes: {
      "POST /api/runtime/install": (body, s) => {
        installJob = s.job("runtime_install", "Install llama.cpp", { state: "queued" });
        return installJob;
      },
    },
  });
  await s.open();
  await until(() => s.buttonIn("runtime", "Install the engine"), "install button");
  s.buttonIn("runtime", "Install the engine").click();
  await until(() => installJob, "install requested");
  s.update(installJob.id, {
    state: "running",
    progress: { stage: "download", done: 10 * 1048576, total: 40 * 1048576, message: "Downloading llama.cpp" },
  });
  await until(() => /10 MiB of 40 MiB/.test(s.stepText("runtime")), "install progress");
  assert.equal(s.step("runtime").querySelector("progress").value, 0.25);
  assert.equal(s.$("jobs-tray").hidden, false);
  assert.match(s.$("jobs-list").textContent, /Install llama\.cpp/);
  s.update(installJob.id, { state: "done", result: installed });
  await until(() => /Engine ready · version b5000/.test(s.stepText("runtime")), "installed");
  assert.match(s.stepText("runtime"), /Done/);
  assert.doesNotMatch(s.panel().textContent, /\/home\/alice/);
  assert.match(s.stepText("test"), /Download the model first/);
  await s.close();
});

test("download shows bytes, speed and time left, and cancel removes partial files", async () => {
  let downloadJob;
  const s = await setup({
    routes: {
      "POST /api/downloads": (body, s) => {
        assert.deepEqual(body, { variant_id: "v1" });
        downloadJob = s.job("download", "Download Model A");
        return downloadJob;
      },
      "POST /api/downloads/remove": { freed_bytes: GiB },
    },
  });
  await s.open();
  s.expand("download");
  await until(() => s.buttonIn("download", "Download (4.0 GiB)"), "download button");
  assert.match(s.stepText("download"), /Free disk space: 100\.0 GiB/);
  s.buttonIn("download", "Download (4.0 GiB)").click();
  await until(() => downloadJob, "download started");
  s.update(downloadJob.id, {
    progress: { stage: "download", done: GiB, total: 4 * GiB, bytes_per_second: 50 * 1048576, eta_seconds: 61 },
  });
  await until(() => /1\.0 GiB of 4\.0 GiB/.test(s.stepText("download")), "progress text");
  assert.match(s.stepText("download"), /50 MiB\/s/);
  assert.match(s.stepText("download"), /about 1 min left/);
  assert.equal(s.step("download").querySelector("progress").value, 0.25);
  s.buttonIn("download", "Cancel download").click();
  await until(() => s.calls.some((c) => c.path === "/api/downloads/remove"), "partial removed");
  assert.ok(s.calls.some((c) => c.path === `/api/jobs/${downloadJob.id}/cancel`));
  await until(() => /Partial files removed/.test(s.stepText("download")), "cancel message");
  await s.close();
});

test("pausing keeps the partial download and offers resume", async () => {
  let downloadJob;
  let paused = false;
  const s = await setup({
    routes: {
      "POST /api/downloads": (body, s) => (downloadJob = s.job("download", "Download Model A")),
      "POST /api/downloads/plan": () => ({
        files: [],
        total_bytes: 4 * GiB,
        remaining_bytes: paused ? 3 * GiB : 4 * GiB,
        disk_free: 100 * GiB,
        enough_space: true,
        local_copy: false,
      }),
    },
  });
  await s.open();
  s.expand("download");
  await until(() => s.buttonIn("download", /^Download/), "download button");
  s.buttonIn("download", /^Download/).click();
  await until(() => s.buttonIn("download", "Pause"), "pause button");
  paused = true;
  s.buttonIn("download", "Pause").click();
  await until(() => s.buttonIn("download", "Resume download"), "resume");
  assert.match(s.stepText("download"), /Paused with 1\.0 GiB of 4\.0 GiB saved/);
  assert.ok(!s.calls.some((c) => c.path === "/api/downloads/remove"));
  await s.close();
});

test("a copy found on disk is reused instead of downloaded", async () => {
  const s = await setup({
    routes: {
      "POST /api/downloads/plan": { total_bytes: 4 * GiB, remaining_bytes: 4 * GiB, disk_free: 1e12, enough_space: true, local_copy: true, files: [] },
      "POST /api/downloads": (body, s) => s.job("download", "Use local copy", { state: "done", result: { reused: true, bytes: 4 * GiB } }),
    },
  });
  await s.open();
  s.expand("download");
  await until(() => s.buttonIn("download", "Use the copy on my disk"), "reuse button");
  assert.match(s.stepText("download"), /Found on your disk/);
  s.buttonIn("download", "Use the copy on my disk").click();
  await until(() => /Using the copy already on your disk/.test(s.stepText("download")), "reused");
  await s.close();
});

test("not enough disk space blocks the download with a plain reason", async () => {
  const s = await setup({
    routes: {
      "POST /api/downloads/plan": { total_bytes: 40 * GiB, remaining_bytes: 40 * GiB, disk_free: 10 * GiB, enough_space: false, local_copy: false, files: [] },
    },
  });
  await s.open();
  await until(() => /Not enough free disk space/.test(s.stepText("download")), "blocked");
  assert.match(s.stepText("download"), /Not yet/);
  await s.close();
});

test("test results show reading speed, first-word delay, writing speed and memory vs estimate", async () => {
  let testJob;
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": (body, s) => {
        assert.deepEqual(body, { candidate_id: "v1|now|8192|0", kind: "full" });
        return (testJob = s.job("test", "Test Model A"));
      },
    },
  });
  await s.open();
  await until(() => s.$("run-step-test-toggle").getAttribute("aria-expanded") === "true", "test step opens first");
  s.buttonIn("test", "Run the full test").click();
  await until(() => testJob, "test started");
  s.update(testJob.id, { progress: { stage: "speed", done: 1, total: 3, message: "Measuring speed" } });
  await until(() => /Measuring speed · step 1 of 3/.test(s.stepText("test")), "test progress");
  s.update(testJob.id, {
    state: "done",
    result: {
      verdict: "works",
      verdict_text: "Loads, answers correctly and writes at a comfortable pace.",
      smoke: { ok: true, load_seconds: 4.2, checks: [{ name: "Answer is correct", ok: true, detail: "" }], message: "" },
      speed: {
        summary: { pp_tps: 412.5, tps: 21.34, ttft_s: 0.42, depth: 7552 },
        memory: { estimated_ram_bytes: 4.8 * GiB, estimated_vram_bytes: 0, peak_ram_bytes: 5 * GiB, peak_vram_bytes: null, within_estimate: true, note: "" },
      },
    },
  });
  await until(() => /It works/.test(s.stepText("test")), "result");
  const text = s.stepText("test");
  assert.match(text, /Reading speed413 tokens\/s/);
  assert.match(text, /First-word delay0\.42 s/);
  assert.match(text, /Writing speed21\.3 tokens\/s/);
  assert.match(text, /Memory used5\.0 GiB RAMWe estimated 4\.8 GiB RAM/);
  assert.match(text, /within our estimate/);
  assert.match(text, /Passed: Answer is correct/);
  assert.match(text, /7,552 tokens already in the chat/);
  assert.match(text, /Done/);
  await s.close();
});

test("a failed quick check explains what went wrong and hides unmeasured speed", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": (body, s) =>
        s.job("test", "Quick check", {
          state: "done",
          result: {
            verdict: "failed",
            verdict_text: "The model did not answer.",
            smoke: { ok: false, stage_failed: "start", checks: [{ name: "model_loads", ok: false, detail: "Out of memory" }], message: "Not enough memory to load the model." },
            speed: null,
          },
        }),
    },
  });
  await s.open();
  await until(() => s.buttonIn("test", "Quick check only"), "quick button");
  s.buttonIn("test", "Quick check only").click();
  await until(() => /It didn’t work/.test(s.stepText("test")), "failure");
  assert.equal(s.calls.find((c) => c.path === "/api/test").body.kind, "smoke");
  assert.match(s.stepText("test"), /Failed: Out of memory/);
  assert.doesNotMatch(s.stepText("test"), /model_loads/, "machine check ids stay off the page");
  assert.match(s.stepText("test"), /Not enough memory to load the model/);
  assert.doesNotMatch(s.stepText("test"), /Reading speed/);
  await s.close();
});

test("tuning takes a time budget and shows before and after", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/tune": (body, s) =>
        s.job("tune", "Tune Model A", {
          state: "done",
          result: {
            best: { threads: 8, batch: 512, ubatch: 512, flash_attn: "on", cache_type_k: "f16", cache_type_v: "f16" },
            baseline: { tps: 20, pp_tps: 400 },
            best_result: { tps: 25, pp_tps: 480 },
            improvement: 1.25,
            trials: [{ status: "ok" }, { status: "ok" }, { status: "skipped_memory" }],
            stopped: "converged",
            notes: [
              "Compressed notes were not tried because you did not allow them.",
              "Stopped early: another round found nothing faster.",
            ],
          },
        }),
    },
  });
  await s.open();
  s.expand("tune");
  await until(() => s.buttonIn("tune", "Start tuning"), "tune button");
  const radios = s.step("tune").querySelectorAll('input[type="radio"]');
  assert.equal(radios.length, 3);
  assert.equal(radios[1].checked, true);
  radios[0].checked = true;
  radios[0].dispatchEvent(new s.w.Event("change"));
  const goal = s.step("tune").querySelector("select");
  goal.value = "balanced";
  goal.dispatchEvent(new s.w.Event("change"));
  s.buttonIn("tune", "Start tuning").click();
  await until(() => /25% faster/.test(s.stepText("tune")), "tune result");
  assert.deepEqual(s.calls.find((c) => c.path === "/api/tune").body, {
    candidate_id: "v1|now|8192|0",
    budget_seconds: 60,
    goal: "balanced",
  });
  const text = s.stepText("tune");
  assert.match(text, /Writing speed20\.0 → 25\.0 tokens\/s25% faster/);
  assert.match(text, /Reading speed400 → 480 tokens\/s20% faster/);
  assert.match(text, /CPU threads: 4 → 8/);
  assert.match(text, /Flash attention.*automatic → on/);
  assert.match(text, /Tried 2 settings\./);
  assert.equal((text.match(/Stopped/g) || []).length, 1, "the stop reason appears once, from the tuner");
  assert.match(text, /Compressed notes were not tried/);
  assert.match(text, /The best settings are saved/);
  await s.close();
});

test("a tune that wasn't saved never says it was, and the page adds no stop sentence of its own", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/tune": (body, s) =>
        s.job("tune", "Tune Model A", {
          state: "done",
          result: {
            best: { threads: 4 },
            baseline: { tps: 20, pp_tps: 400 },
            best_result: { tps: 20, pp_tps: 400 },
            improvement: 1,
            trials: [],
            stopped: "cancelled",
            notes: [],
            saved: false,
          },
        }),
    },
  });
  await s.open();
  s.expand("tune");
  await until(() => s.buttonIn("tune", "Start tuning"), "tune button");
  s.buttonIn("tune", "Start tuning").click();
  await until(() => /Tried 0 settings/.test(s.stepText("tune")), "tune result");
  const text = s.stepText("tune");
  assert.match(text, /These settings were not saved/);
  assert.doesNotMatch(text, /are saved on this computer/);
  assert.doesNotMatch(text, /Stopped/);
  await s.close();
});

test("export tabs load each format, copy to the clipboard and follow the platform toggle", async () => {
  const s = await setup({
    platform: "Win32",
    routes: {
      "POST /api/export": (body) => ({
        format: body.format,
        filename: body.platform === "windows" ? "start-model.ps1" : "start-model.sh",
        content: `${body.format} for ${body.platform}`,
        instructions: ["Save the file.", "Run it."],
        notes: ["Model is not downloaded yet; the path is a placeholder."],
      }),
    },
  });
  await s.open();
  s.expand("use");
  await until(() => s.step("use").querySelector("[role=tab]"), "tabs");
  const tabs = [...s.step("use").querySelectorAll("[role=tab]")];
  assert.deepEqual(
    tabs.map((t) => t.dataset.format),
    ["llama-server", "ollama", "lmstudio", "open-webui", "continue", "openai-python", "docker-compose"],
  );
  const pressed = s.step("use").querySelector('[aria-pressed="true"]');
  assert.equal(pressed.textContent, "Windows");
  tabs[0].click();
  await until(() => /llama-server for windows/.test(s.stepText("use")), "export content");
  assert.match(s.stepText("use"), /start-model\.ps1/);
  assert.match(s.stepText("use"), /Save the file\./);
  assert.match(s.stepText("use"), /placeholder/);
  s.step("use").querySelector('[data-copy="export"]').click();
  await until(() => s.copied.length === 1, "copied");
  assert.equal(s.copied[0], "llama-server for windows");
  await until(() => /Copied\./.test(s.stepText("use")), "copy status");
  // Arrow keys move between tabs and load the next format.
  s.step("use").querySelector('[aria-selected="true"]').dispatchEvent(
    new s.w.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }),
  );
  await until(() => /ollama for windows/.test(s.stepText("use")), "arrow tab");
  assert.equal(s.w.document.activeElement.dataset.format, "ollama");
  s.buttonIn("use", "Mac or Linux").click();
  await until(() => /ollama for posix/.test(s.stepText("use")), "platform switch");
  assert.equal(s.calls.filter((c) => c.path === "/api/export" && c.method === "POST").at(-1).body.platform, "posix");
  await s.close();
});

test("copy falls back when the clipboard API is unavailable", async () => {
  const s = await setup({
    clipboard: false,
    routes: { "POST /api/export": { format: "ollama", filename: "Modelfile", content: "FROM x", instructions: [], notes: [] } },
  });
  let copiedText = null;
  s.w.document.execCommand = (command) => {
    copiedText = command === "copy" ? s.w.document.activeElement.value : null;
    return true;
  };
  await s.open();
  s.expand("use");
  await until(() => s.step("use").querySelector('[data-format="ollama"]'), "tabs");
  s.step("use").querySelector('[data-format="ollama"]').click();
  await until(() => s.step("use").querySelector('[data-copy="export"]'), "copy button");
  s.step("use").querySelector('[data-copy="export"]').click();
  await until(() => copiedText === "FROM x", "fallback copy");
  await s.close();
});

test("serve starts and stops and shows the OpenAI address with a copy button", async () => {
  let running = false;
  const status = () => ({
    running,
    base_url: running ? "http://127.0.0.1:8080" : null,
    openai_base_url: running ? "http://127.0.0.1:8080/v1" : null,
    model: running ? "Model A" : null,
    candidate_id: running ? "v1|now|8192|0" : null,
  });
  let serveJob;
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "GET /api/serve": status,
      "POST /api/serve/start": (body, s) => {
        assert.deepEqual(body, { candidate_id: "v1|now|8192|0" });
        return (serveJob = s.job("serve", "Start Model A", { progress: { stage: "loading", message: "Loading the model" } }));
      },
      "POST /api/serve/stop": () => {
        running = false;
        return status();
      },
    },
  });
  await s.open();
  s.expand("use");
  await until(() => s.buttonIn("use", "Start server"), "start button");
  s.buttonIn("use", "Start server").click();
  await until(() => /Loading the model/.test(s.stepText("use")), "loading");
  running = true;
  s.update(serveJob.id, { state: "done", result: status() });
  await until(() => /Server running/.test(s.stepText("use")), "running");
  assert.equal(s.step("use").querySelector(".base-url").textContent, "http://127.0.0.1:8080/v1");
  s.step("use").querySelector('[data-copy="url"]').click();
  await until(() => s.copied.includes("http://127.0.0.1:8080/v1"), "url copied");
  s.buttonIn("use", "Stop server").click();
  await until(() => s.buttonIn("use", "Start server"), "stopped");
  assert.doesNotMatch(s.stepText("use"), /8080/);
  await s.close();
});

test("demo mode explains that downloads, tests and serving are unavailable", async () => {
  const demoError = () => {
    throw { status: 409, error: "Demo mode never downloads models." };
  };
  const s = await setup({
    demo: true,
    candidates: [candidate({ demo: true })],
    routes: {
      "POST /api/downloads/plan": demoError,
      "GET /api/runtime": demoError,
    },
  });
  await s.open();
  await until(() => /Not available in demo mode/.test(s.stepText("download")), "demo download");
  assert.match(s.stepText("download"), /Demo/);
  assert.match(s.stepText("test"), /Not available in demo mode/);
  assert.match(s.stepText("tune"), /Not available in demo mode/);
  s.expand("use");
  await until(() => s.step("use").querySelector("[role=tab]"), "use step");
  assert.match(s.stepText("use"), /Not available in demo mode/);
  assert.equal(s.buttonIn("use", "Start server"), undefined);
  assert.match(s.stepText("runtime"), /Demo/);
  assert.equal(s.buttonIn("runtime", "Install the engine"), undefined, "no button that can only fail");
  assert.equal(s.$("run-content").textContent.split("The demo models are made up").length - 1, 1, "long demo note shown once");
  await s.close();
});

test("other 409 errors keep the server's plain message", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": () => {
        throw { status: 409, error: "Compare again first." };
      },
    },
  });
  await s.open();
  await until(() => s.buttonIn("test", "Run the full test"), "test button");
  s.buttonIn("test", "Run the full test").click();
  await until(() => /Compare again first\./.test(s.stepText("test")), "409 text");
  assert.match(s.stepText("test"), /Needs attention/);
  await s.close();
});

test("jobs tray lists running work from the server and cancels it", async () => {
  const s = await setup();
  s.job("download", "Download Model B", { progress: { done: 2 * GiB, total: 8 * GiB, bytes_per_second: 0 } });
  await s.w.Jobs.list();
  assert.equal(s.$("jobs-tray").hidden, false);
  assert.match(s.$("jobs-count").textContent, /1 running/);
  assert.match(s.$("jobs-list").textContent, /Download Model B/);
  assert.match(s.$("jobs-list").textContent, /2\.0 GiB of 8\.0 GiB/);
  s.$("jobs-list").querySelector("button").click();
  await until(() => /Stopped/.test(s.$("jobs-list").textContent), "cancelled in tray");
  assert.ok(s.calls.some((c) => c.path === "/api/jobs/job-1/cancel" && c.method === "POST"));
  s.$("jobs-toggle").click();
  assert.equal(s.$("jobs-list").hidden, true);
  assert.equal(s.$("jobs-toggle").getAttribute("aria-expanded"), "false");
  await s.close();
});

test("Jobs.watch polls until done and reports a vanished job plainly", async () => {
  const s = await setup();
  const job = s.job("test", "Test");
  const seen = [];
  s.w.Jobs.watch(job.id, (j) => seen.push(j.state));
  await until(() => seen.length >= 1, "first poll");
  s.update(job.id, { state: "done" });
  await until(() => seen.at(-1) === "done", "done");
  const count = seen.length;
  await tick();
  assert.equal(seen.length, count, "stops polling after the job ends");
  const lost = [];
  s.w.Jobs.watch("job-404", (j) => lost.push(j));
  await until(() => lost.length === 1, "lost");
  assert.equal(lost[0].state, "failed");
  assert.match(lost[0].error, /no longer tracked/);
  await s.close();
});

test("panel is keyboard navigable and returns focus on close", async () => {
  const s = await setup();
  await s.open();
  const opener = s.w.document.querySelector("[data-run]");
  const toggles = [...s.w.document.querySelectorAll("[data-step-toggle]")];
  const key = (name) =>
    s.w.document.activeElement.dispatchEvent(
      new s.w.KeyboardEvent("keydown", { key: name, bubbles: true, cancelable: true }),
    );
  toggles[0].focus();
  key("ArrowDown");
  assert.equal(s.w.document.activeElement, toggles[1]);
  key("End");
  assert.equal(s.w.document.activeElement, toggles[5]);
  key("ArrowDown");
  assert.equal(s.w.document.activeElement, toggles[0]);
  key("ArrowUp");
  assert.equal(s.w.document.activeElement, toggles[5]);
  key("Home");
  assert.equal(s.w.document.activeElement, toggles[0]);
  toggles[2].click();
  assert.equal(toggles[2].getAttribute("aria-expanded"), "true");
  assert.equal(s.$("run-step-test-body").hidden, false);
  assert.equal(toggles[0].getAttribute("aria-expanded"), "false");
  assert.equal(s.$(toggles[2].getAttribute("aria-controls")).getAttribute("role"), "region");
  key("Escape");
  assert.equal(s.panel().hasAttribute("open"), false);
  assert.equal(s.w.document.activeElement, opener);
  await s.close();
});

test("quality step mounts QualityPanel with the agreed context, or a placeholder", async () => {
  const s = await setup({
    candidates: [candidate(), candidate({ id: "b", name: "Model B" })],
    routes: { "POST /api/quality/quiz": (body, s) => s.job("quiz", "Quiz", { body }) },
  });
  let mounted = null;
  s.w.QualityPanel = { mount: (container, ctx) => (mounted = { container, ctx }) };
  await s.open();
  s.expand("quality");
  await until(() => mounted, "mounted");
  assert.equal(mounted.ctx.candidate.id, "v1|now|8192|0");
  assert.equal(mounted.ctx.candidates.length, 2);
  assert.equal(mounted.ctx.workload, "coding");
  assert.equal(typeof mounted.ctx.onJob, "function");
  const job = await mounted.ctx.api("/api/quality/quiz", { method: "POST", body: JSON.stringify({ candidate_id: "b" }) });
  assert.equal(job.kind, "quiz");
  assert.deepEqual(s.calls.at(-1).body, { candidate_id: "b" });
  await mounted.ctx.api("/api/quality/quiz", { candidate_id: "c" });
  assert.deepEqual(s.calls.at(-1).body, { candidate_id: "c" });
  mounted.ctx.onJob(job);
  assert.match(s.$("jobs-list").textContent, /Quiz/);
  await s.close();
  const plain = await setup();
  await plain.open();
  plain.expand("quality");
  await until(() => /aren’t available in this version yet/.test(plain.stepText("quality")), "placeholder");
  await plain.close();
});

test("job errors never show file paths and repeat clicks start one job", async () => {
  let starts = 0;
  let release;
  const gate = new Promise((resolve) => (release = resolve));
  const s = await setup({
    routes: {
      "POST /api/downloads": async (body, s) => {
        starts++;
        await gate;
        return s.job("download", "Download Model A", {
          state: "failed",
          error: "[Errno 28] No space left on device: '/home/alice/models/model-a.gguf.part'",
        });
      },
    },
  });
  await s.open();
  s.expand("download");
  await until(() => s.buttonIn("download", /^Download/), "download button");
  const start = s.buttonIn("download", /^Download/);
  start.click();
  start.click();
  release();
  await until(() => /No space left on device/.test(s.stepText("download")), "error");
  assert.equal(starts, 1);
  assert.doesNotMatch(s.panel().textContent + s.$("jobs-tray").textContent, /\/home\/alice/);
  assert.match(s.stepText("download"), /a file in the app’s folder/);
  await s.close();
});

test("a download already running after a reload is picked up and finishes", async () => {
  let plans = 0;
  const s = await setup({
    routes: {
      "POST /api/downloads/plan": () => ({
        files: [],
        total_bytes: 4 * GiB,
        remaining_bytes: plans++ === 0 ? 3 * GiB : 0,
        disk_free: 100 * GiB,
        enough_space: true,
        local_copy: false,
      }),
    },
  });
  const job = s.job("download", "Download Model A", {
    subject: { variant_id: "v1" },
    progress: { done: GiB, total: 4 * GiB },
  });
  await s.open();
  await until(() => /Working/.test(s.stepText("download")), "reattached");
  assert.equal(s.$("run-step-download-toggle").getAttribute("aria-expanded"), "true");
  s.update(job.id, { state: "done", result: { reused: false, bytes: 4 * GiB } });
  await until(() => /Done/.test(s.stepText("download")), "finished after reattach");
  await s.close();
});

test("an unreachable app gives a plain message instead of a browser error", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": () => {
        throw new TypeError("Failed to fetch");
      },
    },
  });
  await s.open();
  await until(() => s.buttonIn("test", "Run the full test"), "test button");
  s.buttonIn("test", "Run the full test").click();
  await until(() => /didn’t respond/.test(s.stepText("test")), "plain network error");
  assert.doesNotMatch(s.stepText("test"), /Failed to fetch/);
  await s.close();
});

// ---- fix-up round: shapes the server really sends ----

test("progress text uses the right unit for each stage", async () => {
  const s = await setup();
  const d = (progress, kind = "test") => s.w.Jobs.describe({ kind, state: "running", progress });
  assert.equal(d({ stage: "tune", done: 75, total: 300, message: "Trying settings" }, "tune").text, "Trying settings · 1 min of 5 min used");
  assert.equal(d({ stage: "tune", done: 75, total: 300 }, "tune").fraction, 0.25);
  const loading = d({ stage: "loading", done: 12.4, total: 300, message: "Loading the model" }, "serve");
  assert.equal(loading.text, "Loading the model · 12 s so far");
  assert.equal(loading.fraction, null, "a time limit is not a progress bar");
  assert.equal(d({ stage: "extract", done: 0, total: null, message: "Unpacking llama.cpp" }, "runtime_install").text, "Unpacking llama.cpp");
  assert.match(d({ stage: "download", done: 1048576 * 5, total: 1048576 * 10, bytes_per_second: 1048576 }, "runtime_install").text, /5 MiB of 10 MiB · 1 MiB\/s/);
  assert.match(d({ stage: "verifying", done: 2 * GiB, total: 4 * GiB, message: "Checking" }, "local_scan").text, /2\.0 GiB of 4\.0 GiB/);
  assert.equal(d({ stage: "bench", done: 1, total: 3, message: "Measuring speed" }).text, "Measuring speed · step 1 of 3");
  assert.equal(d({ stage: "x", message: "See /home/alice/models/a.gguf" }).text, "See a file in the app’s folder");
  await s.close();
});

test("tuning counts only new settings, shows batch changes and doesn't repeat notes", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/tune": (body, s) =>
        s.job("tune", "Tune Model A", {
          state: "done",
          result: {
            best: { threads: 4, batch: 2048, ubatch: 512, flash_attn: "auto", cache_type_k: "f16", cache_type_v: "f16" },
            baseline: { tps: 20, pp_tps: 400 },
            best_result: { tps: 20.2, pp_tps: 700 },
            improvement: 1.3,
            trials: [
              { status: "ok", step: "baseline" },
              { status: "ok", step: "batch" },
              { status: "ok", step: "batch" },
              { status: "failed", step: "threads" },
              { status: "ok", step: "confirm" },
            ],
            stopped: "budget",
            notes: ["Overall, the tuned settings made reading faster than where we started.", "Stopped because the time budget ran out."],
          },
        }),
    },
    candidates: [candidate({ launch: { threads: 4, cache_type_k: "f16", cache_type_v: "f16", flash_attn: "auto" } })],
  });
  await s.open();
  s.expand("tune");
  await until(() => s.buttonIn("tune", "Start tuning"), "tune button");
  s.buttonIn("tune", "Start tuning").click();
  await until(() => /Tried/.test(s.stepText("tune")), "tune result");
  const text = s.stepText("tune");
  assert.match(text, /Tried 2 settings\./);
  assert.match(text, /Text read in one go: default → 2048/);
  assert.equal(text.match(/Stopped/g).length, 1, "stop reason shown once");
  await s.close();
});

test("a finished speed test can be shared: the JSON is shown before any link", async () => {
  let shareBody;
  const share = {
    json: '{\n "schema": 1,\n "records": [{"model": "<script>alert(1)</script>"}]\n}',
    issue_url: "https://github.com/Eyad-3D/LLM-Configurator/issues/new?title=x",
    fits_in_url: false,
    records: 1,
  };
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": (body, s) =>
        s.job("test", "Test", {
          state: "done",
          result: {
            verdict: "works",
            verdict_text: "It works.",
            smoke: { ok: true, checks: [{ name: "not_empty", ok: true, detail: "The model wrote an answer." }] },
            speed: {
              measurement: { id: "a1b2c3d4e5f6" },
              summary: { pp_tps: 100, tps: 20, ttft_s: 0.5, depth: 1000 },
              memory: {},
            },
          },
        }),
      "POST /api/community/share": (body) => {
        shareBody = body;
        return share;
      },
    },
  });
  await s.open();
  await until(() => s.buttonIn("test", "Run the full test"), "test button");
  s.buttonIn("test", "Run the full test").click();
  await until(() => s.buttonIn("test", "Show what would be shared"), "share offer");
  assert.equal(s.step("test").querySelector("a"), null, "no link before the person has seen the JSON");
  s.buttonIn("test", "Show what would be shared").click();
  await until(() => s.step("test").querySelector(".share-json"), "json shown");
  assert.deepEqual(shareBody, { measurement_ids: ["a1b2c3d4e5f6"] });
  assert.equal(s.step("test").querySelector(".share-json").textContent, share.json);
  assert.equal(s.step("test").querySelector("script"), null);
  assert.match(s.stepText("test"), /too long to fit in a link/);
  const link = s.step("test").querySelector("a");
  assert.equal(link.getAttribute("href"), share.issue_url);
  assert.equal(link.getAttribute("target"), "_blank");
  assert.match(link.getAttribute("rel"), /noopener/);
  s.step("test").querySelector('[data-copy="share"]').click();
  await until(() => s.copied.includes(share.json), "json copied");
  await s.close();
});

test("a share link that isn't GitHub is never shown", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": (body, s) =>
        s.job("test", "Test", {
          state: "done",
          result: { verdict: "works", smoke: { ok: true, checks: [] }, speed: { measurement: { id: "abcd1234" }, summary: {}, memory: {} } },
        }),
      "POST /api/community/share": { json: "{}", issue_url: "javascript:alert(1)", fits_in_url: true, records: 1 },
    },
  });
  await s.open();
  await until(() => s.buttonIn("test", "Run the full test"), "test button");
  s.buttonIn("test", "Run the full test").click();
  await until(() => s.buttonIn("test", "Show what would be shared"), "share offer");
  s.buttonIn("test", "Show what would be shared").click();
  await until(() => s.step("test").querySelector(".share-json"), "json shown");
  assert.equal(s.step("test").querySelector("a"), null);
  await s.close();
});

test("a server that stopped by itself says why, without paths", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "GET /api/serve": { running: false, starting: false, error: "llama-server exited: cannot open /home/alice/m.gguf", base_url: null },
    },
  });
  await s.open();
  s.expand("use");
  await until(() => /The last server stopped/.test(s.stepText("use")), "error shown");
  assert.doesNotMatch(s.stepText("use"), /alice/);
  await s.close();
});

test("hardware shows the CPU name, shared memory and graphics chips we can't read", async () => {
  const state = (hw) => () => ({
    hardware: { ram_available: 16e9, ram_total: 32e9, cpu_percent: 1, cores: 8, threads: 16, disk_free: 100e9, processes: [], ...hw },
    demo: false,
    status: { variants: 1, timestamp: new Date().toISOString(), warnings: [] },
    definitions: [],
    scores: [],
  });
  let s = await setup({
    routes: {
      "GET /api/state": state({
        cpu: "x86_64",
        cpu_name: "AMD Ryzen 7 7840U",
        gpus: [],
        other_gpus: [{ name: "Radeon 780M", available: null }],
        warnings: ["Graphics found, but their free memory cannot be read."],
      }),
    },
  });
  let text = s.$("hardware").textContent;
  assert.match(text, /AMD Ryzen 7 7840U/);
  assert.match(text, /A graphics chip was found, but free memory can’t be read, so estimates use RAM/);
  assert.match(text, /Graphics found, but their free memory cannot be read/);
  assert.doesNotMatch(text, /NVIDIA/);
  await s.close();
  s = await setup({
    routes: {
      "GET /api/state": state({
        cpu: "arm",
        cpu_name: "Apple M3",
        gpus: [{ index: 0, name: "Apple M3", unified: true, available: 10 * GiB, total: 12 * GiB, backend: "metal" }],
      }),
    },
  });
  text = s.$("hardware").textContent;
  assert.match(text, /Shared with system RAM/);
  await s.close();
});

test("MoE cards say Experts on CPU; details use the real memory total and KV setting; hostile text stays text", async () => {
  const evil = '<img src=x onerror="window.pwned=1">';
  const s = await setup({
    candidates: [
      candidate({
        name: evil,
        quant: evil,
        verdict_text: evil,
        explanation: evil,
        mode: "split",
        n_cpu_moe: 12,
        kv_cache_type: "q8_0",
        memory_total_bytes: 9 * GiB,
        ram_headroom_bytes: GiB,
        kv_bytes: GiB,
        file_bytes: 5 * GiB,
        memory_max_context: 32768,
        metadata_date: evil,
        score_source: "javascript:alert(1)",
        quality_evidence: evil,
      }),
    ],
  });
  assert.match(s.$("cards").textContent, /Experts on CPU/);
  assert.equal(s.$("cards").querySelector("img"), null);
  s.w.document.querySelector("[data-detail]").click();
  const detail = s.$("detail_content");
  assert.match(detail.textContent, /Estimated memory in total9\.0 GiB/);
  assert.match(detail.textContent, /Conversation memory \(half size\)/);
  assert.match(detail.textContent, /"kv_cache": "q8_0"/);
  assert.equal(detail.querySelector("img"), null);
  assert.equal(detail.querySelector("a"), null, "non-https sources are not linked");
  assert.equal(s.w.pwned, undefined);
  await s.close();
});

test("cards show nearby-test and community speeds when that's the evidence", async () => {
  const s = await setup({
    candidates: [
      candidate({ id: "a", verdict: "unknown", evidence: "interpolated", speed_interpolated: { tps: 18.25 } }),
      candidate({ id: "b", verdict: "unknown", evidence: "community", community: { median_tps: 30, n: 3 } }),
    ],
  });
  const text = s.$("cards").textContent;
  assert.match(text, /~18\.3 tok\/s/);
  assert.match(text, /Reported by 3 similar computers/);
  assert.match(text, /Not tested yet · estimated from your tests at other lengths/);
  await s.close();
});

test("the comparison request carries the conversation memory setting", async () => {
  const s = await setup();
  assert.equal(s.run("requirements().kv_cache_type"), "f16");
  s.$("kv_cache_type").value = "q4_0";
  assert.equal(s.run("requirements().kv_cache_type"), "q4_0");
  await s.close();
});

test("community results can be downloaded from the results page", async () => {
  let importJob;
  const s = await setup({
    routes: {
      "GET /api/community": { records: [], source: null, fetched_at: null },
      "POST /api/community/import": (body, s) => {
        assert.deepEqual(body, {});
        return (importJob = s.job("community_import", "Import community results"));
      },
    },
  });
  s.$("community-box").open = true;
  s.$("community-box").dispatchEvent(new s.w.Event("toggle"));
  await until(() => /No shared results/.test(s.$("community-status").textContent), "status");
  s.$("community-import").click();
  await until(() => importJob, "import started");
  s.update(importJob.id, { state: "done", result: { count: 120, rejected: 2, source: "x", fetched_at: "2026-09-01T00:00:00Z" } });
  await until(() => /Got 120 shared results \(2 skipped/.test(s.$("community-status").textContent), "done");
  assert.match(s.$("community-status").textContent, /Recalculate/);
  await s.close();
});

test("many 'couldn't reach' warnings become one plain sentence", async () => {
  const s = await setup();
  const list = Array.from({ length: 44 }, (_, i) => `org/model-${i}: Metadata unavailable (URLError); cached results remain available`);
  const text = s.run(`summarizeWarnings(${JSON.stringify([...list, "Scores need a key"])})`);
  assert.match(text, /Couldn’t get the latest details for 44 models/);
  assert.match(text, /Scores need a key/);
  assert.doesNotMatch(text, /URLError|org\/model-3/);
  await s.close();
});

// ---- round 3 ----

function shareSetup(share) {
  return setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": downloadedPlan,
      "POST /api/test": (body, s) =>
        s.job("test", "Test", {
          state: "done",
          result: { verdict: "works", smoke: { ok: true, checks: [] }, speed: { measurement: { id: "abcd1234" }, summary: {}, memory: {} } },
        }),
      "POST /api/community/share": share,
    },
  });
}
async function openShare(s) {
  await s.open();
  await until(() => s.buttonIn("test", "Run the full test"), "test button");
  s.buttonIn("test", "Run the full test").click();
  await until(() => s.buttonIn("test", "Show what would be shared"), "share offer");
  s.buttonIn("test", "Show what would be shared").click();
  await until(() => !/Preparing/.test(s.stepText("test")) && !s.buttonIn("test", "Show what would be shared"), "share shown");
}

test("sharing says how many results were left out for being from different hardware", async () => {
  const s = await shareSetup({ json: "{}", issue_url: "https://github.com/x/y/issues/new", fits_in_url: true, records: 1, skipped: 2 });
  await openShare(s);
  await until(() => s.step("test").querySelector(".share-json"), "json shown");
  assert.match(s.stepText("test"), /2 results were measured on different hardware and were left out/);
  await s.close();
  const t = await shareSetup({ json: "{}", issue_url: "https://github.com/x/y/issues/new", fits_in_url: true, records: 1, skipped: 0 });
  await openShare(t);
  await until(() => t.step("test").querySelector(".share-json"), "json shown");
  assert.doesNotMatch(t.stepText("test"), /left out/);
  await t.close();
});

test("when every result is left out there is nothing to copy or post", async () => {
  const s = await shareSetup({ json: "", issue_url: "https://github.com/x/y/issues/new", fits_in_url: true, records: 0, skipped: 1 });
  await openShare(s);
  await until(() => /nothing to share/.test(s.stepText("test")), "nothing");
  assert.match(s.stepText("test"), /different hardware/);
  assert.equal(s.step("test").querySelector("a"), null);
  assert.equal(s.step("test").querySelector(".share-json"), null);
  await s.close();
});

test("a model from the person's own disk is never offered as a download", async () => {
  const s = await setup({
    routes: {
      "GET /api/runtime": installed,
      "POST /api/downloads/plan": { ...downloadedPlan, remaining_bytes: 4 * GiB, local: true, local_copy: false },
    },
  });
  await s.open();
  s.expand("download");
  await until(() => /isn’t where it was found/.test(s.stepText("download")), "missing file");
  assert.equal(s.buttonIn("download", "Download"), undefined);
  assert.ok(s.buttonIn("download", "Check again"));
  assert.match(s.stepText("download"), /Your file is missing/);
  await s.close();
});

test("a short tune reads as a short tune, not a measurement or a plain 'not tested'", async () => {
  const s = await setup({
    candidates: [
      candidate({
        id: "a",
        verdict: "unknown",
        evidence: "tuned",
        tps: null,
        tuned: { tps: 31.04, verified: false, same_placement: true },
      }),
    ],
  });
  const text = s.$("cards").textContent;
  assert.match(text, /~31\.0 tok\/s/);
  assert.match(text, /Tuned with a short test/);
  assert.match(text, /Not tested yet · tuned with a short test/);
  assert.doesNotMatch(text, /Measured/);
  assert.doesNotMatch(text, /Speed verified locally/);
  await s.close();
});

test("details list every speed number with its own honest label", async () => {
  const s = await setup({
    candidates: [
      candidate({
        verdict: "runs_well",
        evidence: "tuned",
        tps: 25,
        tuned: { tps: 25, verified: true },
        speed_interpolated: { tps: 18 },
        community: { median_tps: 30, n: 1 },
        ram_headroom_bytes: GiB,
        kv_bytes: GiB,
        file_bytes: GiB,
        memory_max_context: 8192,
      }),
    ],
  });
  assert.match(s.$("cards").textContent, /Measured after tuning/);
  s.w.document.querySelector("[data-detail]").click();
  const text = s.$("detail_content").textContent;
  assert.match(text, /Measured after tuning, on this computer25\.0 tok\/s/);
  assert.match(text, /Estimated from your tests at other lengths \(not a test at this length\)~18\.0 tok\/s/);
  assert.match(text, /Other people’s results: middle value from 1 similar computer \(not this one\)~30\.0 tok\/s/);
  assert.doesNotMatch(text, /Tuning run/, "a verified tune isn't listed twice");
  await s.close();
});

test("every unreadable graphics chip is listed with its reason, and names stay text", async () => {
  const s = await setup({
    routes: {
      "GET /api/state": () => ({
        hardware: {
          ram_available: 16e9, ram_total: 32e9, cpu_percent: 1, cores: 8, threads: 16, disk_free: 100e9, processes: [],
          cpu: "x86_64",
          gpus: [{ index: 0, name: '<img src=x onerror="window.pwned=1">', available: 8 * GiB, total: 12 * GiB }],
          other_gpus: [
            { name: "Intel Arc A380", reason: "Intel drivers do not report free graphics memory in a way we can read." },
            { name: "<b>Radeon</b>" },
          ],
          warnings: [],
        },
        demo: false,
        status: { variants: 1, timestamp: new Date().toISOString(), warnings: [] },
        definitions: [],
        scores: [],
      }),
    },
  });
  const box = s.$("hardware");
  assert.match(box.textContent, /Also found Intel Arc A380 \(not counted in memory estimates\): Intel drivers do not report/);
  assert.match(box.textContent, /Also found <b>Radeon<\/b> .*free memory can’t be read/);
  assert.equal(box.querySelector("img, b"), null);
  assert.match(box.textContent, /<img src=x/);
  assert.equal(s.w.pwned, undefined);
  await s.close();
});
