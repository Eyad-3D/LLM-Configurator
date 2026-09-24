const { test } = require("node:test");
const assert = require("node:assert/strict");
const { JSDOM } = require("jsdom");
const fs = require("node:fs");
const vm = require("node:vm");
const root = "src/llm_configurator/static/";
const tick = (ms = 20) => new Promise((resolve) => setTimeout(resolve, ms));
async function until(check, label = "condition") {
  for (let i = 0; i < 200; i++) {
    if (check()) return;
    await tick(5);
  }
  throw new Error("Timed out waiting for " + label);
}

const candidates = [
  { id: "c-big", name: "Alpha 8B", quant: "Q8_0", mode: "gpu", variant_id: "alpha-q8", repo: "org/alpha", file_bytes: 8.5e9, context: 8192 },
  { id: "c-mid", name: "Alpha 8B", quant: "Q4_K_M", mode: "gpu", variant_id: "alpha-q4", repo: "org/alpha", file_bytes: 4.9e9, context: 8192 },
  { id: "c-small", name: "Alpha 8B", quant: "Q2_K", mode: "cpu", variant_id: "alpha-q2", repo: "org/alpha", file_bytes: 3.1e9, context: 8192 },
  { id: "c-other", name: "Beta 3B", quant: "Q4_K_M", mode: "cpu", variant_id: "beta-q4", repo: "org/beta", file_bytes: 2e9, context: 8192 },
];

// routes: { "METHOD path": value | (body, calls) => value }. A function may throw.
function setup(routes = {}, jobs = {}) {
  const dom = new JSDOM(
    '<!doctype html><html><head><meta name="session-token" content="t0k"></head><body><div id="q"></div><div id="l"></div></body></html>',
    { url: "http://127.0.0.1:8765", runScripts: "outside-only" },
  );
  const w = dom.window;
  const calls = [];
  const jobEvents = [];
  const polls = {};
  const api = async (path, options) => {
    const method = options?.method || "GET";
    const body = options?.body === undefined ? undefined : JSON.parse(JSON.stringify(options.body));
    calls.push({ method, path, body });
    const jobMatch = path.match(/^\/api\/jobs\/([^/]+)$/);
    if (method === "GET" && jobMatch) {
      const seq = jobs[decodeURIComponent(jobMatch[1])];
      if (!seq) throw new Error("Unknown job " + path);
      polls[jobMatch[1]] = (polls[jobMatch[1]] || 0) + 1;
      return seq[Math.min(polls[jobMatch[1]] - 1, seq.length - 1)];
    }
    const key = `${method} ${path}`;
    if (!(key in routes)) {
      if (path === "/api/quality/results") return { results: [] };
      if (path === "/api/local-models") return { files: [], locations: [] };
      throw new Error("Unexpected " + key);
    }
    const route = routes[key];
    return typeof route === "function" ? route(body, calls) : route;
  };
  vm.runInContext(fs.readFileSync(root + "quality.js", "utf8"), dom.getInternalVMContext());
  const ctx = {
    api,
    candidate: candidates[1],
    candidates,
    workload: "coding",
    onJob: (job) => jobEvents.push(job.state),
    pollMs: 1,
  };
  const $ = (sel, from = w.document) => from.querySelector(sel);
  const $$ = (sel, from = w.document) => [...from.querySelectorAll(sel)];
  const byText = (text, from = w.document, tag = "button") =>
    $$(tag, from).find((b) => b.textContent.trim() === text);
  return { w, dom, calls, jobEvents, ctx, $, $$, byText, polls };
}
const job = (id, state, extra = {}) => ({ id, kind: "x", title: "x", state, progress: {}, result: null, error: null, ...extra });
const tabPanel = (t, name) => {
  const tab = t.$$('[role="tab"]').find((x) => x.textContent === name);
  tab.click();
  return t.w.document.getElementById(tab.getAttribute("aria-controls"));
};
const keydown = (t, target, key) =>
  target.dispatchEvent(new t.w.KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));

test("mounts both panels and unmount stops polling and clears the DOM", async () => {
  let polled = 0;
  const t = setup({ "POST /api/quality/quiz": job("j1", "running") }, {});
  t.ctx.api = ((inner) => async (path, o) => {
    if (path === "/api/jobs/j1") {
      polled++;
      return job("j1", "running", { progress: { stage: "quiz", done: 1, total: 10, message: "Question 1 of 10" } });
    }
    return inner(path, o);
  })(t.ctx.api);
  const q = t.$("#q");
  const handle = t.w.QualityPanel.mount(q, t.ctx);
  assert.equal(typeof handle.unmount, "function");
  assert.equal(t.$$('[role="tab"]', q).length, 3);
  assert.deepEqual(t.$$('[role="tab"]', q).map((x) => x.textContent), ["Quick quiz", "Try my prompts", "Compression check"]);
  assert.equal(t.$$('[role="tabpanel"]', q).filter((p) => !p.hidden).length, 1);
  const local = t.w.LocalModels.mount(t.$("#l"), t.ctx);
  assert.ok(t.$("#l .qp-local"));
  t.byText("Start quiz", q).click();
  await until(() => polled >= 2, "polling");
  handle.unmount();
  local.unmount();
  assert.equal(q.children.length, 0);
  assert.equal(t.$("#l").children.length, 0);
  const before = polled;
  await tick(30);
  assert.equal(polled, before, "no polling after unmount");
  // Remounting the same container replaces the old instance.
  t.w.QualityPanel.mount(q, t.ctx);
  t.w.QualityPanel.mount(q, t.ctx);
  assert.equal(t.$$(".qp", q).length, 1);
});

test("tabs use roving focus with arrow, Home and End keys", () => {
  const t = setup();
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const tabs = t.$$('[role="tab"]');
  assert.equal(t.$('[role="tablist"]').getAttribute("aria-label"), "Quality checks");
  assert.deepEqual(tabs.map((x) => x.tabIndex), [0, -1, -1]);
  tabs[0].focus();
  keydown(t, tabs[0], "ArrowRight");
  assert.equal(t.w.document.activeElement, tabs[1]);
  assert.equal(tabs[1].getAttribute("aria-selected"), "true");
  assert.deepEqual(tabs.map((x) => x.tabIndex), [-1, 0, -1]);
  const panel = t.w.document.getElementById(tabs[1].getAttribute("aria-controls"));
  assert.equal(panel.hidden, false);
  assert.equal(panel.getAttribute("aria-labelledby"), tabs[1].id);
  keydown(t, tabs[1], "End");
  assert.equal(t.w.document.activeElement, tabs[2]);
  keydown(t, tabs[2], "ArrowRight");
  assert.equal(t.w.document.activeElement, tabs[0], "wraps around");
  keydown(t, tabs[0], "ArrowLeft");
  assert.equal(t.w.document.activeElement, tabs[2]);
  keydown(t, tabs[2], "Home");
  assert.equal(t.w.document.activeElement, tabs[0]);
  assert.equal(t.$$('[role="tabpanel"]').filter((p) => !p.hidden).length, 1);
});

test("quiz runs a job and shows the score with its range and a tie verdict", async () => {
  const quiz = {
    workload: "coding", correct: 7, total: 10, score: 0.7, ci_low: 0.397, ci_high: 0.892,
    items: [
      { id: "code-1", ok: true, expected: "4", got: "4" },
      { id: "code-2", ok: false, expected: "hello", got: "<b>hi</b>" },
    ],
    seconds: 12, note: "Small quiz.",
  };
  const results = [
    { kind: "quiz", variant_id: "alpha-q8", workload: "coding", score: 0.8, ci_low: 0.49, ci_high: 0.94, timestamp: "2026-01-01" },
    { kind: "quiz", variant_id: "alpha-q4", workload: "coding", ...quiz, timestamp: "2026-01-02" },
    { kind: "quiz", variant_id: "beta-q4", workload: "general", score: 0.2, ci_low: 0.05, ci_high: 0.5, timestamp: "2026-01-02" },
  ];
  let posted;
  const t = setup(
    {
      "POST /api/quality/quiz": (body) => {
        posted = body;
        return job("q1", "queued");
      },
      "GET /api/quality/results": { results },
    },
    { q1: [job("q1", "running", { progress: { stage: "quiz", done: 3, total: 10, message: "Question 3 of 10" } }), job("q1", "done", { result: quiz })] },
  );
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const panel = tabPanel(t, "Quick quiz");
  assert.equal(t.$("select", panel).value, "c-mid", "defaults to the current candidate");
  assert.equal(t.$$("select", panel)[1].value, "coding", "defaults to the user's workload");
  t.$('input[type="checkbox"]', panel).click();
  t.byText("Start quiz", panel).click();
  await until(() => panel.textContent.includes("7 of 10 right"), "quiz result");
  assert.deepEqual(posted, { candidate_id: "c-mid", workload: "coding", include_needle: true });
  assert.deepEqual(t.jobEvents, ["queued", "done"]);
  const text = panel.textContent;
  assert.match(text, /7 of 10 right · 70%/);
  assert.match(text, /likely between 40% and 89%/);
  assert.match(text, /Too close to call against Alpha 8B · Q8_0/);
  assert.match(text, /Scores so far on these questions/);
  assert.ok(!t.$(".qp-history", panel).textContent.includes("Beta 3B"), "other workloads are not mixed in");
  assert.ok(Math.abs(parseFloat(t.$(".qp-range-band", panel).style.left) - 39.7) < 0.01);
  assert.ok(text.includes("<b>hi</b>"), "model output is shown as text");
  assert.equal(t.$("b", panel), null, "model output is never parsed as HTML");
  assert.equal(t.$(".qp-status", panel).getAttribute("aria-live"), "polite");
});

test("quiz calls a clear winner only when ranges do not overlap", async () => {
  const t = setup({
    "GET /api/quality/results": {
      results: [
        { kind: "quiz", variant_id: "alpha-q8", workload: "coding", score: 0.95, ci_low: 0.8, ci_high: 0.99 },
        { kind: "quiz", variant_id: "alpha-q2", workload: "coding", score: 0.3, ci_low: 0.1, ci_high: 0.6 },
      ],
    },
  });
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  await until(() => t.$(".qp-history").textContent.length > 0, "history");
  assert.match(t.$(".qp-history").textContent, /Alpha 8B · Q8_0 scored clearly higher/);
});

test("quiz shows plain demo and failure messages", async () => {
  const t = setup({
    "POST /api/quality/quiz": () => {
      const e = new Error("Demo mode can't run models. Start without --demo to test real files.");
      e.status = 409;
      throw e;
    },
  });
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const start = t.byText("Start quiz");
  start.click();
  await until(() => t.$(".qp-status").textContent.includes("Demo mode"), "409 message");
  assert.ok(t.$(".qp-status").classList.contains("qp-error"));
  assert.equal(start.disabled, false);

  const f = setup({ "POST /api/quality/quiz": job("q2", "queued") }, { q2: [job("q2", "failed", { error: "The model ran out of memory." })] });
  f.w.QualityPanel.mount(f.$("#q"), f.ctx);
  f.byText("Start quiz").click();
  await until(() => f.$(".qp-status").textContent.includes("out of memory"), "job error");
});

test("a running job can be stopped", async () => {
  let cancelled = false;
  const t = setup(
    {
      "POST /api/quality/quiz": job("q3", "running"),
      "POST /api/jobs/q3/cancel": () => {
        cancelled = true;
        return job("q3", "cancelled");
      },
    },
    { q3: [job("q3", "running")] },
  );
  t.ctx.api = ((inner) => (path, o) => (cancelled && path === "/api/jobs/q3" ? Promise.resolve(job("q3", "cancelled")) : inner(path, o)))(t.ctx.api);
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  t.byText("Start quiz").click();
  await until(() => t.byText("Stop") && !t.byText("Stop").hidden, "stop button");
  t.byText("Stop").click();
  await until(() => t.$(".qp-status").textContent.includes("Stopped before it finished"), "cancelled");
});

test("blind compare: validation, voting, no labels before reveal, then reveal and tallies", async () => {
  const blind = {
    items: [
      { prompt: "Write a haiku", outputs: [{ slot: "A", text: "Leaves fall" }, { slot: "B", text: "Snow melts" }, { slot: "C", text: "Sun sets" }] },
      { prompt: "Explain DNS", outputs: [{ slot: "A", text: "Phone book" }, { slot: "B", text: "Address list" }, { slot: "C", text: "Name lookup" }] },
    ],
  };
  const votes = [];
  let comparePost;
  const t = setup(
    {
      "POST /api/quality/compare": (body) => {
        comparePost = body;
        return job("cmp", "queued");
      },
      "GET /api/quality/compare/cmp-1": blind,
      // Tallies from vote are keyed by model and must never be shown before reveal.
      "POST /api/quality/vote": (body) => {
        votes.push(body);
        return { tallies: { "c-mid": 1 } };
      },
      "POST /api/quality/reveal": {
        mapping: [
          { A: "c-mid", B: "c-big", C: "c-other" },
          { A: "c-other", B: "c-mid", C: "c-big" },
        ],
        tallies: { "c-mid": 1, "c-big": 0, "c-other": 0 },
      },
    },
    { cmp: [job("cmp", "running", { progress: { stage: "generate", done: 1, total: 6, message: "Model 1 of 3 is answering" } }), job("cmp", "done", { result: { comparison_id: "cmp-1" } })] },
  );
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const panel = tabPanel(t, "Try my prompts");
  const run = () => t.byText("Run the models", panel).click();
  const problem = () => t.$(".qp-problem", panel).textContent;

  // Validation: model count, empty prompt, too long, max prompts.
  run();
  assert.match(problem(), /at least two models/);
  const boxes = t.$$('fieldset input[type="checkbox"]', panel);
  assert.equal(boxes.length, 4);
  assert.equal(boxes[1].checked, true, "current candidate pre-selected");
  boxes[0].click();
  boxes[3].click();
  assert.equal(boxes[2].disabled, true, "at most three models");
  run();
  assert.match(problem(), /at least one prompt/);
  const area = () => t.$$("textarea", panel);
  const type = (el, value) => {
    el.value = value;
    el.dispatchEvent(new t.w.Event("input", { bubbles: true }));
  };
  type(area()[0], "x".repeat(4001));
  assert.match(t.$(".qp-counter", panel).textContent, /4,001 \/ 4,000 characters/);
  assert.ok(t.$(".qp-counter", panel).classList.contains("qp-over"));
  run();
  assert.match(problem(), /Prompt 1 is too long/);
  type(area()[0], "Write a haiku");
  const add = t.byText("Add a prompt", panel);
  for (let i = 0; i < 6; i++) if (!add.disabled) add.click();
  assert.equal(area().length, 5, "at most five prompts");
  assert.equal(add.disabled, true);
  assert.equal(area()[0].getAttribute("maxlength"), "4000");
  t.$$('button[aria-label^="Remove prompt"]', panel).slice(2).forEach((b) => b.click());
  assert.equal(area().length, 2);
  type(area()[1], "  Explain DNS  ");
  run();
  assert.equal(problem(), "");
  await until(() => panel.textContent.includes("Prompt 1 of 2"), "blind view: " + panel.textContent.slice(-300));
  assert.deepEqual(comparePost, { candidate_ids: ["c-big", "c-mid", "c-other"], prompts: ["Write a haiku", "Explain DNS"] });

  // Nothing identifying: no model names, ids or variant ids in text or any attribute.
  const leaks = () => {
    const html = panel.outerHTML;
    return candidates.flatMap((c) => [c.id, c.name, c.variant_id, c.quant]).filter((s) => html.includes(s));
  };
  assert.deepEqual(leaks(), []);
  assert.equal(t.$("select", panel), null);
  const reveal = t.byText("Reveal which model wrote each answer", panel);
  assert.equal(reveal.disabled, true, "reveal waits for votes or skip");
  t.byText("A is best", panel).click();
  await until(() => panel.textContent.includes("Vote saved: answer A."), "vote recorded");
  assert.deepEqual(votes, [{ comparison_id: "cmp-1", item: 0, slot: "A" }]);
  assert.deepEqual(leaks(), [], "vote response is not shown");
  assert.equal(reveal.disabled, true);
  t.byText("Skip voting on the rest", panel).click();
  assert.equal(reveal.disabled, false);
  assert.deepEqual(leaks(), []);
  reveal.click();
  await until(() => panel.textContent.includes("Your votes"), "reveal");
  const who = t.$$(".qp-who", panel).map((x) => x.textContent);
  assert.deepEqual(who, [
    "Written by Alpha 8B · Q4_K_M", "Written by Alpha 8B · Q8_0", "Written by Beta 3B · Q4_K_M",
    "Written by Beta 3B · Q4_K_M", "Written by Alpha 8B · Q4_K_M", "Written by Alpha 8B · Q8_0",
  ]);
  assert.match(panel.textContent, /Alpha 8B · Q4_K_M: 1 vote/);
  assert.match(panel.textContent, /Too close to call: the top two are within one vote/);
  assert.equal(t.w.document.activeElement.textContent, "Your votes", "focus moves to the result");
  assert.equal(votes.length, 1, "skipped prompts send no vote");
  t.byText("Start a new comparison", panel).click();
  assert.ok(t.byText("Run the models", panel));
});

test("blind compare needs two options and reports job errors", async () => {
  const t = setup();
  t.ctx.candidates = [candidates[0]];
  t.ctx.candidate = candidates[0];
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  assert.match(tabPanel(t, "Try my prompts").textContent, /at least two models to compare/);

  const f = setup({
    "POST /api/quality/compare": () => {
      const e = new Error("Compare again first");
      e.status = 409;
      throw e;
    },
  });
  f.w.QualityPanel.mount(f.$("#q"), f.ctx);
  const panel = tabPanel(f, "Try my prompts");
  f.$$('fieldset input[type="checkbox"]', panel)[0].click();
  f.$("textarea", panel).value = "hi";
  f.byText("Run the models", panel).click();
  await until(() => panel.textContent.includes("Compare again first"), "409");
});

test("compression check defaults to the biggest file and shows plain results", async () => {
  let posted;
  const result = {
    reference: "alpha-q8",
    results: {
      "alpha-q4": { mean_kld: 0.0123, median_kld: 0.004, kld_99: 0.2, same_top_p: 95.9, ppl_base: 7.1, ppl: 7.3, mean_delta_p: -0.5, plain: "Picks a different top word about 4% of the time." },
      "alpha-q2": { mean_kld: 0.21, same_top_p: 82, ppl_base: 7.1, ppl: 9.9, mean_delta_p: null },
    },
    notes: ["The temporary file used 1.2 GiB and was deleted."],
  };
  const t = setup(
    {
      "GET /api/local-models": { files: [{ path: "/m/a.gguf", variant_id: "alpha-q2", size_bytes: 3.1e9, source: "models_dir" }], locations: [] },
      "POST /api/quality/quant-check": (body) => {
        posted = body;
        return job("k1", "running");
      },
    },
    { k1: [job("k1", "done", { result })] },
  );
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const panel = tabPanel(t, "Compression check");
  await until(() => panel.textContent.includes("on your disk"), "local files marked");
  const ref = t.$("select", panel);
  assert.equal(ref.value, "alpha-q8", "largest of the same model");
  const labels = t.$$("fieldset label", panel).map((l) => l.textContent);
  assert.equal(labels.length, 2, "only other files of the same model");
  assert.ok(labels.every((l) => !l.includes("Beta")));
  assert.equal(t.$$("fieldset input:checked", panel).length, 1, "current candidate pre-selected");
  assert.match(panel.textContent, /temporary file/);
  assert.match(panel.textContent, /several minutes per file/);
  t.$$("fieldset input", panel)[1].click();
  t.byText("Run compression check", panel).click();
  await until(() => panel.textContent.includes("Compared with"), "result");
  assert.deepEqual(posted, { reference_variant_id: "alpha-q8", variant_ids: ["alpha-q4", "alpha-q2"] });
  const text = panel.textContent;
  assert.match(text, /Picks a different top word about 4% of the time\./);
  assert.match(text, /Picks a different top word about 18% of the time\./, "percentage (as llama.cpp prints it) phrased");
  assert.match(text, /95\.9%/);
  assert.match(text, /0\.0123/);
  assert.match(text, /7\.10 → 7\.30/);
  assert.match(text, /-0\.50 points/, "mean Δp is already in percentage points");
  assert.match(text, /1\.2 GiB and was deleted/);
  assert.match(text, /-0\.50 points/, "mean delta-p is already in percentage points");
  assert.ok(t.$$("details summary", panel).some((s) => s.textContent === "Show the numbers"));

  // Changing the reference to another model resets the choices; none selected -> validation.
  ref.value = "beta-q4";
  ref.dispatchEvent(new t.w.Event("change"));
  assert.match(panel.textContent, /no other files of this model/);
  t.byText("Run compression check", panel).click();
  assert.match(t.$(".qp-problem", panel).textContent, /at least one file/);
});

test("compression check explains when there is nothing to compare", () => {
  const t = setup();
  t.ctx.candidates = [candidates[3]];
  t.ctx.candidate = candidates[3];
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  assert.match(tabPanel(t, "Compression check").textContent, /at least two files of the same model/);
});

test("local models list shows files and runs a scan job", async () => {
  let scanned = false;
  const files = [
    { path: "/home/u/.cache/huggingface/hub/blobs/abc/qwen-q4.gguf", size_bytes: 4.9e9, sha256: "abc", source: "hf_cache", variant_id: "alpha-q4", gguf: { name: "Alpha 8B", quant: "Q4_K_M", layers: 32 }, verified: true },
  ];
  const t = setup(
    {
      "GET /api/local-models": () => ({
        files: scanned ? [...files, { path: "C:\\Models\\mystery.gguf", size_bytes: 1e9, source: "custom", variant_id: null, gguf: null, verified: false }] : files,
        locations: [{ source: "ollama", path: "/home/u/.ollama/models", exists: false }],
      }),
      "POST /api/local-models/scan": () => {
        scanned = true;
        return job("s1", "queued");
      },
    },
    { s1: [job("s1", "running", { progress: { stage: "scan", done: 1, total: null, message: "Looking in LM Studio" } }), job("s1", "done", { result: [] })] },
  );
  t.w.LocalModels.mount(t.$("#l"), t.ctx);
  const l = t.$("#l");
  await until(() => l.textContent.includes("Alpha 8B"), "files");
  assert.match(l.textContent, /4\.6 GiB · Q4_K_M · 32 layers · found in the Hugging Face download folder/);
  assert.match(l.textContent, /Known model/);
  assert.match(l.textContent, /fingerprint matches/);
  assert.match(l.textContent, /Ollama: \/home\/u\/.ollama\/models \(folder not found\)/);
  t.byText("Scan my disk", l).click();
  await until(() => l.textContent.includes("Found 2 model files"), "scan");
  assert.match(l.textContent, /mystery\.gguf/);
  assert.match(l.textContent, /Not in the catalogue/);
  assert.ok(t.calls.some((c) => c.method === "POST" && c.path === "/api/local-models/scan"));
  assert.deepEqual(t.jobEvents, ["queued", "done"]);
});

test("local models shows an empty state and falls back to fetch with the session token", async () => {
  const t = setup();
  const seen = [];
  t.w.fetch = async (path, init) => {
    seen.push({ path, init });
    return { ok: true, json: async () => ({ files: [], locations: [] }) };
  };
  delete t.ctx.api;
  t.w.LocalModels.mount(t.$("#l"), t.ctx);
  await until(() => t.$("#l").textContent.includes("No model files found yet"), "empty");
  assert.equal(seen[0].init.headers["X-Session-Token"], "t0k");

  t.w.fetch = async () => ({ ok: false, status: 409, json: async () => ({ error: "Demo mode never scans your disk." }) });
  t.byText("Scan my disk").click();
  await until(() => t.$("#l .qp-status").textContent.includes("Demo mode never scans"), "demo 409");
});

test("static assets keep the strict CSP rules", () => {
  const js = fs.readFileSync(root + "quality.js", "utf8");
  const css = fs.readFileSync(root + "quality.css", "utf8");
  assert.ok(!/innerHTML|insertAdjacentHTML|document\.write|eval\(|new Function/.test(js));
  assert.ok(!/setAttribute\(\s*["']on/.test(js));
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(css, /prefers-color-scheme: dark/);
});

test("reveal waits for a vote still being saved, even after skipping the rest", async () => {
  let release;
  const gate = new Promise((resolve) => (release = resolve));
  const t = setup(
    {
      "POST /api/quality/compare": job("v", "done", { result: { comparison_id: "v1" } }),
      "GET /api/quality/compare/v1": { items: [0, 1].map((i) => ({ prompt: "p" + i, outputs: [{ slot: "A", text: "a" }, { slot: "B", text: "b" }] })) },
      "POST /api/quality/vote": async () => {
        await gate;
        return {};
      },
    },
  );
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const panel = tabPanel(t, "Try my prompts");
  t.$$('fieldset input[type="checkbox"]', panel)[0].click();
  t.$("textarea", panel).value = "hi";
  t.byText("Run the models", panel).click();
  await until(() => panel.textContent.includes("Prompt 1 of 2"), "blind view");
  t.byText("A is best", panel).click();
  t.byText("Skip voting on the rest", panel).click();
  const reveal = t.byText("Reveal which model wrote each answer", panel);
  assert.equal(reveal.disabled, true, "still saving the first vote");
  const pressed = t.$$('[aria-pressed="true"]', panel).map((b) => b.textContent);
  assert.deepEqual(pressed, ["No preference"], "only the skipped prompt is marked skipped");
  release();
  await until(() => !reveal.disabled, "vote saved");
  assert.deepEqual(t.$$('[aria-pressed="true"]', panel).map((b) => b.textContent), ["A is best", "No preference"]);
  assert.equal(t.$(".qp-answer-text", panel).getAttribute("tabindex"), "0", "long answers are keyboard-scrollable");
});

test("a failed poll is retried instead of abandoning the job", async () => {
  let n = 0;
  const t = setup({ "POST /api/local-models/scan": job("r1", "running") });
  t.ctx.api = ((inner) => async (path, o) => {
    if (path === "/api/jobs/r1") {
      n++;
      if (n === 1) throw new Error("Failed to fetch");
      return job("r1", "done", { result: [] });
    }
    return inner(path, o);
  })(t.ctx.api);
  t.w.LocalModels.mount(t.$("#l"), t.ctx);
  t.byText("Scan my disk").click();
  await until(() => t.$("#l .qp-status").textContent.includes("Scan finished"), "retry then done");
  assert.equal(n, 2);
});

test("local models replaces Loading with a message when the list fails", async () => {
  const t = setup({
    "GET /api/local-models": () => {
      throw new Error("Demo mode has no local files.");
    },
  });
  t.w.LocalModels.mount(t.$("#l"), t.ctx);
  await until(() => t.$("#l").textContent.includes("Couldn't load the list"), "error state");
  assert.ok(!t.$("#l").textContent.includes("Loading…"));
  assert.match(t.$("#l .qp-status").textContent, /Demo mode has no local files/);
});

// ---- fix-up round: shapes the server really sends ----

test("blind compare: pinned reveal shape uses friendly labels, 'No preference' sends tie, hostile text stays text", async () => {
  const evil = '<img src=x onerror="window.pwned=1">';
  const votes = [];
  const t = setup(
    {
      "POST /api/quality/compare": () => job("cmp", "queued"),
      "GET /api/quality/compare/cmp-1": {
        items: [
          { prompt: evil, outputs: [{ slot: "A", text: evil, ready: true }, { slot: "B", text: "Fine", ready: true }] },
          { prompt: "Two", outputs: [{ slot: "A", text: "x", ready: true }, { slot: "B", text: "y", ready: false }] },
        ],
      },
      "POST /api/quality/vote": (body) => {
        votes.push(body);
        // Older servers only take A/B/C: a tie is refused with 400.
        if (body.slot === "tie") throw Object.assign(new Error("slot must be A, B or C"), { status: 400 });
        return { item: body.item, vote: body.slot };
      },
      "POST /api/quality/reveal": {
        mapping: [{ A: "c-mid", B: "c-zzz" }, { A: "c-zzz", B: "c-mid" }],
        tallies: { "c-mid": 1, "c-zzz": 0 },
        labels: { "c-mid": "Alpha 8B · Q4_K_M", "c-zzz": "Gamma 1B · Q8_0" },
      },
    },
    { cmp: [job("cmp", "done", { result: { comparison_id: "cmp-1" } })] },
  );
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const panel = tabPanel(t, "Try my prompts");
  t.$$('fieldset input[type="checkbox"]', panel)[0].click();
  const area = t.$("textarea", panel);
  area.value = "Hello";
  area.dispatchEvent(new t.w.Event("input", { bubbles: true }));
  t.byText("Run the models", panel).click();
  await until(() => t.byText("A is best", panel), "blind view");
  assert.equal(t.w.pwned, undefined);
  assert.equal(panel.querySelector("img"), null, "model output is never parsed as HTML");
  assert.match(panel.textContent, /Still writing this answer/);
  const html = panel.outerHTML;
  for (const secret of ["c-mid", "c-zzz", "Gamma", "alpha-q4"]) assert.ok(!html.includes(secret), `no ${secret} before reveal`);
  t.$$("button", panel).filter((b) => b.textContent === "A is best")[0].click();
  await until(() => votes.length === 1, "first vote");
  t.$$("button", panel).filter((b) => b.textContent === "No preference")[1].click();
  await until(() => votes.length === 2, "tie vote");
  assert.deepEqual(votes[1], { comparison_id: "cmp-1", item: 1, slot: "tie" });
  const reveal = t.byText("Reveal which model wrote each answer", panel);
  await until(() => !reveal.disabled, "reveal enabled after a refused tie");
  reveal.click();
  await until(() => /Written by/.test(panel.textContent), "revealed");
  assert.match(panel.textContent, /Written by Gamma 1B · Q8_0/);
  assert.match(panel.textContent, /Gamma 1B · Q8_0: 0 votes/);
  assert.ok(!panel.textContent.includes("c-zzz"));
});

test("compression check maps quant labels to files, shows verdict badges, errors and percentages", async () => {
  const t = setup(
    {
      "POST /api/quality/quant-check": () => job("qc", "queued"),
    },
    {
      qc: [
        job("qc", "done", {
          result: {
            reference: "Q8_0",
            variant_ids: { Q4_K_M: "alpha-q4", Q2_K: "alpha-q2" },
            results: {
              Q4_K_M: { same_top_p: 0.9, mean_kld: 0.5, verdict: "severe", plain: null },
              Q2_K: { same_top_p: null, verdict: null, error: "<b>The file could not be loaded.</b>" },
            },
            notes: [],
          },
        }),
      ],
    },
  );
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  const panel = tabPanel(t, "Compression check");
  t.$$('input[type="checkbox"]', panel).forEach((b) => b.checked || b.click());
  t.byText("Run compression check", panel).click();
  await until(() => /Compared with/.test(panel.textContent), "result");
  const heads = t.$$("h5", panel).map((x) => x.textContent);
  assert.ok(heads.some((x) => /Alpha 8B · Q4_K_M · 4\.6 GiB/.test(x)), "full label from variant_ids");
  assert.match(panel.textContent, /Very large difference/);
  assert.match(panel.textContent, /0\.9%/, "same_top_p is a percentage, never rescaled");
  assert.match(panel.textContent, /Picks a different top word about 99% of the time/);
  assert.match(panel.textContent, /<b>The file could not be loaded\.<\/b>/, "errors shown as text");
  assert.equal(panel.querySelector("b"), null);
});

test("local models read filename and folder labels (the server sends no paths)", async () => {
  const t = setup({
    "GET /api/local-models": {
      files: [{ filename: "mystery.gguf", size_bytes: 2e9, source: "lmstudio", variant_id: null, gguf: {}, verified: false, match_note: "Likely the same file as the catalogue model, not yet verified." }],
      locations: [{ source: "lmstudio", label: "~/.lmstudio/models", exists: true }],
    },
  });
  t.w.LocalModels.mount(t.$("#l"), t.ctx);
  await until(() => t.$(".qp-file-name"), "list");
  assert.equal(t.$(".qp-file-name").textContent, "mystery.gguf");
  assert.match(t.$("#l").textContent, /Likely the same file/);
  assert.match(t.$("#l").textContent, /~\/\.lmstudio\/models/);
  assert.ok(!t.$("#l").textContent.includes("undefined"));
});

test("quiz history falls back to the saved model name", async () => {
  const t = setup({
    "GET /api/quality/results": {
      results: [
        { kind: "quiz", variant_id: "gone-1", name: "Old Model", quant: "Q5_K_M", workload: "coding", score: 0.9, ci_low: 0.7, ci_high: 0.97, timestamp: "2026-01-02" },
        { kind: "quiz", variant_id: "alpha-q4", workload: "coding", score: 0.5, ci_low: 0.3, ci_high: 0.7, timestamp: "2026-01-01" },
      ],
    },
  });
  t.w.QualityPanel.mount(t.$("#q"), t.ctx);
  await until(() => /Scores so far/.test(t.$("#q").textContent), "history");
  assert.match(t.$("#q").textContent, /Old Model · Q5_K_M/);
  assert.ok(!t.$("#q").textContent.includes("gone-1"));
});

test("a file the catalogue doesn't know is labelled as the person's own file, by its local id", async () => {
  const t = setup({
    "GET /api/local-models": {
      files: [
        { filename: "my-finetune.gguf", size_bytes: 2e9, source: "custom", variant_id: null, local_variant_id: "local-abc123", gguf: {}, verified: false },
        { filename: "unknown.gguf", size_bytes: 1e9, source: "custom", variant_id: null, gguf: {}, verified: false },
      ],
      locations: [],
    },
  });
  t.w.LocalModels.mount(t.$("#l"), t.ctx);
  await until(() => t.$$(".qp-file-name").length === 2, "list");
  const [mine, unknown] = t.$$(".qp-files li");
  assert.match(mine.textContent, /Your own file/);
  assert.match(mine.textContent, /you can still use it here/);
  assert.match(unknown.textContent, /Not in the catalogue/);
});
