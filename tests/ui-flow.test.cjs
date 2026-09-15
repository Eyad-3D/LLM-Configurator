const { test } = require("node:test");
const assert = require("node:assert/strict");
const { JSDOM } = require("jsdom");
const fs = require("node:fs");
const root = "src/llm_configurator/static/";
const tick = () => new Promise((resolve) => setTimeout(resolve, 30));
function setup({ demo = true, cached = 1 } = {}) {
  const dom = new JSDOM(fs.readFileSync(root + "index.html", "utf8"), {
    url: "http://127.0.0.1:8765",
    runScripts: "outside-only",
  });
  const w = dom.window;
  w.scrollTo = () => {};
  const calls = [];
  const hardware = {
    ram_available: 16e9,
    ram_total: 32e9,
    cpu_percent: 1,
    cores: 8,
    threads: 16,
    cpu: "Test CPU",
    gpus: [],
    disk_free: 100e9,
    processes: [],
  };
  w.fetch = async (path, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    calls.push({ path, body });
    let result;
    if (path === "/api/state")
      result = {
        hardware,
        demo,
        status: {
          variants: cached,
          timestamp: new Date().toISOString(),
          warnings: [],
        },
        definitions: [],
        scores: [],
      };
    else if (path === "/api/credentials")
      result = { configured: true, source: "saved" };
    else if (path === "/api/refresh")
      result = { running: false, result: { warnings: [] } };
    else if (path === "/api/recommend") {
      const candidates = Array.from({ length: 4 }, (_, i) => ({
        id: String(i),
        name: "Model " + i,
        quant: "Q4",
        context: body.context,
        mode: "cpu",
        gpu_layers: 0,
        total_layers: 32,
        scenario: "now",
        quality_score: null,
        quality_metric: "general",
        tps: null,
        ram_bytes: 2e9,
        vram_bytes: 0,
        ram_headroom_bytes: 4e9,
        explanation: "Estimated fit.",
        quality_comparison: { rank: null, reason: "missing" },
      }));
      result = {
        requirements: body,
        candidates,
        shortlist: ["0", "1", "2"],
        notes: [],
        rejected: {},
        quality_comparison: {
          rated_models: 0,
          eligible_models: 4,
          comparable: true,
        },
        hardware: { timestamp: new Date().toISOString() },
      };
    } else throw new Error("Unexpected endpoint " + path);
    return { ok: true, json: async () => result };
  };
  const vm = require("node:vm");
  vm.runInContext(
    fs.readFileSync(root + "app.js", "utf8"),
    dom.getInternalVMContext(),
  );
  vm.runInContext(
    fs.readFileSync(root + "wizard.js", "utf8"),
    dom.getInternalVMContext(),
  );
  const $ = (id) => w.document.getElementById(id);
  const visible = () =>
    [...w.document.querySelectorAll("[data-screen]")]
      .filter((x) => !x.hidden)
      .map((x) => x.dataset.screen);
  const next = () =>
    $("requirements").dispatchEvent(
      new w.Event("submit", { bubbles: true, cancelable: true }),
    );
  const review = () => {
    $("start").click();
    for (let i = 0; i < 5; i++) next();
  };
  return { w, $, calls, visible, next, review, close: () => w.close() };
}

test("welcome reveals neither questions nor key setup; one question at a time", async () => {
  const s = setup();
  await tick();
  assert.deepEqual(s.visible(), ["welcome"]);
  assert.equal(
    s.calls.some((c) => c.path === "/api/credentials"),
    false,
  );
  s.$("start").click();
  assert.deepEqual(s.visible(), ["q1"]);
  s.next();
  assert.deepEqual(s.visible(), ["q2"]);
  s.close();
});

test("review allows editing a single answer and retains other values", async () => {
  const s = setup();
  await tick();
  s.review();
  assert.deepEqual(s.visible(), ["review"]);
  s.w.document.querySelector('[data-edit="3"]').click();
  s.$("users").value = "3";
  s.next();
  assert.deepEqual(s.visible(), ["review"]);
  assert.equal(s.$("context").value, "8192");
  assert.match(s.$("answer-summary").textContent, /3/);
  s.$("review-next").click();
  await tick();
  assert.deepEqual(s.visible(), ["ranking"]);
  s.close();
});

test("skip rankings sends explicit opt-out and shows three cards; edit results preserves choice", async () => {
  const s = setup();
  await tick();
  s.review();
  s.$("review-next").click();
  await tick();
  s.$("skip-ranking").click();
  await tick();
  assert.deepEqual(s.visible(), ["results"]);
  assert.equal(
    s.calls.find((c) => c.path === "/api/recommend").body.include_rankings,
    false,
  );
  assert.equal(s.w.document.querySelectorAll(".card").length, 3);
  s.$("show_all").checked = true;
  s.$("show_all").dispatchEvent(new s.w.Event("change"));
  assert.equal(s.w.document.querySelectorAll(".card").length, 4);
  s.$("adjust").click();
  s.w.document.querySelector('[data-edit="2"]').click();
  s.$("context").value = "16384";
  s.next();
  s.$("review-next").click();
  await tick();
  assert.deepEqual(s.visible(), ["results"]);
  const latest = s.calls.filter((c) => c.path === "/api/recommend").at(-1).body;
  assert.equal(latest.context, 16384);
  assert.equal(latest.include_rankings, false);
  s.close();
});

test("first-run skip refreshes models without requesting benchmark scores", async () => {
  const s = setup({ demo: false, cached: 0 });
  await tick();
  s.review();
  s.$("review-next").click();
  await tick();
  s.$("skip-ranking").click();
  await new Promise((resolve) => setTimeout(resolve, 1150));
  assert.equal(
    s.calls.find((c) => c.path === "/api/refresh" && c.body).body
      .include_scores,
    false,
  );
  assert.deepEqual(s.visible(), ["results"]);
  s.close();
});

test("saved credential can be used after review without re-entry", async () => {
  const s = setup();
  await tick();
  s.review();
  s.$("review-next").click();
  await tick();
  s.$("with-ranking").click();
  await tick();
  assert.deepEqual(s.visible(), ["results"]);
  assert.equal(
    s.calls.find((c) => c.path === "/api/recommend").body.include_rankings,
    true,
  );
  s.close();
});

test("invalid numeric answer cannot advance", async () => {
  const s = setup();
  await tick();
  s.review();
  s.w.document.querySelector('[data-edit="3"]').click();
  s.$("users").value = "0";
  s.next();
  assert.deepEqual(s.visible(), ["q4"]);
  s.close();
});
