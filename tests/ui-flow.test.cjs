const { test } = require("node:test");
const assert = require("node:assert/strict");
const { JSDOM } = require("jsdom");
const fs = require("node:fs");
const root = "src/llm_configurator/static/";
const tick = () => new Promise((resolve) => setTimeout(resolve, 30));
function setup({
  demo = true,
  cached = 1,
  processes = [],
  refreshGate = null,
  refreshFails = false,
} = {}) {
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
    processes,
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
    else if (path === "/api/refresh") {
      if (refreshFails) throw new Error("Offline");
      if (!body && refreshGate) await refreshGate;
      result = { running: false, result: { warnings: [] } };
    } else if (path === "/api/recommend") {
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
  vm.runInContext(
    fs.readFileSync(root + "selects.js", "utf8"),
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
  return { w, $, calls, visible, next, review, close: async () => { await tick(); w.close(); } };
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
  await s.close();
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
  await s.close();
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
  await s.close();
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
  await s.close();
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
  await s.close();
});

test("invalid numeric answer cannot advance", async () => {
  const s = setup();
  await tick();
  s.review();
  s.w.document.querySelector('[data-edit="3"]').click();
  s.$("users").value = "0";
  s.next();
  assert.deepEqual(s.visible(), ["q4"]);
  await s.close();
});

test("themed dropdowns preserve keyboard selection and Escape cancellation", async () => {
  const s = setup();
  await tick();
  s.$("start").click();
  const select = s.$("workload");
  const trigger = select.parentElement.querySelector("[role=combobox]");
  assert.equal(select.hidden, true);
  assert.equal(
    s.w.document.querySelector('label[for="' + trigger.id + '"]').textContent,
    "Primary use",
  );
  const key = (name) =>
    trigger.dispatchEvent(
      new s.w.KeyboardEvent("keydown", {
        key: name,
        bubbles: true,
        cancelable: true,
      }),
    );
  trigger.click();
  key("ArrowDown");
  key("Escape");
  assert.equal(select.value, "general");
  assert.equal(trigger.getAttribute("aria-expanded"), "false");
  trigger.click();
  key("ArrowDown");
  key("Enter");
  assert.equal(select.value, "coding");
  assert.match(trigger.textContent, /code/);
  await s.close();
});

test("dynamically loaded dropdown options use the theme and update the native value", async () => {
  const s = setup();
  await tick();
  const select = s.w.document.createElement("select");
  select.innerHTML =
    '<option value="a">Alpha</option><option value="b">Beta</option>';
  s.w.document.body.append(select);
  await tick();
  const trigger = select.parentElement.querySelector("[role=combobox]");
  assert.ok(trigger);
  let changes = 0;
  select.addEventListener("change", () => changes++);
  trigger.click();
  select.parentElement.querySelectorAll("[role=option]")[1].click();
  assert.equal(select.value, "b");
  assert.equal(changes, 1);
  select.innerHTML = '<option value="c">Gamma</option>';
  await tick();
  assert.match(trigger.textContent, /Gamma/);
  await s.close();
});

test("concurrency question explicitly includes sessions and agents", async () => {
  const s = setup();
  await tick();
  s.review();
  s.w.document.querySelector('[data-edit="3"]').click();
  assert.match(
    s.w.document.querySelector("[data-screen=q4] h1").textContent,
    /sessions or agents/,
  );
  s.$("users").value = "3";
  s.next();
  assert.match(
    s.$("answer-summary").textContent,
    /Concurrent sessions \/ agents/,
  );
  await s.close();
});

test("application list sorts by used memory and labels reclaim separately", async () => {
  const s = setup({
    processes: [
      {
        pid: 1,
        name: "Small",
        rss: 1073741824,
        reclaimable: 800000000,
        created: 1,
      },
      { pid: 2, name: "Large", rss: 3221225472, reclaimable: null, created: 2 },
    ],
  });
  await tick();
  const rows = s.w.document.querySelectorAll(".process-row");
  assert.match(rows[0].textContent, /Large/);
  assert.match(rows[0].textContent, /3.0 GiB used/);
  assert.match(rows[0].textContent, /Freeable memory unknown/);
  assert.match(rows[1].textContent, /Small/);
  await s.close();
});

test("model preparation runs during questions, is reused after edits, and skip makes no ranking request", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const s = setup({ demo: false, cached: 0, refreshGate: gate });
  await tick();
  s.$("start").click();
  await tick();
  assert.deepEqual(s.visible(), ["q1"]);
  assert.equal(
    s.calls.filter((c) => c.path === "/api/refresh" && c.body).length,
    1,
  );
  assert.deepEqual(
    s.calls.find((c) => c.path === "/api/refresh" && c.body).body,
    { include_scores: false, include_models: true },
  );
  s.next();
  assert.deepEqual(s.visible(), ["q2"]);
  release();
  await new Promise((resolve) => setTimeout(resolve, 1100));
  assert.match(s.$("preparation-status").textContent, /ready/);
  for (let i = 0; i < 4; i++) s.next();
  s.$("review-next").click();
  await tick();
  s.$("skip-ranking").click();
  await tick();
  assert.deepEqual(s.visible(), ["results"]);
  s.$("adjust").click();
  s.w.document.querySelector('[data-edit="3"]').click();
  s.$("users").value = "3";
  s.next();
  s.$("review-next").click();
  await tick();
  assert.equal(
    s.calls.filter((c) => c.path === "/api/refresh" && c.body).length,
    1,
  );
  assert.equal(
    s.calls.filter((c) => c.path === "/api/recommend").at(-1).body.users,
    3,
  );
  await s.close();
});

test("ranking waits for in-flight preparation and then requests scores only", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const s = setup({ demo: false, refreshGate: gate });
  await tick();
  s.review();
  s.$("review-next").click();
  await tick();
  s.$("with-ranking").click();
  await tick();
  assert.deepEqual(s.visible(), ["loading"]);
  assert.equal(
    s.calls.filter((c) => c.path === "/api/refresh" && c.body).length,
    1,
  );
  release();
  await new Promise((resolve) => setTimeout(resolve, 2200));
  assert.deepEqual(s.visible(), ["results"]);
  const jobs = s.calls.filter((c) => c.path === "/api/refresh" && c.body);
  assert.equal(jobs.length, 2);
  assert.deepEqual(jobs[1].body, {
    include_scores: true,
    include_models: false,
  });
  await s.close();
});

test("background failure keeps answers usable and allows cached recommendations", async () => {
  const s = setup({ demo: false, refreshFails: true });
  await tick();
  s.review();
  await tick();
  assert.match(s.$("preparation-status").textContent, /could not be updated/);
  s.$("review-next").click();
  await tick();
  s.$("skip-ranking").click();
  await tick();
  assert.deepEqual(s.visible(), ["results"]);
  assert.match(s.$("message").textContent, /Offline/);
  await s.close();
});
