"use strict";
// Quality checks on the user's own computer: a quick quiz, a blind prompt
// comparison and a compression check, plus the list of model files already on
// disk. Honesty rules: scores always come with their range, close results are
// called a tie, and in the blind comparison the page never learns which model
// wrote which answer until the user asks to reveal it.
(function () {
  const POLL_MS = 700;
  const MAX_PROMPTS = 5;
  const MAX_CHARS = 4000;
  const FINAL = new Set(["done", "failed", "cancelled"]);
  const WORKLOADS = [
    ["general", "Everyday chat & reasoning"],
    ["coding", "Writing & understanding code"],
    ["agentic", "Agents & automation (tool-call format)"],
    ["documents", "Working with documents"],
  ];
  const SOURCES = {
    hf_cache: "the Hugging Face download folder",
    lmstudio: "LM Studio",
    ollama: "Ollama",
    models_dir: "this app's models folder",
    custom: "a folder you added",
  };
  const mounted = new WeakMap();
  let uid = 0;

  // Small DOM builder. Text always goes in through textContent, never HTML.
  function h(tag, props, ...kids) {
    const node = document.createElement(tag);
    let value;
    for (const [key, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (key === "class") node.className = v;
      else if (key === "text") node.textContent = v;
      else if (key === "value") value = v;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), v);
      else if (typeof v === "boolean") node[key] = v;
      else node.setAttribute(key, String(v));
    }
    for (const kid of kids.flat(Infinity))
      if (kid != null && kid !== false)
        node.append(kid instanceof Node ? kid : String(kid));
    if (value !== undefined) node.value = value;
    return node;
  }
  // replaceChildren does not flatten arrays; this does, and skips empty kids.
  const fill = (node, ...kids) => node.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false));
  const chars = (text) => [...String(text || "")].length;
  const gib = (bytes) =>
    bytes == null ? "size unknown" : `${(bytes / 1073741824).toFixed(1)} GiB`;
  const pct = (v) => (v == null ? null : v <= 1 ? v * 100 : v);
  const fmtPct = (v, digits = 0) =>
    v == null ? "unknown" : `${pct(v).toFixed(digits)}%`;
  const fmtPercent = (v, digits = 0) =>
    typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(digits)}%` : "unknown";
  const num = (v, digits = 3) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "unknown";
  const capital = (text) => String(text || "").replace(/^./, (c) => c.toUpperCase());
  const basename = (path) => String(path || "").split(/[\\/]/).pop();

  function defaultApi(path, options = {}) {
    const meta = document.querySelector('meta[name="session-token"]');
    const headers = { "X-Session-Token": meta ? meta.content : "" };
    const init = { method: options.method || "GET", headers };
    if (options.body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body =
        typeof options.body === "string"
          ? options.body
          : JSON.stringify(options.body);
    }
    return fetch(path, init).then(async (response) => {
      let payload = {};
      try {
        payload = await response.json();
      } catch (_) {
        payload = {};
      }
      if (!response.ok) {
        const error = new Error(
          payload.error || `The app answered with an error (${response.status}).`,
        );
        error.status = response.status;
        throw error;
      }
      return payload;
    });
  }

  function errorText(error) {
    const text = (error && error.message) || String(error || "");
    if (/failed to fetch|networkerror|load failed/i.test(text) || (error instanceof TypeError && !error.status))
      return "Couldn't reach the app. Check that LLM Configurator is still running, then try again.";
    return text || "Something went wrong. Please try again.";
  }

  // One session per mount: owns the timers so unmount stops all polling.
  function createSession(ctx) {
    const api = typeof ctx.api === "function" ? ctx.api : defaultApi;
    const pollMs = ctx.pollMs ?? POLL_MS;
    const timers = new Set();
    let alive = true;
    const session = {
      ctx,
      get alive() {
        return alive;
      },
      get: (path) => api(path),
      post: (path, body) => api(path, { method: "POST", body }),
      notify(job) {
        try {
          if (typeof ctx.onJob === "function") ctx.onJob(job);
        } catch (_) {
          /* the jobs tray is optional */
        }
      },
      watch(job, onUpdate) {
        return new Promise((resolve, reject) => {
          let misses = 0;
          const later = (fn, ms) => {
            const timer = setTimeout(() => {
              timers.delete(timer);
              if (alive) fn();
            }, ms);
            timers.add(timer);
          };
          // A single failed poll is not a failed job: retry a few times first.
          const poll = () =>
            api(`/api/jobs/${encodeURIComponent(job.id)}`).then(step, (error) =>
              ++misses >= 5 ? reject(error) : later(poll, pollMs * 2),
            );
          const step = (current) => {
            if (!alive) return;
            misses = 0;
            onUpdate(current);
            if (FINAL.has(current.state)) return resolve(current);
            later(poll, pollMs);
          };
          step(job);
        });
      },
      destroy() {
        alive = false;
        timers.forEach(clearTimeout);
        timers.clear();
      },
    };
    return session;
  }

  // A status line, progress bar and Stop button for one long job.
  function jobBox(session) {
    const text = h("p", { class: "qp-status", role: "status", "aria-live": "polite" });
    // Per-poll detail ("Question 3 of 10") is not live, so screen readers aren't flooded.
    const detail = h("p", { class: "hint qp-detail" });
    const bar = h("progress", { class: "qp-progress", max: "1", hidden: true, "aria-label": "Progress" });
    const stop = h("button", { type: "button", class: "secondary", hidden: true, text: "Stop" });
    const root = h("div", { class: "qp-job" }, text, detail, h("div", { class: "qp-job-row" }, bar, stop));
    let current = null;
    let stage = null;
    stop.addEventListener("click", async () => {
      if (!current) return;
      stop.disabled = true;
      text.textContent = "Stopping…";
      try {
        await session.post(`/api/jobs/${encodeURIComponent(current)}/cancel`, {});
      } catch (error) {
        text.textContent = errorText(error);
        stop.disabled = false;
      }
    });
    function update(job) {
      const p = job.progress || {};
      const total = Number(p.total);
      if (job.state === "queued") {
        text.textContent = "Waiting for another check to finish…";
        bar.hidden = false;
        bar.removeAttribute("value");
        return;
      }
      const key = `${job.state}|${p.stage || ""}`;
      if (key !== stage) {
        stage = key;
        text.textContent = p.stage ? `Working: ${String(p.stage).replace(/_/g, " ")}…` : "Working…";
      }
      detail.textContent = p.message || "";
      bar.hidden = false;
      if (total > 0) {
        bar.max = total;
        bar.value = Math.min(Number(p.done) || 0, total);
      } else bar.removeAttribute("value");
    }
    return {
      root,
      say(message, isError = false) {
        text.textContent = message;
        text.classList.toggle("qp-error", isError);
      },
      async run(start, labels) {
        text.classList.remove("qp-error");
        text.textContent = labels.starting;
        detail.textContent = "";
        stage = null;
        bar.hidden = true;
        const first = await start();
        const job = first && first.job ? first.job : first;
        if (!job || !job.id || !job.state) return first;
        current = job.id;
        session.notify(job);
        stop.hidden = false;
        stop.disabled = false;
        try {
          const final = await session.watch(job, update);
          session.notify(final);
          if (final.state === "done") {
            text.textContent = labels.done;
            return final.result;
          }
          if (final.state === "cancelled")
            throw new Error("Stopped before it finished. Nothing from this run was kept.");
          throw new Error(final.error || "The check failed without saying why. Try again.");
        } finally {
          current = null;
          if (stop.contains(document.activeElement)) text.focus?.();
          stop.hidden = true;
          bar.hidden = true;
          detail.textContent = "";
        }
      },
    };
  }

  function candidateLabel(c, all) {
    const mode = { cpu: "CPU", gpu: "GPU", split: "GPU + CPU" }[c.mode];
    let label = [c.name || c.variant_id || c.id, c.quant, mode]
      .filter(Boolean)
      .join(" · ");
    const twins = all.filter(
      (o) => o.name === c.name && o.quant === c.quant && o.mode === c.mode,
    );
    if (twins.length > 1) label += ` (${c.context ? c.context.toLocaleString() + " tokens" : "#" + (twins.indexOf(c) + 1)})`;
    return label;
  }
  function modelName(ctx, key, labels) {
    if (labels && typeof labels === "object" && typeof labels[key] === "string") return labels[key];
    const list = ctx.candidates || [];
    const c =
      list.find((x) => x.id === key) ||
      list.find((x) => x.variant_id === key) ||
      list.find((x) => x.name === key);
    return c ? [c.name, c.quant].filter(Boolean).join(" · ") : String(key);
  }
  function candidateOptions(ctx) {
    const list = ctx.candidates && ctx.candidates.length ? ctx.candidates : ctx.candidate ? [ctx.candidate] : [];
    return list.filter((c) => c && c.id != null);
  }

  // ---------------- Quick quiz ----------------
  function quizTab(session, ids) {
    const ctx = session.ctx;
    const list = candidateOptions(ctx);
    const panel = h("div");
    panel.append(
      h("p", {
        class: "qp-intro",
        text: "A short exam matched to your use. Answers are checked automatically. It's small, so treat close scores as a tie.",
      }),
    );
    if (!list.length) {
      panel.append(h("p", { class: "qp-empty", text: "Compare models first, then pick one here to quiz." }));
      return panel;
    }
    const model = h(
      "select",
      { id: ids("quiz-model"), value: (ctx.candidate || list[0]).id },
      list.map((c) => h("option", { value: c.id, text: candidateLabel(c, list) })),
    );
    const workload = h(
      "select",
      { id: ids("quiz-workload"), value: WORKLOADS.some(([k]) => k === ctx.workload) ? ctx.workload : "general" },
      WORKLOADS.map(([k, label]) => h("option", { value: k, text: label })),
    );
    const needle = h("input", { type: "checkbox", id: ids("quiz-needle") });
    const start = h("button", { type: "button", text: "Start quiz" });
    const job = jobBox(session);
    const result = h("div", { class: "qp-result" });
    const history = h("div", { class: "qp-history" });
    panel.append(
      h("div", { class: "qp-fields" },
        h("div", {}, h("label", { for: model.id, text: "Model to test" }), model),
        h("div", {}, h("label", { for: workload.id, text: "Type of questions" }), workload),
      ),
      h("label", { class: "check qp-check", for: needle.id }, needle,
        " Also test memory for long documents. We hide one fact in a long text and ask for it back."),
      h("div", { class: "qp-actions" }, start),
      job.root,
      result,
      history,
    );
    let past = [];
    async function loadHistory() {
      try {
        const data = await session.get("/api/quality/results");
        past = (data && data.results) || [];
      } catch (_) {
        past = [];
      }
      if (session.alive) renderHistory();
    }
    // Saved records carry the model's name; use it when the model isn't in today's list.
    function rowName(row) {
      const name = modelName(ctx, row.key);
      return name === String(row.key) && row.name ? row.name : name;
    }
    function quizRows() {
      const latest = new Map();
      for (const entry of past) {
        if (entry.kind !== "quiz") continue;
        const r = entry.score != null ? entry : entry.result || {};
        if ((r.workload || entry.workload) !== workload.value || r.score == null) continue;
        const key = entry.variant_id || entry.candidate_id || r.name;
        const prev = latest.get(key);
        if (!prev || String(entry.timestamp || "") >= String(prev.timestamp || ""))
          latest.set(key, { key, timestamp: entry.timestamp, r, name: [entry.name, entry.quant].filter(Boolean).join(" · ") });
      }
      return [...latest.values()].sort((a, b) => b.r.score - a.r.score);
    }
    function overlaps(a, b) {
      if (a.ci_low == null || a.ci_high == null || b.ci_low == null || b.ci_high == null) return true;
      return pct(a.ci_low) <= pct(b.ci_high) && pct(b.ci_low) <= pct(a.ci_high);
    }
    function rangeBar(r) {
      const lo = pct(r.ci_low ?? r.score), hi = pct(r.ci_high ?? r.score);
      const track = h("div", { class: "qp-range", "aria-hidden": "true" },
        h("span", { class: "qp-range-band" }),
        h("span", { class: "qp-range-dot" }));
      track.firstChild.style.left = `${lo}%`;
      track.firstChild.style.width = `${Math.max(hi - lo, 0.5)}%`;
      track.lastChild.style.left = `${pct(r.score)}%`;
      return track;
    }
    function rangeText(r) {
      return r.ci_low == null || r.ci_high == null
        ? "range unknown"
        : `likely between ${fmtPct(r.ci_low)} and ${fmtPct(r.ci_high)}`;
    }
    function renderHistory() {
      const rows = quizRows();
      fill(history);
      if (rows.length < 2) return;
      const [best, second] = rows;
      const verdict = overlaps(best.r, second.r)
        ? `Too close to call: ${rowName(best)} and ${rowName(second)} have overlapping ranges, so this quiz can't tell them apart.`
        : `${rowName(best)} scored clearly higher than the others on this quiz.`;
      history.append(
        h("h4", { text: "Scores so far on these questions" }),
        h("p", { class: "qp-verdict", text: verdict }),
        h("ul", { class: "qp-score-list" },
          rows.map((row) => h("li", {},
            h("span", { class: "qp-score-name", text: rowName(row) }),
            h("span", { text: `${fmtPct(row.r.score)} (${rangeText(row.r)})` }),
            row.r.ci_low == null || row.r.ci_high == null ? null : rangeBar(row.r)))),
      );
    }
    function renderResult(r, needleResult) {
      const others = quizRows().filter((row) => row.key !== r.variant_id && row.key !== r.candidate_id);
      const tie = others.find((row) => overlaps(row.r, r));
      const items = r.items || [];
      fill(result, 
        h("h4", { text: `${r.correct ?? "?"} of ${r.total ?? "?"} right · ${fmtPct(r.score)}` }),
        h("p", {
          text: r.ci_low == null
            ? "The likely range for this score is unknown."
            : `The true score is likely between ${fmtPct(r.ci_low)} and ${fmtPct(r.ci_high)}. The quiz is small, so the real score could be anywhere in that range.`,
        }),
        r.ci_low == null || r.ci_high == null ? null : rangeBar(r),
        tie ? h("p", { class: "qp-verdict", text: `Too close to call against ${modelName(ctx, tie.key)}: the ranges overlap.` }) : null,
        r.note ? h("p", { class: "hint", text: r.note }) : null,
        needleResult ? renderNeedle(needleResult) : null,
        items.length
          ? h("details", {},
              h("summary", { text: `See each question (${items.length})` }),
              h("ol", { class: "qp-items" },
                items.map((item) => h("li", { class: item.ok ? "qp-ok" : "qp-bad" },
                  h("strong", { text: item.ok ? "Right" : "Wrong" }),
                  ` · ${item.id ?? ""}`,
                  item.ok ? null : h("div", { class: "hint", text: `Expected: ${item.expected ?? "?"} · Got: ${String(item.got ?? "").slice(0, 300)}` })))))
          : null,
      );
    }
    function renderNeedle(n) {
      const rows = n.results || n.positions || [];
      const found = rows.filter((x) => x.ok || x.found).length;
      return h("div", { class: "qp-needle" },
        h("h4", { text: "Long-document recall" }),
        h("p", {
          text: rows.length
            ? `Found the hidden fact ${found} of ${rows.length} times${n.context_tokens ? ` in about ${Number(n.context_tokens).toLocaleString()} tokens (word pieces) of text` : ""}.`
            : n.note || "No recall result was returned.",
        }),
        rows.length
          ? h("ul", {}, rows.map((x) => h("li", {
              text: `${x.position != null ? `Fact placed ${Math.round(pct(x.position))}% of the way in` : "Fact"}: ${x.ok || x.found ? "found" : "missed"}`,
            })))
          : null);
    }
    start.addEventListener("click", async () => {
      start.disabled = true;
      fill(result);
      try {
        const body = { candidate_id: model.value, workload: workload.value };
        if (needle.checked) body.include_needle = true;
        const out = await job.run(() => session.post("/api/quality/quiz", body), {
          starting: "Starting the quiz…",
          done: "Quiz finished.",
        });
        if (!session.alive) return;
        const quiz = out && out.quiz ? out.quiz : out || {};
        const picked = list.find((c) => c.id === model.value);
        quiz.variant_id = quiz.variant_id || (picked && picked.variant_id);
        quiz.candidate_id = model.value;
        await loadHistory();
        renderResult(quiz, out && out.needle);
        job.say(`Quiz finished: ${quiz.correct ?? "?"} of ${quiz.total ?? "?"} right.`);
      } catch (error) {
        job.say(errorText(error), true);
      } finally {
        start.disabled = false;
      }
    });
    workload.addEventListener("change", renderHistory);
    loadHistory();
    return panel;
  }

  // ---------------- Try my prompts (blind compare) ----------------
  function compareTab(session, ids) {
    const ctx = session.ctx;
    const list = candidateOptions(ctx);
    const panel = h("div");
    const intro = h("p", {
      class: "qp-intro",
      text: "Type your own questions. Each model answers them, one after another. Then you judge the answers without knowing which model wrote which, like a blind taste test.",
    });
    function setup() {
      fill(panel, intro);
      if (list.length < 2) {
        panel.append(h("p", { class: "qp-empty", text: "You need at least two models to compare. Tick “Compare all configurations” on the results page, or change your answers." }));
        return;
      }
      const selected = new Set([(ctx.candidate || list[0]).id]);
      const boxes = list.map((c, i) => {
        const box = h("input", { type: "checkbox", id: ids(`cmp-c${i}`), checked: selected.has(c.id) });
        box.addEventListener("change", () => {
          box.checked ? selected.add(c.id) : selected.delete(c.id);
          syncBoxes();
        });
        return { c, box, row: h("label", { class: "check qp-check", for: box.id }, box, ` ${candidateLabel(c, list)}`) };
      });
      const limit = h("p", { class: "hint", "aria-live": "polite" });
      function syncBoxes() {
        for (const { box } of boxes) box.disabled = !box.checked && selected.size >= 3;
        limit.textContent = selected.size >= 3 ? "That's the maximum of three. Untick one to pick another." : "";
      }
      syncBoxes();
      const promptList = h("ol", { class: "qp-prompts" });
      const add = h("button", { type: "button", class: "secondary", text: "Add a prompt" });
      const prompts = [];
      function renumber() {
        prompts.forEach((p, i) => {
          p.label.textContent = `Prompt ${i + 1}`;
          p.remove.setAttribute("aria-label", `Remove prompt ${i + 1}`);
          p.remove.hidden = prompts.length === 1;
        });
        add.disabled = prompts.length >= MAX_PROMPTS;
      }
      function addPrompt(text = "") {
        const n = uid++;
        const area = h("textarea", { id: ids(`cmp-p${n}`), rows: "3", maxlength: String(MAX_CHARS), value: text, "aria-describedby": ids(`cmp-n${n}`) });
        const counter = h("span", { class: "qp-counter", id: ids(`cmp-n${n}`) });
        const count = () => {
          const used = chars(area.value);
          counter.textContent = `${used.toLocaleString()} / ${MAX_CHARS.toLocaleString()} characters`;
          counter.classList.toggle("qp-over", used > MAX_CHARS);
        };
        area.addEventListener("input", count);
        count();
        const entry = { area, label: h("label", { for: area.id }), remove: h("button", { type: "button", class: "text-button", text: "Remove" }) };
        entry.row = h("li", {}, h("div", { class: "qp-prompt-head" }, entry.label, entry.remove), area, counter);
        entry.remove.addEventListener("click", () => {
          prompts.splice(prompts.indexOf(entry), 1);
          entry.row.remove();
          renumber();
          prompts[0].area.focus();
        });
        prompts.push(entry);
        promptList.append(entry.row);
        renumber();
        return entry;
      }
      add.addEventListener("click", () => {
        if (prompts.length < MAX_PROMPTS) addPrompt().area.focus();
      });
      addPrompt();
      const problem = h("p", { class: "qp-problem", role: "alert" });
      const start = h("button", { type: "button", text: "Run the models" });
      const job = jobBox(session);
      panel.append(
        h("fieldset", {}, h("legend", { text: "Models to compare (pick 2 or 3)" }), boxes.map((b) => b.row), limit),
        h("fieldset", {}, h("legend", { text: `Your prompts (1 to ${MAX_PROMPTS})` }), promptList, add),
        h("p", { class: "hint", text: "Models answer one at a time to save memory, so this can take a few minutes." }),
        problem,
        h("div", { class: "qp-actions" }, start),
        job.root,
      );
      function validate() {
        const texts = prompts.map((p) => p.area.value.trim());
        if (selected.size < 2) return "Pick at least two models to compare.";
        if (selected.size > 3) return "Pick at most three models.";
        const filled = texts.filter(Boolean);
        if (!filled.length) return "Type at least one prompt.";
        if (filled.length > MAX_PROMPTS) return `Use at most ${MAX_PROMPTS} prompts.`;
        const long = texts.findIndex((t) => chars(t) > MAX_CHARS);
        if (long >= 0) return `Prompt ${long + 1} is too long. Keep each prompt under ${MAX_CHARS.toLocaleString()} characters.`;
        return null;
      }
      start.addEventListener("click", async () => {
        const issue = validate();
        problem.textContent = issue || "";
        if (issue) return;
        start.disabled = true;
        const body = {
          candidate_ids: list.filter((c) => selected.has(c.id)).map((c) => c.id),
          prompts: prompts.map((p) => p.area.value.trim()).filter(Boolean),
        };
        try {
          const out = await job.run(() => session.post("/api/quality/compare", body), {
            starting: "Starting the models…",
            done: "All answers are ready.",
          });
          if (!session.alive) return;
          const id = out && out.comparison_id;
          if (!id) throw new Error("The comparison finished but returned no answers. Try again.");
          const blind = await session.get(`/api/quality/compare/${encodeURIComponent(id)}`);
          if (session.alive) {
            vote(id, (blind && blind.items) || []);
            panel.querySelector("h4, .qp-empty")?.setAttribute("tabindex", "-1");
            panel.querySelector("h4, .qp-empty")?.focus();
          }
        } catch (error) {
          job.say(errorText(error), true);
          start.disabled = false;
        }
      });
    }
    // The blind view only ever holds slot letters and answer text.
    function vote(comparisonId, items) {
      const decided = new Array(items.length).fill(null);
      const pending = new Set();
      // Screen-reader announcements; it only becomes visible to show an error.
      const status = h("p", { class: "qp-status qp-sr-only", role: "status", "aria-live": "polite" });
      const tell = (text, isError = false) => {
        status.textContent = text;
        status.classList.toggle("qp-sr-only", !isError);
        status.classList.toggle("qp-error", isError);
      };
      const reveal = h("button", { type: "button", text: "Reveal which model wrote each answer", disabled: true });
      const skipRest = h("button", { type: "button", class: "secondary", text: "Skip voting on the rest" });
      const revealSlots = [];
      const itemControls = [];
      const views = items.map((item, index) => {
        const outputs = item.outputs || [];
        const answers = outputs.map((o) => {
          const slot = String(o.slot);
          const who = h("p", { class: "qp-who", hidden: true });
          revealSlots.push({ index, slot, who });
          return h("article", { class: "qp-answer" },
            h("h5", { text: `Answer ${slot}` }),
            h("div", { class: "qp-answer-text", tabindex: "0", role: "region", "aria-label": `Answer ${slot}, prompt ${index + 1}`, text: o.ready === false ? "Still writing this answer…" : o.text ?? "" }),
            who);
        });
        const choice = h("p", { class: "hint" });
        const buttons = outputs.map((o) => h("button", { type: "button", class: "secondary", "aria-pressed": "false", text: `${o.slot} is best` }));
        const none = h("button", { type: "button", class: "secondary", "aria-pressed": "false", text: "No preference" });
        const all = [...buttons, none];
        const pick = async (button, slot) => {
          all.forEach((b) => (b.disabled = true));
          pending.add(index);
          sync();
          try {
            if (slot) await session.post("/api/quality/vote", { comparison_id: comparisonId, item: index, slot });
            else {
              try {
                await session.post("/api/quality/vote", { comparison_id: comparisonId, item: index, slot: "tie" });
              } catch (error) {
                if (error.status !== 400) throw error; // older server: keep "no preference" on this page only
              }
            }
            if (!session.alive) return;
            button.setAttribute("aria-pressed", "true");
            decided[index] = slot || "skip";
            choice.textContent = slot ? `Vote saved: answer ${slot}.` : "Skipped: no preference.";
            tell(`Prompt ${index + 1}: ${choice.textContent}`);
            focusNext(index);
          } catch (error) {
            tell(errorText(error), true);
            all.forEach((b) => (b.disabled = false));
          } finally {
            pending.delete(index);
            sync();
          }
        };
        buttons.forEach((b, i) => b.addEventListener("click", () => pick(b, String(outputs[i].slot))));
        none.addEventListener("click", () => pick(none, null));
        itemControls.push({ all, none, choice, index });
        // Long answers scroll, so the box itself must be reachable by keyboard.
        return h("section", { class: "qp-item", "aria-labelledby": ids(`cmp-i${index}`) },
          h("h4", { id: ids(`cmp-i${index}`), text: `Prompt ${index + 1} of ${items.length}` }),
          h("blockquote", { class: "qp-prompt-text", text: item.prompt ?? "" }),
          h("div", { class: "qp-answers" }, answers),
          h("div", { class: "qp-vote", role: "group", "aria-label": `Which answer to prompt ${index + 1} is best?` }, all),
          choice);
      });
      function sync() {
        const open = decided.filter((d) => d == null).length;
        reveal.disabled = open > 0 || pending.size > 0;
        skipRest.hidden = open === 0;
      }
      // After a vote the pressed button is disabled, so move focus somewhere useful.
      function focusNext(index) {
        const next = itemControls.find((c) => c.index > index && decided[c.index] == null && !pending.has(c.index));
        (next ? next.all[0] : reveal.disabled ? skipRest : reveal).focus();
      }
      skipRest.addEventListener("click", () => {
        for (const c of itemControls) {
          if (decided[c.index] != null || pending.has(c.index)) continue;
          decided[c.index] = "skip";
          c.all.forEach((b) => (b.disabled = true));
          c.none.setAttribute("aria-pressed", "true");
          c.choice.textContent = "Skipped: no preference.";
        }
        tell(pending.size ? "Skipped the rest. Waiting for your last vote to save." : "Skipped the remaining prompts. You can reveal now.");
        sync();
        reveal.focus();
      });
      const summary = h("div", { class: "qp-result" });
      reveal.addEventListener("click", async () => {
        reveal.disabled = true;
        try {
          const out = await session.post("/api/quality/reveal", { comparison_id: comparisonId });
          if (!session.alive) return;
          for (const r of revealSlots) {
            const label = slotLabel(out, items, r.index, r.slot);
            r.who.textContent = label == null ? "Written by an unknown model" : `Written by ${modelName(ctx, label, friendly(out))}`;
            r.who.hidden = false;
          }
          renderTallies(out);
          summary.querySelector("h4").focus();
          reveal.hidden = true;
          skipRest.hidden = true;
          tell("Revealed. Each answer now shows which model wrote it.");
        } catch (error) {
          tell(errorText(error), true);
          reveal.disabled = false;
        }
      });
      function renderTallies(out) {
        const tallies = (out && (out.tallies || out.votes)) || {};
        const rows = Object.entries(tallies).filter(([, n]) => typeof n === "number").sort((a, b) => b[1] - a[1]);
        const again = h("button", { type: "button", class: "secondary", text: "Start a new comparison" });
        again.addEventListener("click", () => {
          setup();
          panel.querySelector("input, textarea")?.focus();
        });
        const lead = rows.length > 1 && rows[0][1] - rows[1][1] <= 1 && rows[0][1] > 0;
        fill(summary, 
          h("h4", { text: "Your votes", tabindex: "-1" }),
          rows.length
            ? h("ul", {}, rows.map(([label, n]) => h("li", { text: `${modelName(ctx, label, friendly(out))}: ${n} ${n === 1 ? "vote" : "votes"}` })))
            : h("p", { text: "No votes were recorded." }),
          lead ? h("p", { class: "qp-verdict", text: "Too close to call: the top two are within one vote of each other." }) : null,
          h("p", { class: "hint", text: "A handful of prompts is a small sample. Treat a one-vote lead as a tie." }),
          again,
        );
      }
      fill(panel, 
        h("p", { class: "qp-intro", text: "Answers are shuffled for each prompt. Pick the one you like best. Model names stay hidden until you reveal them." }),
        items.length ? views : h("p", { class: "qp-empty", text: "No answers came back. Try again with a different prompt." }),
        status,
        h("div", { class: "qp-actions" }, skipRest, reveal),
        summary,
      );
      sync();
    }
    setup();
    return panel;
  }
  // Pinned reveal shape: {mapping: [{A: candidate_id, …}], tallies, labels: {candidate_id: name}}.
  // `labels` is a name lookup only when a mapping is present too (older servers used it as the map).
  const friendly = (out) => (out && out.mapping && out.labels && !Array.isArray(out.labels) ? out.labels : null);
  // The reveal shape is "mapping plus tallies"; accept per-item or shared maps.
  function slotLabel(out, items, index, slot) {
    if (!out) return null;
    const map = out.mapping ?? out.labels ?? out.items;
    if (Array.isArray(map)) {
      const entry = map[index];
      if (!entry) return null;
      if (Array.isArray(entry.outputs)) {
        const o = entry.outputs.find((x) => String(x.slot) === slot);
        return o ? o.label ?? null : null;
      }
      const slots = entry.slots || entry.mapping || entry;
      return slots[slot] ?? null;
    }
    if (map && typeof map === "object") {
      if (map[index] && typeof map[index] === "object") return map[index][slot] ?? null;
      return map[slot] ?? null;
    }
    return null;
  }

  // ---------------- Compression check ----------------
  function compressionTab(session, ids) {
    const ctx = session.ctx;
    const panel = h("div");
    panel.append(h("p", {
      class: "qp-intro",
      text: "Smaller files are squeezed versions of the same model. This check counts how often the smaller file picks a different next word than a bigger one, like comparing a compressed photo to the original.",
    }));
    const seen = new Map();
    for (const c of candidateOptions(ctx)) {
      if (!c.variant_id || seen.has(c.variant_id)) continue;
      seen.set(c.variant_id, { id: c.variant_id, name: c.name || c.variant_id, quant: c.quant, bytes: c.file_bytes ?? null, model: c.repo || c.name || c.variant_id });
    }
    const variants = [...seen.values()];
    if (variants.length < 2) {
      panel.append(h("p", { class: "qp-empty", text: "This needs at least two files of the same model, for example Q8 and Q4. Tick “Compare all configurations” on the results page to see more." }));
      return panel;
    }
    const onDisk = new Set();
    const label = (v) => `${v.name}${v.quant ? " · " + v.quant : ""} · ${gib(v.bytes)}${onDisk.has(v.id) ? " · on your disk" : ""}`;
    const bySize = (a, b) => (b.bytes ?? -1) - (a.bytes ?? -1);
    const reference = h("select", { id: ids("qc-ref") });
    const choices = h("div", { class: "qp-choices" });
    const selected = new Set();
    function fillReference(keep) {
      const groups = new Map();
      for (const v of variants) {
        if (!groups.has(v.model)) groups.set(v.model, []);
        groups.get(v.model).push(v);
      }
      fill(reference, ...[...groups.entries()].map(([model, vs]) =>
        h("optgroup", { label: vs[0].name || model }, vs.sort(bySize).map((v) => h("option", { value: v.id, text: label(v) })))));
      reference.value = keep || defaultReference();
    }
    function defaultReference() {
      const mine = ctx.candidate && seen.get(ctx.candidate.variant_id);
      const group = variants.filter((v) => v.model === (mine ? mine.model : variants[0].model)).sort(bySize);
      const local = group.filter((v) => onDisk.has(v.id));
      return ((local.length ? local : group)[0] || variants[0]).id;
    }
    function fillChoices(reset) {
      const ref = seen.get(reference.value);
      const same = variants.filter((v) => v.model === ref.model && v.id !== ref.id).sort(bySize);
      if (reset) {
        selected.clear();
        const mine = ctx.candidate && same.find((v) => v.id === ctx.candidate.variant_id);
        if (mine || same[0]) selected.add((mine || same[0]).id);
      }
      for (const id of [...selected]) if (!same.some((v) => v.id === id)) selected.delete(id);
      fill(choices, same.length
        ? h("fieldset", {}, h("legend", { text: "Files to check (pick 1 to 3)" }), same.map((v, i) => {
            const box = h("input", { type: "checkbox", id: ids(`qc-c${i}`), checked: selected.has(v.id), disabled: !selected.has(v.id) && selected.size >= 3 });
            box.addEventListener("change", () => {
              box.checked ? selected.add(v.id) : selected.delete(v.id);
              fillChoices(false);
              choices.querySelector(`[id="${box.id}"]`)?.focus();
            });
            return h("label", { class: "check qp-check", for: box.id }, box, ` ${label(v)}`);
          }))
        : h("p", { class: "qp-empty", text: "There are no other files of this model to check. Pick a different reference." }));
    }
    fillReference();
    fillChoices(true);
    reference.addEventListener("change", () => fillChoices(true));
    const problem = h("p", { class: "qp-problem", role: "alert" });
    const start = h("button", { type: "button", text: "Run compression check" });
    const job = jobBox(session);
    const result = h("div", { class: "qp-result" });
    panel.append(
      h("label", { for: reference.id, text: "Compare against (the reference)" }),
      reference,
      h("p", { class: "hint", text: "Usually the biggest file you have of the same model. Bigger files lose less." }),
      choices,
      h("div", { class: "qp-warn" },
        h("strong", { text: "Before you start: " }),
        "every file must already be on your disk. Each file is loaded in turn, so this can take several minutes per file. It also needs free disk space for a temporary file, which is deleted afterwards."),
      problem,
      h("div", { class: "qp-actions" }, start),
      job.root,
      result,
    );
    session.get("/api/local-models").then((data) => {
      if (!session.alive) return;
      for (const f of (data && data.files) || []) {
        const id = f.variant_id || f.local_variant_id;
        if (id) onDisk.add(id);
      }
      if (!onDisk.size) return;
      // Re-rendering replaces the controls; put focus back where it was.
      const focused = panel.contains(document.activeElement) ? document.activeElement.id : null;
      fillReference(reference.value);
      fillChoices(false);
      if (focused) panel.querySelector(`[id="${focused}"]`)?.focus();
    }, () => {});
    start.addEventListener("click", async () => {
      const issue = selected.size < 1 ? "Pick at least one file to check." : selected.size > 3 ? "Pick at most three files." : null;
      problem.textContent = issue || "";
      if (issue) return;
      start.disabled = true;
      fill(result);
      try {
        const out = await job.run(() => session.post("/api/quality/quant-check", {
          reference_variant_id: reference.value,
          variant_ids: variants.filter((v) => selected.has(v.id)).map((v) => v.id),
        }), { starting: "Starting the compression check…", done: "Compression check finished." });
        if (session.alive) renderResult(out || {});
      } catch (error) {
        job.say(errorText(error), true);
      } finally {
        start.disabled = false;
      }
    });
    function sentence(r) {
      if (r.plain) return r.plain;
      if (r.same_top_p == null) return "No result: the check did not report how often the top word matched.";
      const diff = 100 - r.same_top_p;
      return `Picks a different top word about ${diff < 1 ? diff.toFixed(1) : Math.round(diff)}% of the time.`;
    }
    // Plain names for quantcheck's verdicts (how much the smaller file differs).
    const VERDICTS = {
      negligible: "No real difference",
      small: "Small difference",
      moderate: "Noticeable difference",
      large: "Large difference",
      severe: "Very large difference",
    };
    function renderResult(out) {
      const entries = Object.entries(out.results || {});
      // Result keys are quant labels (e.g. "Q4_K_M"); `variant_ids` maps them back to files.
      const idOf = (key) => (out.variant_ids && out.variant_ids[key]) || key;
      const nameFor = (key) => (seen.get(idOf(key)) ? label(seen.get(idOf(key))) : modelName(ctx, idOf(key)));
      const refName = seen.get(reference.value) ? label(seen.get(reference.value)) : seen.get(out.reference) ? label(seen.get(out.reference)) : typeof out.reference_quant === "string" ? out.reference_quant : typeof out.reference === "string" ? out.reference : "the reference";
      const row = (term, explain, value) => [h("dt", {}, term, h("span", { class: "hint", text: ` (${explain})` })), h("dd", { text: value })];
      fill(result, 
        h("h4", { text: `Compared with ${refName}` }),
        entries.length
          ? entries.map(([key, r]) => h("section", { class: "qp-item" },
              h("div", { class: "qp-file-head" },
                h("h5", { text: nameFor(key) }),
                VERDICTS[r.verdict] ? h("span", { class: `qp-badge qp-badge-${r.verdict}`, text: VERDICTS[r.verdict] }) : null),
              h("p", { text: sentence(r) }),
              r.error ? h("p", { class: "qp-error-text", text: String(r.error) }) : null,
              h("details", {},
                h("summary", { text: "Show the numbers" }),
                h("dl", { class: "qp-numbers" },
                  row("Same top word", "how often both files pick the same next word", fmtPercent(r.same_top_p, 1)),
                  row("Average difference", "KL divergence: how far apart the two files' word guesses are; 0 means identical", num(r.mean_kld, 4)),
                  row("Median difference", "the typical word", num(r.median_kld, 4)),
                  row("Worst 1% difference", "the same, for the hardest 1 in 100 words", num(r.kld_99, 4)),
                  row("Surprise score", "perplexity: how surprised the model is by real text; reference → this file, lower is better", `${num(r.ppl_base, 2)} → ${num(r.ppl, 2)}`),
                  row("Change in confidence", "average change in the top word's probability, in percentage points", r.mean_delta_p == null ? "unknown" : `${num(r.mean_delta_p, 2)} points`)))))
          : h("p", { text: "The check finished but returned no results." }),
        (out.notes || []).length ? h("ul", { class: "hint qp-notes" }, out.notes.map((n) => h("li", { text: n }))) : null,
      );
    }
    return panel;
  }

  // ---------------- Tabs ----------------
  const TABS = [
    ["quiz", "Quick quiz", quizTab],
    ["compare", "Try my prompts", compareTab],
    ["compress", "Compression check", compressionTab],
  ];
  function mountQuality(container, ctx) {
    const session = createSession(ctx || {});
    const base = `qp${uid++}`;
    const ids = (name) => `${base}-${name}`;
    const tablist = h("div", { class: "qp-tabs", role: "tablist", "aria-label": "Quality checks" });
    const tabs = [];
    const panels = [];
    TABS.forEach(([key, label, build], i) => {
      const tab = h("button", {
        type: "button", role: "tab", class: "qp-tab", id: ids(`tab-${key}`),
        "aria-controls": ids(`panel-${key}`), "aria-selected": i === 0 ? "true" : "false",
        tabindex: i === 0 ? "0" : "-1", text: label,
      });
      const panel = h("div", {
        role: "tabpanel", class: "qp-panel", id: ids(`panel-${key}`),
        "aria-labelledby": tab.id, tabindex: "0", hidden: i !== 0,
      }, build(session, ids));
      tab.addEventListener("click", () => select(i, false));
      tabs.push(tab);
      panels.push(panel);
      tablist.append(tab);
    });
    function select(index, focus) {
      tabs.forEach((tab, i) => {
        const on = i === index;
        tab.setAttribute("aria-selected", on ? "true" : "false");
        tab.tabIndex = on ? 0 : -1;
        panels[i].hidden = !on;
      });
      if (focus) tabs[index].focus();
    }
    tablist.addEventListener("keydown", (event) => {
      const current = tabs.indexOf(document.activeElement);
      if (current < 0) return;
      const last = tabs.length - 1;
      const next = { ArrowRight: current === last ? 0 : current + 1, ArrowLeft: current === 0 ? last : current - 1, Home: 0, End: last }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      select(next, true);
    });
    const root = h("div", { class: "qp" }, tablist, panels);
    return attach(container, root, session, { select: (key) => select(Math.max(0, TABS.findIndex(([k]) => k === key)), false) });
  }

  // ---------------- Local models ----------------
  function mountLocal(container, ctx) {
    const session = createSession(ctx || {});
    const list = h("div", { class: "qp-local-list" });
    const where = h("div");
    const scan = h("button", { type: "button", text: "Scan my disk" });
    const job = jobBox(session);
    const root = h("div", { class: "qp qp-local" },
      h("p", { class: "qp-intro", text: "Model files already on this computer, so you don't download them twice. We look in the usual folders (Hugging Face, LM Studio, Ollama and this app's models folder). Nothing leaves your computer." }),
      h("div", { class: "qp-actions" }, scan),
      job.root, list, where);
    function render(data) {
      const files = (data && data.files) || [];
      fill(list, files.length
        ? h("ul", { class: "qp-files" }, files.map((f) => {
            const g = f.gguf || {};
            const facts = [gib(f.size_bytes), g.quant, g.layers ? `${g.layers} layers` : null, `found in ${SOURCES[f.source] || f.source || "an unknown folder"}`].filter(Boolean);
            return h("li", {},
              h("div", { class: "qp-file-head" },
                h("strong", { class: "qp-file-name", text: g.name || f.filename || basename(f.path) || "Unnamed file" }),
                h("span", {
                class: f.variant_id ? "tag" : "tag unknown",
                // A file the catalogue doesn't know still gets its own id, so it can be tested and compared.
                text: f.variant_id ? "Known model" : f.local_variant_id ? "Your own file" : "Not in the catalogue",
              })),
              h("div", { class: "hint", text: facts.join(" · ") }),
              f.filename || f.path ? h("div", { class: "hint", text: f.filename || basename(f.path) }) : null,
              h("div", { class: "hint", text: f.match_note || (f.verified ? "Checked: the file's fingerprint matches." : f.variant_id ? "Matched by name and size (not fingerprint-checked yet)." : f.local_variant_id ? "Not in our list of models, but you can still use it here." : "Not checked.") }));
          }))
        : h("p", { class: "qp-empty", text: "No model files found yet. Click Scan my disk to look." }));
      const locations = (data && data.locations) || [];
      fill(where, locations.length
        ? h("details", {}, h("summary", { text: "Where we looked" }),
            h("ul", { class: "hint" }, locations.map((l) => h("li", { text: `${capital(SOURCES[l.source] || l.source)}: ${l.label ?? l.path ?? "unknown folder"} ${l.exists ? "" : "(folder not found)"}`.trim() }))))
        : null);
    }
    async function load() {
      try {
        const data = await session.get("/api/local-models");
        if (session.alive) render(data);
      } catch (error) {
        if (!session.alive) return;
        fill(list, h("p", { class: "qp-empty", text: "Couldn't load the list of model files. Try Scan my disk." }));
        job.say(errorText(error), true);
      }
    }
    scan.addEventListener("click", async () => {
      scan.disabled = true;
      try {
        await job.run(() => session.post("/api/local-models/scan", {}), { starting: "Starting the scan…", done: "Scan finished." });
        if (!session.alive) return;
        await load();
        const count = list.querySelectorAll("li").length;
        job.say(`Scan finished. Found ${count} model ${count === 1 ? "file" : "files"}.`);
      } catch (error) {
        job.say(errorText(error), true);
      } finally {
        scan.disabled = false;
      }
    });
    list.append(h("p", { class: "hint", text: "Loading…" }));
    load();
    return attach(container, root, session, { refresh: load });
  }

  function attach(container, root, session, extra) {
    const previous = mounted.get(container);
    if (previous) previous.unmount();
    fill(container, root);
    const handle = Object.assign({
      unmount() {
        session.destroy();
        if (mounted.get(container) === handle) mounted.delete(container);
        if (root.parentNode === container) root.remove();
      },
    }, extra);
    mounted.set(container, handle);
    return handle;
  }

  window.QualityPanel = { mount: mountQuality };
  window.LocalModels = { mount: mountLocal };
})();
