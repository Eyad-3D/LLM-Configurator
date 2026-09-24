/* Background tasks: polls /api/jobs/<id> and shows a small tray of running work.
   Progress text reports only what the server measured; unknown totals stay unknown. */
(() => {
  "use strict";
  const FINAL = new Set(["done", "failed", "cancelled"]);
  const config = { interval: 700, hiddenInterval: 4000, listInterval: 2000 };
  const known = new Map(); // id -> latest job record
  const watchers = new Map(); // id -> Set(onUpdate)
  let listTimer = null;
  const call = (path, body) => window.api(path, body);
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function bytes(n) {
    if (n == null || !Number.isFinite(Number(n))) return "?";
    n = Number(n);
    if (n >= 1073741824) return (n / 1073741824).toFixed(1) + " GiB";
    if (n >= 1048576) return Math.round(n / 1048576) + " MiB";
    return Math.max(0, Math.round(n / 1024)) + " KiB";
  }
  function duration(seconds) {
    if (seconds == null || !Number.isFinite(Number(seconds))) return null;
    const s = Math.max(0, Math.round(Number(seconds)));
    if (s < 60) return `${s} s`;
    if (s < 3600) return `${Math.round(s / 60)} min`;
    return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
  }
  const isBytes = (job, p) =>
    p.bytes_per_second != null ||
    /download|install|runtime/.test(job.kind || "") ||
    p.unit === "bytes";

  /** {fraction: 0..1 | null, text} — a plain summary of a job's progress. */
  function describe(job) {
    if (!job) return { fraction: null, text: "" };
    if (job.state === "queued")
      return { fraction: null, text: "Waiting for another task to finish…" };
    if (job.state === "done") return { fraction: 1, text: "Finished" };
    if (job.state === "cancelled") return { fraction: null, text: "Stopped" };
    if (job.state === "failed")
      return { fraction: null, text: job.error || "Something went wrong" };
    const p = job.progress || {};
    const total = Number(p.total);
    const done = Number(p.done);
    const fraction =
      total > 0 && Number.isFinite(done)
        ? Math.max(0, Math.min(1, done / total))
        : null;
    const parts = [];
    if (isBytes(job, p) && Number.isFinite(done)) {
      parts.push(
        total > 0 ? `${bytes(done)} of ${bytes(total)}` : `${bytes(done)}`,
      );
      if (p.bytes_per_second > 0) parts.push(`${bytes(p.bytes_per_second)}/s`);
      const eta = duration(p.eta_seconds);
      if (eta) parts.push(`about ${eta} left`);
      if (p.message) parts.unshift(p.message);
    } else {
      if (p.message) parts.push(p.message);
      else if (p.stage) parts.push(p.stage.replace(/_/g, " "));
      if (total > 0 && Number.isFinite(done) && !isBytes(job, p))
        parts.push(`step ${Math.min(done, total)} of ${total}`);
    }
    return { fraction, text: parts.join(" · ") || "Working…" };
  }

  function remember(job) {
    if (!job?.id) return job;
    known.set(job.id, job);
    render();
    if (!FINAL.has(job.state)) scheduleList();
    return job;
  }

  /** Poll one job until it finishes. Returns stop(). */
  function watch(jobId, onUpdate) {
    let stopped = false;
    if (!watchers.has(jobId)) watchers.set(jobId, new Set());
    watchers.get(jobId).add(onUpdate);
    (async () => {
      let failures = 0;
      while (!stopped) {
        let job;
        try {
          job = await call(`/api/jobs/${encodeURIComponent(jobId)}`);
          failures = 0;
        } catch (error) {
          if (error.status === 404) {
            job = {
              ...(known.get(jobId) || {}),
              id: jobId,
              state: "failed",
              error:
                "This task is no longer tracked. The app may have restarted — please try again.",
            };
          } else if (++failures >= 6) {
            job = {
              ...(known.get(jobId) || {}),
              id: jobId,
              state: "failed",
              error: "Lost contact with the app. Is it still running?",
            };
          } else {
            await pause(config.interval * 2 ** failures);
            continue;
          }
        }
        if (stopped) break;
        remember(job);
        try {
          onUpdate(job);
        } catch (error) {
          console.error(error);
        }
        if (FINAL.has(job.state)) break;
        await pause(document.hidden ? config.hiddenInterval : config.interval);
      }
      watchers.get(jobId)?.delete(onUpdate);
    })();
    return () => {
      stopped = true;
    };
  }

  /** Current jobs from the server (also refreshes the tray). */
  async function list() {
    const result = await call("/api/jobs");
    const jobs = Array.isArray(result?.jobs) ? result.jobs : [];
    const ids = new Set(jobs.map((j) => j.id));
    // Forget finished jobs the server no longer reports.
    for (const [id, job] of known)
      if (!ids.has(id) && FINAL.has(job.state)) known.delete(id);
    jobs.forEach((job) => known.set(job.id, job));
    render();
    return jobs;
  }

  function scheduleList() {
    if (listTimer) return;
    listTimer = setTimeout(
      async () => {
        listTimer = null;
        try {
          await list();
        } catch {
          return; // Older servers have no jobs endpoint; the tray stays empty.
        }
        if ([...known.values()].some((j) => !FINAL.has(j.state)))
          scheduleList();
      },
      document.hidden ? config.hiddenInterval : config.listInterval,
    );
  }

  async function cancel(jobId) {
    const job = await call(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {});
    return remember(job);
  }

  // ---- Tray -------------------------------------------------------------
  const tray = document.getElementById("jobs-tray");
  const hideAfter = new Map(); // finished job id -> when to drop it from the tray
  let fadeTimer = null;
  function visibleJobs() {
    const now = Date.now();
    return [...known.values()].filter((job) => {
      if (!FINAL.has(job.state)) return true;
      if (!hideAfter.has(job.id)) hideAfter.set(job.id, now + 6000);
      return hideAfter.get(job.id) > now;
    });
  }
  function progressBar(fraction, label) {
    const bar = document.createElement("progress");
    bar.max = 1;
    if (fraction != null) bar.value = fraction;
    bar.setAttribute("aria-label", label);
    return bar;
  }
  function render() {
    if (!tray) return;
    const jobs = visibleJobs();
    const active = jobs.filter((j) => !FINAL.has(j.state));
    tray.hidden = jobs.length === 0;
    document.body.classList.toggle("has-jobs", !tray.hidden);
    const count = tray.querySelector("#jobs-count");
    if (count)
      count.textContent = active.length
        ? `${active.length} running`
        : "All done";
    const listEl = tray.querySelector("#jobs-list");
    if (!listEl) return;
    listEl.replaceChildren(
      ...jobs.map((job) => {
        const info = describe(job);
        const item = document.createElement("li");
        item.className = `job job-${job.state}`;
        item.dataset.job = job.id;
        const title = document.createElement("strong");
        title.textContent = job.title || job.kind || "Task";
        const text = document.createElement("span");
        text.className = "job-text";
        text.textContent = info.text;
        item.append(title);
        if (!FINAL.has(job.state))
          item.append(progressBar(info.fraction, `${title.textContent} progress`));
        item.append(text);
        if (!FINAL.has(job.state)) {
          const stop = document.createElement("button");
          stop.type = "button";
          stop.className = "secondary small";
          stop.textContent = "Cancel";
          stop.setAttribute("aria-label", `Cancel ${title.textContent}`);
          stop.addEventListener("click", async () => {
            stop.disabled = true;
            try {
              await cancel(job.id);
            } catch (error) {
              text.textContent = error.message;
              stop.disabled = false;
            }
          });
          item.append(stop);
        }
        return item;
      }),
    );
    clearTimeout(fadeTimer);
    if (jobs.some((j) => FINAL.has(j.state)))
      fadeTimer = setTimeout(render, 6100); // Finished rows leave the tray.
  }
  function expandTray(open) {
    tray.querySelector("#jobs-toggle").setAttribute("aria-expanded", String(open));
    tray.querySelector("#jobs-list").hidden = !open;
  }
  if (tray) {
    // Small screens start with the tray folded so it doesn't cover the page.
    expandTray(!window.matchMedia?.("(max-width: 600px)").matches);
    tray.querySelector("#jobs-toggle").addEventListener("click", (event) =>
      expandTray(event.currentTarget.getAttribute("aria-expanded") !== "true"),
    );
  }

  window.Jobs = {
    watch,
    list,
    cancel,
    track: remember,
    describe,
    format: { bytes, duration },
    config,
  };
  // Pick up work that was already running before this page loaded.
  if (tray)
    list().then(
      (jobs) => jobs.some((j) => !FINAL.has(j.state)) && scheduleList(),
      () => {},
    );
})();
