"use strict";
let screen = "welcome";
let editReturn = null;
let hasResults = false;
let busy = false;
let generation = 0;
const questionScreens = ["q1", "q2", "q3", "q4", "q5"];
const startScan = loadState().catch(() => null);
function showScreen(name) {
  screen = name;
  document.querySelectorAll("[data-screen]").forEach((element) => {
    element.hidden = element.dataset.screen !== name;
  });
  const step = questionScreens.indexOf(name);
  $("journey").hidden = step < 0;
  $("question-nav").hidden = step < 0;
  $("step-label").textContent = `Question ${step + 1} of 5`;
  $("step-progress").value = step + 1;
  $("next").textContent = editReturn
    ? "Save answer →"
    : step === 4
      ? "Review answers →"
      : "Next →";
  if (name === "review") renderAnswers();
  const heading = document.querySelector(
    `[data-screen="${name}"] h1, [data-screen="${name}"] h2`,
  );
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus();
  }
  window.scrollTo(0, 0);
}
function answerRows() {
  const text = (id) => $(id).selectedOptions[0].textContent;
  return [
    ["Main use", text("workload")],
    ["Priority", text("priority") + ` · minimum ${$("min_tps").value} tok/s`],
    [
      "Context",
      Number($("context").value).toLocaleString() + " tokens per session",
    ],
    ["Concurrent sessions / agents", $("users").value],
    ["Resources", text("resource-mode")],
  ];
}
function renderAnswers() {
  $("answer-summary").innerHTML = answerRows()
    .map(
      ([label, value], index) =>
        `<div class="answer-row"><div><span>${esc(label)}</span><strong>${esc(value)}</strong></div><button class="text-button" type="button" data-edit="${index}">Edit ${esc(label.toLowerCase())}</button></div>`,
    )
    .join("");
  document.querySelectorAll("[data-edit]").forEach((button) =>
    button.addEventListener("click", () => {
      editReturn = "review";
      showScreen(questionScreens[Number(button.dataset.edit)]);
    }),
  );
  $("review-next").textContent = hasResults
    ? "Update my recommendations →"
    : "Continue →";
  $("review-back").textContent = hasResults ? "Back to results" : "Back";
}
function validQuestion() {
  for (const input of document.querySelectorAll(
    `[data-screen="${screen}"] input`,
  )) {
    if (
      !input.checkValidity() ||
      (input.type === "number" && input.value === "")
    ) {
      const details = input.closest("details");
      if (details) details.open = true;
      input.required = true;
      input.reportValidity();
      return false;
    }
  }
  return true;
}
async function openRanking() {
  showScreen("ranking");
  try {
    await credentialStatus();
  } catch {
    $("key-status").textContent =
      "Credential status unavailable. You can continue without rankings.";
  }
}
$("start").addEventListener("click", () => showScreen("q1"));
$("requirements").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!validQuestion()) return;
  if (editReturn) {
    const target = editReturn;
    editReturn = null;
    showScreen(target);
  } else {
    const index = questionScreens.indexOf(screen);
    showScreen(index === 4 ? "review" : questionScreens[index + 1]);
  }
});
$("back").addEventListener("click", () => {
  if (editReturn) {
    editReturn = null;
    showScreen("review");
    return;
  }
  const index = questionScreens.indexOf(screen);
  showScreen(index === 0 ? "welcome" : questionScreens[index - 1]);
});
$("context-preset").addEventListener("change", () => {
  $("context").value = $("context-preset").value;
});
$("resource-mode").addEventListener("change", () => {
  $("resource-choices").hidden = $("resource-mode").value !== "free";
});
$("review-back").addEventListener("click", () =>
  showScreen(hasResults ? "results" : "q5"),
);
$("review-next").addEventListener("click", () =>
  hasResults ? findConfigurations(includeRankings) : openRanking(),
);
$("ranking-back").addEventListener("click", () => showScreen("review"));
$("skip-ranking").addEventListener("click", () => findConfigurations(false));
$("with-ranking").addEventListener("click", async () => {
  try {
    if ($("aa-api-key").value) {
      message("Save your entered key before continuing.", true);
      return;
    }
    const state = await api("/api/credentials");
    if (!state.configured) {
      message("Add and save a key, or continue without rankings.", true);
      return;
    }
    await findConfigurations(true);
  } catch (error) {
    message(error.message, true);
  }
});
$("adjust").addEventListener("click", () => showScreen("review"));
$("ranking-settings").addEventListener("click", openRanking);
$("compare").addEventListener("click", () =>
  findConfigurations(includeRankings),
);
$("loading-back").addEventListener("click", () => {
  generation++;
  busy = false;
  showScreen("review");
});
async function findConfigurations(rankings) {
  if (busy) return;
  busy = true;
  const request = ++generation;
  includeRankings = rankings;
  message("");
  showScreen("loading");
  try {
    await startScan;
    if (!appState) await loadState();
    if (!appState.demo && (rankings || !appState.status?.variants)) {
      $("loading-text").textContent = rankings
        ? "Retrieving model and benchmark metadata. No model weights are downloaded."
        : "Retrieving model metadata. Benchmark rankings are skipped.";
      const result = await refreshMetadata(rankings);
      if (request !== generation) return;
      if (result.warnings?.length) message(result.warnings.join(" · "));
    }
    if (request !== generation) return;
    $("loading-text").textContent =
      "Comparing configurations against your current resources…";
    const nextReport = await api("/api/recommend", requirements());
    if (request !== generation) return;
    report = nextReport;
    hasResults = true;
    $("show_all").checked = false;
    renderResults();
    $("results-answer-summary").textContent = answerRows()
      .slice(0, 4)
      .map((row) => row[1])
      .join(" · ");
    showScreen("results");
  } catch (error) {
    if (request === generation) {
      showScreen("review");
      message(error.message + " Your answers have been kept.", true);
    }
  } finally {
    if (request === generation) busy = false;
  }
}
showScreen("welcome");
