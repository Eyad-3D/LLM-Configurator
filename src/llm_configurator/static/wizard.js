"use strict";
let screen = "welcome";
let editReturn = null;
let hasResults = false;
let busy = false;
let generation = 0;
let loadingTimer = null;
const questionScreens = ["q1", "q2", "q3", "q4", "q5"];
const startScan = loadState().catch(() => null);
let modelPreparation = null;
let preparationResult = null;
let preparedWithRankings = false;
let rankingPreparation = null;
let rankingReturn = "q1";
let rankingBack = "welcome";
function rememberedRanking() {
  try {
    return localStorage.getItem("include-rankings");
  } catch {
    return null;
  }
}
function chooseRanking(value) {
  includeRankings = value;
  try {
    localStorage.setItem("include-rankings", String(value));
  } catch {}
  message("");
  void prepareModels();
  if (value) void prepareRankings();
  if (rankingReturn === "results") void findConfigurations(value);
  else showScreen(rankingReturn);
}
document.addEventListener("credentials-changed", () => {
  rankingPreparation = null;
  preparedWithRankings = false;
  generation++;
});
function prepareRankings() {
  if (!rankingPreparation)
    rankingPreparation = (async () => {
      const result = await prepareModels();
      if (preparedWithRankings || appState?.demo || !includeRankings)
        return result;
      return refreshMetadata(true, false);
    })().catch((error) => ({ warnings: [error.message] }));
  return rankingPreparation;
}
function prepareModels() {
  if (modelPreparation) return modelPreparation;
  $("preparation-status").hidden = false;
  $("preparation-status").classList.add("is-preparing");
  $("preparation-status").textContent = includeRankings
    ? "Preparing models and rankings while you answer…"
    : "Preparing model data while you answer…";
  preparedWithRankings = includeRankings;
  modelPreparation = (async () => {
    await startScan;
    if (!appState) await loadState();
    const result = appState.demo
      ? { warnings: [] }
      : await refreshMetadata(preparedWithRankings, true);
    preparationResult = result;
    $("preparation-status").classList.remove("is-preparing");
    $("preparation-status").textContent = result?.warnings?.length
      ? "Model data prepared with some unavailable sources. Details will appear with your results."
      : "Model data ready.";
    return result;
  })().catch((error) => {
    // Handle background failures immediately; surface them when results are requested.
    preparationResult = { warnings: [error.message] };
    $("preparation-status").classList.remove("is-preparing");
    $("preparation-status").textContent =
      "Model data could not be updated. Cached data will be used if available.";
    return preparationResult;
  });
  return modelPreparation;
}
function showScreen(name) {
  clearTimeout(loadingTimer);
  $("wait-details").hidden = true;
  $("results-busy").hidden = true;
  if (name === "loading")
    loadingTimer = setTimeout(() => {
      if (screen === "loading") $("wait-details").hidden = false;
    }, 10000);
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
    : "Show my recommendations →";
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
async function openRanking(target = screen) {
  rankingBack = screen;
  rankingReturn = ["welcome", "ranking"].includes(target) ? "q1" : target;
  $("ranking-details").hidden = true;
  $("include-ranking").hidden = false;
  $("ranking-back").textContent = rankingReturn === "q1" ? "Back" : "Cancel";
  showScreen("ranking");
  try {
    await credentialStatus();
  } catch {
    $("key-status").textContent =
      "Credential status unavailable. You can continue without rankings.";
  }
}
$("start").addEventListener("click", async () => {
  rankingReturn = "q1";
  const saved = rememberedRanking();
  if (saved === "false") return chooseRanking(false);
  if (saved === "true") {
    try {
      const state = await api("/api/credentials");
      if (state.configured && screen === "welcome") return chooseRanking(true);
    } catch {}
  }
  if (screen === "welcome") await openRanking("q1");
});
$("include-ranking").addEventListener("click", () => {
  $("ranking-details").hidden = false;
  $("include-ranking").hidden = true;
  $("aa-api-key").focus();
});
$("change-ranking").addEventListener("click", () => openRanking(screen));
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
  findConfigurations(includeRankings),
);
$("ranking-back").addEventListener("click", () => showScreen(rankingBack));
$("skip-ranking").addEventListener("click", () => chooseRanking(false));
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
    chooseRanking(true);
  } catch (error) {
    message(error.message, true);
  }
});
$("adjust").addEventListener("click", () => showScreen("review"));
$("ranking-settings").addEventListener("click", () => openRanking("results"));
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
    $("loading-text").textContent = "Finishing model preparation…";
    // Cached catalogue reads never wait for a remote refresh.
    await startScan;
    if (!appState) await loadState();
    const preparation = prepareModels();
    if (!appState.demo && !appState.status?.variants) await preparation;
    if (request !== generation) return;
    const warnings = [...(preparationResult?.warnings || [])];
    if (warnings.length) message(warnings.join(" · "));
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
    if (!appState.demo && rankings) {
      message(
        "Showing available results. Benchmark rankings are updating in the background.",
      );
      $("results-busy").hidden = false;
      void updateRankings(request, JSON.stringify(requirements()), preparation);
    }
  } catch (error) {
    if (request === generation) {
      showScreen("review");
      message(error.message + " Your answers have been kept.", true);
    }
  } finally {
    if (request === generation) busy = false;
  }
}
// Refresh scores independently and never overwrite a later comparison or edited answers.
async function updateRankings(request, answers, preparation) {
  try {
    await preparation;
    if (request !== generation || !includeRankings) return;
    const result = await prepareRankings();
    if (
      request !== generation ||
      screen !== "results" ||
      answers !== JSON.stringify(requirements())
    )
      return;
    const updated = await api("/api/recommend", JSON.parse(answers));
    if (
      request !== generation ||
      screen !== "results" ||
      answers !== JSON.stringify(requirements())
    )
      return;
    report = updated;
    renderResults();
    message(
      result?.warnings?.length
        ? result.warnings.join(" · ")
        : "Benchmark update complete.",
    );
  } catch (error) {
    if (request === generation && screen === "results")
      message(
        "Rankings could not be updated. Available results are still shown. " +
          error.message,
      );
  } finally {
    if (request === generation) $("results-busy").hidden = true;
  }
}
showScreen("welcome");
