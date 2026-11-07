"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  config: null,
  dataset: null,
  stats: null,
  selected: new Set(),
  modelParams: new Map(),
  currentProject: null,
  pollTimer: null,
  toastTimer: null,
  initialized: false,
  csrfToken: "",
  manualTarget: false,
  detectTimer: null,
  uploadBusy: false,
  followLog: true,
  lastLog: "",
  fullRunTouched: false,
  variants: [],
  queue: [],
  batchActive: false,
  serverActive: false,
  serverBusy: false,
  stale: false,
  comparisonKey: "",
  projectTimer: null,
  lastStatus: null,
  failureKey: "",
  failure: null,
  logLevel: "",
  logQuery: "",
  queueFilter: "all",
  llm: {
    plan: null,
    busy: false,
    message: "",
  },
  kaggle: {
    auth: null,
    hasSubmission: false,
    runId: null,
    submission: null,
    environmentOverride: false,
    busy: false,
    loading: false,
    message: "",
  },
};

const STATUS_LABELS = {
  idle: "idle",
  starting: "starting",
  running: "running",
  stopping: "stopping",
  completed: "completed",
  completed_with_warnings: "completed with warnings",
  partial: "partial",
  failed: "failed",
  stopped: "stopped",
  queued: "queued",
  cancelled: "cancelled",
  stale: "stale",
  unknown: "unknown",
};

const STAGE_LABELS = {
  pending: "pending",
  queued: "queued",
  running: "running",
  done: "done",
  failed: "failed",
  skipped: "skipped",
};

function api(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const needsToken = (method !== "GET" && method !== "HEAD") || String(path).startsWith("/api/upload");
  const headers = new Headers(options.headers || {});
  if (needsToken) {
    if (!state.csrfToken) {
      return Promise.reject(new Error("CSRF-токен не загружен. Обновите страницу."));
    }
    headers.set("X-Kconv-Token", state.csrfToken);
  }
  return fetch(path, { ...options, headers }).then(async (response) => {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      const error = new Error(payload && payload.message ? payload.message : response.statusText);
      error.hint = payload && payload.hint ? payload.hint : null;
      error.status = response.status;
      throw error;
    }
    return payload;
  });
}

function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function toast(message, bad = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("bad", Boolean(bad));
  node.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.remove("show"), 4200);
}

function fmtBytes(value) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${n} B`;
}

function fmtScore(value) {
  if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1000) return n.toLocaleString("ru-RU", { maximumFractionDigits: 2 });
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(3);
  const fixed = n.toFixed(6);
  return fixed.replace(/0+$/, "").replace(/\.$/, "");
}

function fmtDuration(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
  const n = Number(value);
  if (n < 60) return `${n.toFixed(n < 10 ? 1 : 0)}s`;
  return `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function setSelectOptions(select, values, autoLabel = null, keepValue = true) {
  const current = keepValue ? select.value : "";
  select.replaceChildren();
  if (autoLabel !== null) {
    const option = el("option", "", autoLabel);
    option.value = "";
    select.append(option);
  }
  for (const value of values) {
    const option = el("option", "", value);
    option.value = value;
    select.append(option);
  }
  if (current && Array.from(select.options).some((option) => option.value === current)) {
    select.value = current;
  } else if (select.options.length) {
    select.selectedIndex = 0;
  }
}

function setStatusBadge(node, status) {
  const normalized = String(status || "idle").toLowerCase();
  node.textContent = STATUS_LABELS[normalized] || normalized;
  node.className = `run-badge ${normalized}`;
}

function projectName() {
  return ($("#project").value || "my-project").trim();
}

function currentTask() {
  return $("#task").value;
}

function modelsForTask(task, search = "") {
  if (!state.config) return [];
  const needle = search.trim().toLowerCase();
  return state.config.models.filter((model) => {
    if (task !== "auto" && !model.tasks.includes(task)) return false;
    if (!needle) return true;
    const haystack = `${model.id} ${model.name || ""} ${model.source || ""} ${model.family || ""}`.toLowerCase();
    return haystack.includes(needle);
  });
}

function setConfigError(message = "") {
  const node = $("#config-error");
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("hidden", !message);
}

async function loadSession() {
  const data = await api("/api/session");
  if (!data || typeof data.token !== "string" || !data.token) {
    throw new Error("Сервер не вернул CSRF-токен.");
  }
  state.csrfToken = data.token;
}

async function restoreLlmPlan() {
  if (state.llm.plan) return;
  try {
    const data = await api("/api/session");
    if (state.llm.plan || !data || !data.llm_plan) return;
    const plan = data.llm_plan;
    if (plan.target !== $("#target").value) return;
    if ((plan.id_col || "") !== ($("#id-col").value || "")) return;
    state.llm.plan = plan;
    state.llm.message = "";
    const approved = $("#llm-approved");
    if (approved) approved.checked = false;
    renderLlmState();
  } catch (_error) {
    // Restoring a previous plan is optional; keep the normal empty state.
  }
}

async function loadConfig() {
  state.config = await api("/api/config");
  state.config.models = state.config.models || [];
  state.config.recipes = state.config.recipes || [];
  state.config.pipeline_presets = state.config.pipeline_presets || [];
  state.config.metrics = state.config.metrics || [];
  state.config.profiles = state.config.profiles || [];
  setConfigError(
    state.config.recipe_error
      ? `Recipe-каталог не загрузился: ${state.config.recipe_error} Метрики и профили доступны; исправьте recipe и перезапустите сервер.`
      : "",
  );
  $("#hero-models").textContent = String(state.config.models.length);
  $("#hero-recipes").textContent = String(state.config.recipes.length);

  const recipe = $("#recipe");
  recipe.replaceChildren();
  const recipeGroups = new Map([
    ["pipeline", { label: "pipeline — executable blocks", items: [] }],
    ["preset", { label: "preset — defaults only", items: [] }],
  ]);
  for (const item of state.config.recipes) {
    const kind = item.kind === "pipeline" ? "pipeline" : "preset";
    recipeGroups.get(kind).items.push(item);
  }
  for (const group of recipeGroups.values()) {
    if (!group.items.length) continue;
    group.items.sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const optgroup = document.createElement("optgroup");
    optgroup.label = group.label;
    for (const item of group.items) {
      const option = el("option", "", `${item.id} — ${item.title || item.name || item.id}`);
      option.value = item.id;
      option.title = item.best_for || item.attribution || "";
      optgroup.append(option);
    }
    recipe.append(optgroup);
  }
  if (!state.config.recipes.length) {
    const unavailable = el("option", "", "recipe-каталог недоступен — см. предупреждение");
    unavailable.value = "";
    recipe.append(unavailable);
  }
  recipe.disabled = !state.config.recipes.length;
  populateRecipeBatch();
  const baseline = state.config.recipes.find((item) => item.id === "pipeline/pesnik-clean")
    || state.config.recipes.find((item) => /baseline/i.test(item.id));
  if (baseline) recipe.value = baseline.id;
  renderRecipeCard();
  applyRecipeModels({ announce: false });

  const metric = $("#metric");
  const previousMetric = metric.value;
  metric.replaceChildren();
  const autoMetric = el("option", "", "auto");
  autoMetric.value = "auto";
  metric.append(autoMetric);
  for (const item of state.config.metrics) {
    const option = el("option", "", item.name);
    option.value = item.name;
    option.dataset.direction = item.direction || "";
    metric.append(option);
  }
  if (Array.from(metric.options).some((option) => option.value === previousMetric)) metric.value = previousMetric;

  const profile = $("#profile");
  const previousProfile = profile.value;
  profile.replaceChildren();
  const autoProfile = el("option", "", "auto / без профиля");
  autoProfile.value = "";
  profile.append(autoProfile);
  for (const item of state.config.profiles) {
    const level = item.level ?? item.id ?? item.index;
    const option = el("option", "", `${level} — ${item.name || item.title || "profile"}`);
    option.value = String(level);
    option.title = item.description || item.notes || "";
    profile.append(option);
  }
  if (Array.from(profile.options).some((option) => option.value === previousProfile)) profile.value = previousProfile;

  refreshModels();
  updateProfileState();
  syncLlmProvider();
  renderLlmState();
}

function populateRecipeBatch() {
  const select = $("#recipe-batch");
  if (!select) return;
  const pipelines = (state.config && state.config.recipes ? state.config.recipes : [])
    .filter((item) => item.kind === "pipeline")
    .sort((a, b) => String(a.id).localeCompare(String(b.id)));
  const groups = new Map();
  for (const item of pipelines) {
    const key = item.group || "other";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  const groupOrder = ["baseline", "engineering", "ensemble", "architecture"];
  const groupLabels = {
    baseline: "baseline",
    engineering: "feature engineering",
    ensemble: "ensembles",
    architecture: "architecture",
    other: "other",
  };
  select.replaceChildren();
  for (const key of groupOrder) {
    const items = groups.get(key);
    if (!items || !items.length) continue;
    const group = el("optgroup");
    group.label = groupLabels[key] || key;
    for (const item of items) {
      const option = el("option", "", `${item.id} — ${item.title || item.name || item.id}`);
      option.value = item.id;
      option.title = item.best_for || item.attribution || "";
      group.append(option);
    }
    select.append(group);
    groups.delete(key);
  }
  for (const [key, items] of groups) {
    const group = el("optgroup");
    group.label = groupLabels[key] || key;
    for (const item of items) {
      const option = el("option", "", `${item.id} — ${item.title || item.name || item.id}`);
      option.value = item.id;
      option.title = item.best_for || item.attribution || "";
      group.append(option);
    }
    select.append(group);
  }
  if (!pipelines.length) {
    const unavailable = el("option", "", "pipelines недоступны — проверьте recipe-каталог");
    unavailable.value = "";
    unavailable.disabled = true;
    select.append(unavailable);
  }
  select.disabled = !pipelines.length;
  for (const id of ["#variant-add-pipelines", "#pipeline-select-all", "#pipeline-select-none"]) {
    const button = $(id);
    if (button) button.disabled = !pipelines.length;
  }
  populatePipelinePresets();
  const presetSelect = $("#pipeline-preset");
  if (presetSelect) {
    presetSelect.disabled = !pipelines.length || !(state.config?.pipeline_presets || []).length;
  }
  updatePipelinePresetNote();
}

function populatePipelinePresets() {
  const select = $("#pipeline-preset");
  if (!select) return;
  const presets = Array.isArray(state.config?.pipeline_presets)
    ? state.config.pipeline_presets
    : [];
  select.replaceChildren();
  const custom = el("option", "", "свой выбор");
  custom.value = "";
  select.append(custom);
  const kindOrder = ["tier", "specialty", "all"];
  const kindLabels = { tier: "Уровень", specialty: "Специализация", all: "Все" };
  for (const kind of kindOrder) {
    const items = presets.filter((preset) => preset.kind === kind);
    if (!items.length) continue;
    const group = el("optgroup");
    group.label = kindLabels[kind];
    for (const preset of items) {
      const count = Array.isArray(preset.pipelines) ? preset.pipelines.length : 0;
      const option = el("option", "", `${preset.title} · ${count}`);
      option.value = preset.id;
      option.title = preset.description || "";
      group.append(option);
    }
    select.append(group);
  }
  select.disabled = !presets.length;
  select.value = "";
}

function currentPipelinePreset() {
  const select = $("#pipeline-preset");
  if (!select || !select.value) return null;
  return (state.config?.pipeline_presets || []).find((preset) => preset.id === select.value) || null;
}

function selectedPipelineIds() {
  const select = $("#recipe-batch");
  return select
    ? Array.from(select.selectedOptions).map((option) => option.value).filter(Boolean)
    : [];
}

function samePipelineSet(left, right) {
  if (left.length !== right.length) return false;
  const wanted = new Set(right);
  return left.every((id) => wanted.has(id));
}

function updatePipelinePresetNote() {
  const note = $("#pipeline-preset-note");
  if (!note) return;
  const selected = selectedPipelineIds();
  const total = $("#recipe-batch")
    ? Array.from($("#recipe-batch").options).filter((option) => option.value).length
    : 0;
  if (!total) {
    note.textContent = "Pipelines недоступны — проверьте recipe-каталог.";
    return;
  }
  const preset = currentPipelinePreset();
  if (preset && samePipelineSet(selected, preset.pipelines || [])) {
    note.textContent = `${preset.description} Выбрано: ${selected.length}.`;
    return;
  }
  note.textContent = selected.length
    ? `Свой выбор: ${selected.length} из ${total} pipelines.`
    : "Ничего не выбрано. Выберите набор или отметьте pipelines вручную.";
}

function syncPipelinePreset() {
  const select = $("#pipeline-preset");
  if (!select) return;
  const selected = selectedPipelineIds();
  const match = (state.config?.pipeline_presets || []).find(
    (preset) => samePipelineSet(selected, preset.pipelines || []),
  );
  select.value = match ? match.id : "";
  updatePipelinePresetNote();
}

function applyPipelinePreset(presetId) {
  const select = $("#recipe-batch");
  if (!select) return;
  const preset = (state.config?.pipeline_presets || []).find((item) => item.id === presetId);
  if (!preset) {
    updatePipelinePresetNote();
    updateRunEnabled();
    return;
  }
  const wanted = new Set(preset.pipelines || []);
  for (const option of select.options) option.selected = wanted.has(option.value);
  updatePipelinePresetNote();
  updateRunEnabled();
  toast(`Набор «${preset.title}»: выбрано ${wanted.size} pipelines.`);
}

function modelGroupKey(model) {
  return `${model.source || "other"}|${model.family || "other"}`;
}

function modelGroupLabel(key) {
  const [source, family] = key.split("|");
  const sourceLabel = { native: "kconv native", sklearn: "scikit-learn", tabpfn: "TabPFN" }[source] || source;
  const familyLabel = {
    tree: "trees / boosting",
    linear: "linear",
    foundation: "foundation",
  }[family] || family;
  return `${sourceLabel} · ${familyLabel}`;
}

function modelGroupRank(key) {
  const [source, family] = key.split("|");
  const sourceRank = { native: 0, sklearn: 1, tabpfn: 2 }[source] ?? 3;
  const familyRank = { tree: 0, linear: 1, foundation: 2 }[family] ?? 3;
  return sourceRank * 10 + familyRank;
}

function renderSelectedChips() {
  const node = $("#model-picked");
  node.replaceChildren();
  if (!state.selected.size) {
    node.append(el("span", "model-picked-empty", "Ничего не выбрано"));
    return;
  }
  for (const name of state.selected) {
    node.append(el("span", "model-chip", name));
  }
}

function renderModelParams() {
  const node = $("#model-params-rows");
  if (!node) return;
  node.replaceChildren();
  if (!state.selected.size) {
    node.append(el("div", "model-empty", "Сначала выберите хотя бы одну модель."));
    return;
  }
  for (const name of state.selected) {
    const row = el("label", "model-params-row");
    const title = el("span", "model-params-name", name);
    const input = document.createElement("input");
    input.type = "text";
    input.className = "model-params-input";
    input.dataset.model = name;
    input.value = state.modelParams.get(name) || "";
    input.placeholder = '{"n_estimators": 300, "learning_rate": 0.05}';
    input.spellcheck = false;
    input.autocomplete = "off";
    input.setAttribute("aria-label", `Гиперпараметры ${name} в JSON`);
    row.append(title, input);
    node.append(row);
  }
}

function parseModelParams() {
  const params = {};
  for (const name of state.selected) {
    const raw = (state.modelParams.get(name) || "").trim();
    if (!raw) continue;
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      return { params: {}, error: `Гиперпараметры ${name}: невалидный JSON (${error.message}).` };
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {
        params: {},
        error: `Гиперпараметры ${name}: нужен JSON-объект, например {"n_estimators": 300}.`,
      };
    }
    params[name] = parsed;
  }
  return { params, error: null };
}

function renderModelItem(model) {
  const label = el("label", "model-item");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.value = model.id;
  checkbox.checked = state.selected.has(model.id);
  checkbox.disabled = !model.available;
  const name = el("span", "model-name", model.id);
  const meta = el("span", "model-meta", model.available ? model.family || model.source || "" : "off");
  label.title = `${model.name || model.id} · ${(model.tasks || []).join(", ")}`;
  if (checkbox.checked) label.classList.add("selected");
  if (!model.available) label.classList.add("unavailable");
  label.append(checkbox, name, meta);
  return label;
}

function refreshModels() {
  if (!state.config) return;
  const task = currentTask();
  const availableSlots = modelsForTask(task);
  const compatible = new Set(availableSlots.filter((model) => model.available).map((model) => model.id));
  for (const name of Array.from(state.selected)) {
    if (!compatible.has(name)) state.selected.delete(name);
  }
  if (state.selected.size === 0) {
    for (const fallback of ["lgbm", "histgb"]) {
      if (compatible.has(fallback)) state.selected.add(fallback);
    }
    if (compatible.has("tabpfn") && trainRows() !== null && trainRows() <= 8000) state.selected.add("tabpfn");
  }

  const search = $("#model-search").value;
  const shown = modelsForTask(task, search);
  const container = $("#models");
  container.replaceChildren();
  if (!shown.length) {
    container.append(el("div", "model-empty", "Нет моделей под выбранную задачу или поиск."));
  }
  const groups = new Map();
  for (const model of shown) {
    const key = modelGroupKey(model);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(model);
  }
  const ordered = Array.from(groups.entries()).sort((a, b) => {
    const rank = modelGroupRank(a[0]) - modelGroupRank(b[0]);
    return rank || modelGroupLabel(a[0]).localeCompare(modelGroupLabel(b[0]));
  });
  for (const [key, models] of ordered) {
    const section = el("div", "model-group");
    const head = el("div", "model-group-head");
    head.append(
      el("span", "model-group-title", modelGroupLabel(key)),
      el("span", "model-group-count", String(models.length)),
    );
    const grid = el("div", "model-grid-inner");
    for (const model of models) grid.append(renderModelItem(model));
    section.append(head, grid);
    container.append(section);
  }
  updateModelCount();
  renderSelectedChips();
  renderModelParams();
  updateRunEnabled();
}

function updateModelCount() {
  $("#model-count").textContent = `${state.selected.size} выбрано`;
}

function currentRecipe() {
  if (!state.config) return null;
  return state.config.recipes.find((item) => item.id === $("#recipe").value) || null;
}

function renderRecipeCard() {
  const card = $("#recipe-card");
  const item = currentRecipe();
  if (!item) {
    card.classList.add("hidden");
    return;
  }
  const defaults = item.defaults || {};
  const validation = defaults.validation || {};
  const ensemble = defaults.ensemble || {};
  const hpo = defaults.hpo || {};
  card.classList.remove("hidden");
  $("#recipe-title").textContent = item.title || item.id;
  $("#recipe-group").textContent = `${item.group || "recipe"} · ${item.author || "unknown"}`;
  $("#recipe-kind").textContent = item.kind || "preset";
  $("#recipe-kind").className = `recipe-kind ${item.kind === "pipeline" ? "pipeline" : "preset"}`;
  $("#recipe-license").textContent = item.license || "license unknown";
  $("#recipe-best").textContent = item.best_for || item.attribution || "";
  const modelNames = item.models || [];
  $("#recipe-models").textContent = modelNames.length ? modelNames.join(" · ") : "—";
  $("#recipe-cv").textContent = `${validation.n_splits ?? "auto"} folds · ${validation.strategy || "auto"}`;
  const ensembleLabel = ensemble.method || "auto";
  $("#recipe-ensemble").textContent = ensemble.meta_model ? `${ensembleLabel} · meta ${ensemble.meta_model}` : ensembleLabel;
  $("#recipe-hpo").textContent = hpo.enabled ? `on · ${hpo.trials ?? "?"} trials` : "off";
  const blocks = item.pipeline && Array.isArray(item.pipeline.blocks)
    ? item.pipeline.blocks.filter((block) => block.enabled !== false).map((block) => block.type)
    : [];
  $("#recipe-blocks").textContent = blocks.length ? blocks.join(" → ") : "preset flags";
  $("#recipe-derived").textContent = item.derived_from || "original";
  $("#recipe-note").textContent = item.kind === "pipeline"
    ? "Исполняемый block pipeline: порядок блоков фиксирован, target/frequency auto разрешаются при создании проекта. Модели можно переопределить ниже."
    : "Preset задаёт defaults; модели подставляются автоматически. Ручной выбор ниже имеет приоритет.";
}

function applyRecipeModels({ announce = false } = {}) {
  const item = currentRecipe();
  if (!item || !state.config) return;
  const byId = new Map(state.config.models.map((model) => [model.id, model]));
  const task = currentTask();
  const names = (item.models || []).filter((name) => {
    const model = byId.get(name);
    return model && model.available && (task === "auto" || model.tasks.includes(task));
  });
  if (!names.length) {
    if (announce) toast("В recipe нет доступных моделей для выбранной задачи. Текущий выбор сохранён.", true);
    return;
  }
  state.selected.clear();
  for (const name of names) state.selected.add(name);
  refreshModels();
  if (announce) toast(`Модели recipe: ${names.join(", ")}.`);
}

function llmEnabled() {
  return Boolean($("#llm-enabled") && $("#llm-enabled").checked);
}

function llmPlanMatches() {
  const plan = state.llm.plan;
  if (!plan) return false;
  return plan.target === $("#target").value
    && (plan.id_col || "") === ($("#id-col").value || "");
}

function resetLlmPlan(message = "") {
  state.llm.plan = null;
  state.llm.message = message;
  const approved = $("#llm-approved");
  if (approved) approved.checked = false;
  renderLlmState();
}

function renderLlmPlan() {
  const plan = state.llm.plan;
  const table = $("#llm-plan-table");
  const body = table ? table.querySelector("tbody") : null;
  const wrap = $("#llm-plan-wrap");
  const approveRow = $("#llm-approve-row");
  const notice = $("#llm-notice");
  const meta = $("#llm-meta");
  if (!body || !wrap || !approveRow) return;
  const hasPlan = Boolean(plan && llmPlanMatches());
  wrap.classList.toggle("hidden", !hasPlan);
  approveRow.classList.toggle("hidden", !hasPlan);
  body.replaceChildren();
  if (hasPlan) {
    for (const feature of plan.features || []) {
      const row = document.createElement("tr");
      row.append(
        el("td", "", feature.name || "—"),
        el("td", "", feature.kind || "—"),
        el("td", "", feature.expression || "—"),
        el("td", "", feature.rationale || "—"),
      );
      body.append(row);
    }
    const bits = [`${plan.feature_count ?? (plan.features || []).length} features`, plan.provider || "provider"];
    if (plan.model) bits.push(plan.model);
    if (plan.usage && plan.usage.total_tokens !== null && plan.usage.total_tokens !== undefined) {
      bits.push(`${plan.usage.total_tokens} tokens`);
    }
    meta.textContent = bits.join(" · ");
  } else {
    meta.textContent = "";
  }
  const warnings = hasPlan ? plan.warnings || [] : [];
  if (warnings.length) {
    notice.textContent = `Проверьте предупреждения: ${warnings.join(" · ")}`;
    notice.classList.remove("hidden", "bad");
  } else {
    notice.textContent = "";
    notice.classList.add("hidden");
    notice.classList.remove("bad");
  }
  $("#llm-approved").disabled = !hasPlan || state.llm.busy;
}

function renderLlmState() {
  const body = $("#llm-body");
  const badge = $("#llm-badge");
  const status = $("#llm-status");
  if (!body || !badge || !status) return;
  const enabled = llmEnabled();
  const hasPlan = Boolean(state.llm.plan && llmPlanMatches());
  body.classList.toggle("hidden", !enabled);
  badge.classList.remove("on", "bad");
  if (!enabled) {
    badge.textContent = "off";
  } else if (state.llm.busy) {
    badge.textContent = "working";
    badge.classList.add("on");
  } else if (hasPlan) {
    badge.textContent = "ready";
    badge.classList.add("on");
  } else if (llmProviderIssue()) {
    badge.textContent = "blocked";
    badge.classList.add("bad");
  } else {
    badge.textContent = "not generated";
  }
  const providerIssue = llmProviderIssue();
  const generate = $("#llm-generate");
  if (generate) {
    generate.disabled = state.llm.busy || state.uploadBusy
      || !state.dataset || !state.dataset.files || !state.dataset.files.train
      || !$("#target").value || Boolean(providerIssue);
    generate.title = providerIssue || "";
  }
  const provider = $("#llm-provider");
  if (provider) provider.disabled = state.llm.busy;
  for (const selector of [
    "#llm-enabled", "#llm-max-features", "#llm-max-tokens", "#llm-model",
    "#llm-base-url", "#llm-json-mode", "#llm-local-discover",
  ]) {
    const node = $(selector);
    if (node) node.disabled = state.llm.busy;
  }
  if (state.llm.busy) {
    status.textContent = "Генерирую и проверяю план…";
  } else if (hasPlan) {
    status.textContent = `План готов: ${(state.llm.plan.features || []).length} признаков. Проверьте формулы.`;
  } else if (providerIssue) {
    status.textContent = providerIssue;
  } else if (state.llm.message) {
    status.textContent = state.llm.message;
  } else {
    status.textContent = "План ещё не создан.";
  }
  renderLlmPlan();
  updateRunEnabled();
}

function llmProviderIssue() {
  const provider = $("#llm-provider") ? $("#llm-provider").value : "mock";
  if (provider !== "openai") return null;
  const configured = Boolean(state.config && state.config.llm && state.config.llm.openai_configured);
  if (configured) return null;
  return "OpenAI key не найден: генерация заблокирована. Задайте KCONV_LLM_API_KEY "
    + "или OPENAI_API_KEY в окружении сервера, перезапустите start.cmd и повторите; "
    + "либо выберите mock/local.";
}

function syncLlmProvider() {
  const provider = $("#llm-provider") ? $("#llm-provider").value : "mock";
  const modelField = $("#llm-model-field");
  const baseField = $("#llm-base-field");
  const model = $("#llm-model");
  const note = $("#llm-provider-note");
  const localTools = $("#llm-local-tools");
  modelField.classList.toggle("hidden", provider === "mock");
  baseField.classList.toggle("hidden", provider !== "local");
  if (localTools) localTools.classList.toggle("hidden", provider !== "local");
  if (provider === "mock") {
    note.textContent = "mock работает без сети и ключа — удобно проверить весь путь.";
  } else if (provider === "openai") {
    if (!model.value || model.value === "qwen2.5-coder:7b") model.value = "gpt-4o-mini";
    model.placeholder = "gpt-4o-mini";
    const configured = Boolean(state.config && state.config.llm && state.config.llm.openai_configured);
    note.textContent = configured
      ? "Ключ найден в окружении. Запрос уйдёт только в OpenAI."
      : "Ключ не найден — генерация заблокирована. Задайте KCONV_LLM_API_KEY "
        + "или OPENAI_API_KEY в окружении и перезапустите start.cmd; либо mock/local.";
  } else {
    if (model.value === "gpt-4o-mini") model.value = "";
    model.placeholder = "auto / loaded model";
    note.textContent = "llama.cpp :8080, Ollama :11434, vLLM :8000. "
      + "Base URL и модель можно оставить пустыми — найду автоматически.";
  }
}

async function discoverLocalServer(quiet = false) {
  const status = $("#llm-local-status");
  const baseInput = $("#llm-base-url");
  const modelInput = $("#llm-model");
  if (!status || !baseInput || !modelInput) return;
  const base = baseInput.value.trim();
  status.textContent = "Ищу llama.cpp / Ollama / vLLM…";
  try {
    const suffix = base ? `?base_url=${encodeURIComponent(base)}` : "";
    const data = await api(`/api/llm/local/discover${suffix}`);
    const endpoints = data.endpoints || [];
    if (!endpoints.length) {
      status.textContent = "Локальный сервер не найден. Запустите llama-server "
        + "(порт 8080), Ollama (11434) или vLLM (8000).";
      if (!quiet) toast("Локальный LLM не найден.", true);
      return;
    }
    const endpoint = endpoints[0];
    const models = Array.isArray(endpoint.models) ? endpoint.models : [];
    let changed = false;
    if (baseInput.value.trim() !== endpoint.base_url) {
      baseInput.value = endpoint.base_url;
      changed = true;
    }
    if (!modelInput.value.trim() && models.length) {
      modelInput.value = models[0];
      changed = true;
    }
    const model = models.length ? models.join(", ") : "модель не объявлена";
    status.textContent = `Найден ${endpoint.runtime}: ${endpoint.base_url} · ${model}`;
    if (changed && state.llm.plan) {
      resetLlmPlan("Локальный endpoint изменён — сгенерируйте план заново.");
    }
  } catch (error) {
    status.textContent = error.hint ? `${error.message} ${error.hint}` : error.message;
    if (!quiet) toast(status.textContent, true);
  }
}

function llmRunBlocked() {
  if (!llmEnabled()) return null;
  if (!state.llm.plan) return "Включены LLM-признаки: сгенерируйте план.";
  if (!llmPlanMatches()) return "План LLM устарел: проверьте target/ID и сгенерируйте заново.";
  if (!$("#llm-approved").checked) return "Подтвердите формулы LLM-плана перед запуском.";
  return null;
}

function kaggleReady() {
  return Boolean(state.kaggle.auth && state.kaggle.auth.ready);
}

function kaggleCompetition() {
  const node = $("#kaggle-competition");
  return node ? node.value.trim() : "";
}

function updateKaggleEnabled() {
  const submit = $("#kaggle-submit");
  if (!submit) return;
  const auth = state.kaggle.auth;
  const blocked = state.kaggle.busy || state.kaggle.loading || isBusy();
  submit.disabled = blocked
    || !auth || !auth.ready
    || !state.kaggle.hasSubmission
    || !kaggleCompetition();
  const save = $("#kaggle-save");
  if (save) save.disabled = state.kaggle.busy;
  const refresh = $("#kaggle-refresh");
  if (refresh) refresh.disabled = state.kaggle.loading || state.kaggle.busy;
  for (const selector of ["#kaggle-competition", "#kaggle-token", "#kaggle-username", "#kaggle-key", "#kaggle-message"]) {
    const node = $(selector);
    if (node) node.disabled = state.kaggle.busy;
  }
  const note = $("#kaggle-submit-note");
  if (!note) return;
  if (!state.kaggle.hasSubmission) {
    note.textContent = "Сначала завершите полный прогон.";
  } else if (!auth) {
    note.textContent = "Проверяю Kaggle CLI и credentials…";
  } else if (!auth.ready) {
    note.textContent = "Нужны Kaggle CLI и credentials — см. статус ниже.";
  } else if (!kaggleCompetition()) {
    note.textContent = "Укажите competition slug.";
  } else if (isBusy()) {
    note.textContent = "Дождитесь завершения текущего run.";
  } else {
    note.textContent = "Готово: submission отправится одной кнопкой.";
  }
}

function renderKaggle() {
  const card = $("#kaggle-card");
  if (!card) return;
  const auth = state.kaggle.auth;
  const ready = Boolean(auth && auth.ready);
  const badge = $("#kaggle-status-badge");
  if (badge) {
    badge.textContent = state.kaggle.loading ? "проверка" : ready ? "готов" : "не настроен";
    badge.className = `card-state ${ready ? "ok" : "warn"}`;
  }
  const notice = $("#kaggle-notice");
  if (notice) {
    const messages = [];
    if (state.kaggle.message) messages.push(state.kaggle.message);
    if (auth && !auth.ready && Array.isArray(auth.warnings)) messages.push(...auth.warnings);
    if (auth && auth.ready && state.kaggle.environmentOverride) {
      messages.push("Используются credentials из переменных окружения — они имеют приоритет над сохранённым файлом.");
    }
    notice.textContent = messages.join(" ");
    notice.classList.toggle("hidden", !messages.length);
    notice.classList.toggle("bad", Boolean(auth && !auth.ready));
  }
  const meta = $("#kaggle-meta");
  if (meta) {
    meta.replaceChildren();
    if (auth) {
      meta.append(el("span", "", auth.cli_available
        ? `CLI: ${auth.cli_version || "found"}`
        : "CLI: не найден"));
      meta.append(el("span", "", auth.credential_source
        ? `credentials: ${auth.credential_source}`
        : "credentials: нет"));
      if (auth.config_path) meta.append(el("span", "", `config: ${auth.config_path}`));
    }
    if (state.kaggle.hasSubmission) {
      const file = state.kaggle.submission && state.kaggle.submission.name
        ? state.kaggle.submission.name
        : "submission.csv";
      meta.append(el("span", "", `submission: ${file}`));
    } else {
      meta.append(el("span", "", "submission: нет"));
    }
  }
  updateKaggleEnabled();
}

async function refreshKaggleStatus(quiet = false) {
  const project = projectName();
  if (!project) return;
  state.kaggle.loading = true;
  if (!quiet) state.kaggle.message = "";
  renderKaggle();
  try {
    const data = await api(`/api/kaggle/status?project=${encodeURIComponent(project)}`);
    if (projectName() !== project) return;
    state.kaggle.auth = data.auth || null;
    state.kaggle.hasSubmission = Boolean(data.has_submission);
    state.kaggle.runId = data.run_id || null;
    state.kaggle.submission = data.submission || null;
    state.kaggle.environmentOverride = Boolean(data.environment_override);
  } catch (error) {
    state.kaggle.message = error.hint ? `${error.message} ${error.hint}` : error.message;
    if (!quiet) toast(state.kaggle.message, true);
  } finally {
    state.kaggle.loading = false;
    renderKaggle();
  }
}

async function saveKaggleCredentials() {
  const token = $("#kaggle-token").value.trim();
  const username = $("#kaggle-username").value.trim();
  const key = $("#kaggle-key").value.trim();
  if (!token && (!username || !key)) {
    toast("Введите API token или пару username + API key.", true);
    return;
  }
  const savedWithToken = Boolean(token);
  state.kaggle.busy = true;
  state.kaggle.message = "";
  renderKaggle();
  try {
    const payload = { project: projectName() };
    if (token) {
      payload.token = token;
    } else {
      payload.username = username;
      payload.key = key;
    }
    const data = await postJson("/api/kaggle/credentials", payload);
    state.kaggle.auth = data.auth || null;
    state.kaggle.hasSubmission = Boolean(data.has_submission);
    state.kaggle.runId = data.run_id || null;
    state.kaggle.submission = data.submission || null;
    state.kaggle.environmentOverride = Boolean(data.environment_override);
    $("#kaggle-token").value = "";
    $("#kaggle-username").value = "";
    $("#kaggle-key").value = "";
    state.kaggle.message = data.environment_override
      ? "Credentials сохранены, но активны переменные окружения — они имеют приоритет."
      : savedWithToken
        ? "API token сохранён локально."
        : "Ключ сохранён локально.";
    toast(state.kaggle.message);
  } catch (error) {
    state.kaggle.message = error.hint ? `${error.message} ${error.hint}` : error.message;
    toast(state.kaggle.message, true);
  } finally {
    state.kaggle.busy = false;
    renderKaggle();
  }
}

async function submitKaggle() {
  const competition = kaggleCompetition();
  if (!competition) {
    toast("Укажите competition slug.", true);
    return;
  }
  if (!state.kaggle.hasSubmission) {
    toast("Submission ещё не создан.", true);
    return;
  }
  if (!kaggleReady()) {
    toast("Сначала сохраните Kaggle credentials.", true);
    return;
  }
  state.kaggle.busy = true;
  state.kaggle.message = "Отправляю submission в Kaggle…";
  renderKaggle();
  try {
    const data = await postJson("/api/kaggle/submit", {
      project: projectName(),
      run_id: state.kaggle.runId,
      competition,
      message: $("#kaggle-message").value.trim() || "kconv submission",
    });
    const rows = data.rows === null || data.rows === undefined ? "" : ` · ${data.rows} rows`;
    state.kaggle.message = `Отправлено: ${data.competition || competition}${rows}.`;
    toast(state.kaggle.message);
  } catch (error) {
    state.kaggle.message = error.hint ? `${error.message} ${error.hint}` : error.message;
    toast(state.kaggle.message, true);
  } finally {
    state.kaggle.busy = false;
    renderKaggle();
  }
}

async function generateLlmPlan() {
  if (state.llm.busy) return;
  const providerIssue = llmProviderIssue();
  if (providerIssue) {
    state.llm.message = providerIssue;
    toast(providerIssue, true);
    renderLlmState();
    return;
  }
  const target = $("#target").value;
  if (!target) {
    toast("Сначала выберите target.", true);
    return;
  }
  const provider = $("#llm-provider").value;
  const maxFeatures = Number($("#llm-max-features").value);
  if (!Number.isInteger(maxFeatures) || maxFeatures < 1 || maxFeatures > 20) {
    toast("Максимум признаков: целое число 1–20.", true);
    return;
  }
  const maxTokens = Number($("#llm-max-tokens").value);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 8192) {
    toast("Max tokens: целое число 1–8192.", true);
    return;
  }
  const ideas = ($("#llm-ideas")?.value || "").trim();
  const code = ($("#llm-code")?.value || "").trim();
  if (code.length > 12000) {
    toast("Код/псевдокод: не более 12000 символов.", true);
    return;
  }
  const repairs = Number($("#llm-repairs")?.value ?? 2);
  if (!Number.isInteger(repairs) || repairs < 0 || repairs > 5) {
    toast("Repair attempts: целое число 0–5.", true);
    return;
  }
  state.llm.plan = null;
  state.llm.message = "";
  state.llm.busy = true;
  renderLlmState();
  try {
    const result = await postJson("/api/llm/propose", {
      target,
      id_col: $("#id-col").value || null,
      provider,
      model: provider === "mock" ? null : $("#llm-model").value.trim(),
      base_url: provider === "local" ? $("#llm-base-url").value.trim() || null : null,
      max_features: maxFeatures,
      max_tokens: maxTokens,
      json_mode: Boolean($("#llm-json-mode").checked),
      ideas,
      code,
      repairs,
    });
    state.llm.plan = result;
    state.llm.message = "";
    const approved = $("#llm-approved");
    if (approved) approved.checked = false;
    toast(`LLM-план готов: ${result.feature_count || (result.features || []).length} признаков.`);
  } catch (error) {
    state.llm.plan = null;
    state.llm.message = error.hint ? `${error.message} ${error.hint}` : error.message;
    toast(state.llm.message, true);
  } finally {
    state.llm.busy = false;
    renderLlmState();
  }
}

function variantId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function variantResources(payload) {
  const resources = [];
  if (payload.device) resources.push(String(payload.device));
  if (payload.cpu_threads !== null && payload.cpu_threads !== undefined) {
    resources.push(Number(payload.cpu_threads) === 0 ? "CPU auto" : `${payload.cpu_threads} CPU`);
  }
  if (payload.ram_gb !== null && payload.ram_gb !== undefined) {
    resources.push(`${payload.ram_gb} GB RAM`);
  }
  if (payload.vram_gb !== null && payload.vram_gb !== undefined) {
    resources.push(`${payload.vram_gb} GB VRAM`);
  }
  return resources;
}

function variantMeta(payload) {
  const sample = payload.sample_pct && payload.sample_pct !== 100 ? ` · ${payload.sample_pct}%` : "";
  const effort = payload.profile ? `profile ${payload.profile}` : payload.strategy;
  const resources = variantResources(payload);
  const resource = resources.length ? ` · ${resources.join(" ")}` : "";
  const llm = payload.llm_plan_id ? " · LLM" : "";
  const tuned = payload.model_params && Object.keys(payload.model_params).length ? " · manual params" : "";
  return `${effort} · ${payload.recipe || "recipe"} · ${payload.until === "report" ? "full run" : "CV only"}${sample}${resource}${tuned}${llm}`;
}

function renderVariants() {
  const list = $("#variants");
  list.replaceChildren();
  if (!state.variants.length) {
    list.append(el("div", "variant-empty", "Список пуст. Настройте запуск и нажмите «Добавить текущий вариант»."));
  } else {
    state.variants.forEach((variant, index) => {
      const row = el("div", "variant-row");
      row.dataset.variantId = variant.id;
      const head = el("div", "variant-row-head");
      const label = document.createElement("input");
      label.type = "text";
      label.className = "variant-label";
      label.value = variant.label;
      label.setAttribute("aria-label", `Название варианта ${index + 1}`);
      const remove = el("button", "button danger small variant-remove", "Удалить");
      remove.type = "button";
      remove.dataset.variantId = variant.id;
      head.append(el("span", "variant-index", String(index + 1)), label, remove);
      const chips = el("div", "variant-chips");
      chips.append(el("span", "variant-chip", variant.payload.models.join(" + ")));
      chips.append(el("span", "variant-chip", variant.payload.until === "report" ? "full" : "CV"));
      if (variant.payload.profile) chips.append(el("span", "variant-chip", `profile ${variant.payload.profile}`));
      variantResources(variant.payload).forEach((resource) => {
        chips.append(el("span", "variant-chip", resource));
      });
      if (variant.payload.model_params && Object.keys(variant.payload.model_params).length) {
        chips.append(el("span", "variant-chip", "manual params"));
      }
      if (variant.payload.llm_plan_id) chips.append(el("span", "variant-chip", "LLM"));
      row.append(head, el("div", "variant-meta", variantMeta(variant.payload)), chips);
      list.append(row);
    });
  }
  $("#variant-hint").textContent = state.variants.length
    ? `${state.variants.length} вариант(ов) готовы к последовательному запуску.`
    : "Список пуст. Добавьте хотя бы один вариант.";
  updateRunEnabled();
}

function addVariant() {
  const error = runConfigError();
  if (error) {
    toast(error, true);
    return;
  }
  const payload = buildRunPayload();
  state.variants.push({ id: variantId(), label: autoVariantLabel(payload), payload });
  renderVariants();
  toast(`Вариант добавлен: ${state.variants.length}.`);
}

function variantSignature(payload) {
  return JSON.stringify({
    recipe: payload.recipe || null,
    models: payload.models || [],
    profile: payload.profile || null,
    strategy: payload.strategy || null,
    until: payload.until || null,
    sample_pct: payload.sample_pct || 100,
    model_params: payload.model_params || {},
    llm_plan_id: payload.llm_plan_id || null,
    llm_approved: Boolean(payload.llm_approved),
  });
}

function addSelectedPipelines() {
  const error = runConfigError();
  if (error) {
    toast(error, true);
    return;
  }
  const select = $("#recipe-batch");
  const recipes = select
    ? Array.from(select.selectedOptions).map((option) => option.value).filter(Boolean)
    : [];
  if (!recipes.length) {
    toast("Выберите хотя бы один pipeline в пакете.", true);
    return;
  }
  const base = buildRunPayload();
  const task = currentTask();
  const knownModels = state.config?.models || [];
  const allRecipes = state.config?.recipes || [];
  let added = 0;
  let skipped = 0;
  for (const recipe of recipes) {
    const item = allRecipes.find((entry) => entry.id === recipe);
    const recipeModels = (item?.models || []).filter((name) => {
      const model = knownModels.find((entry) => entry.id === name);
      return model && model.available && (task === "auto" || model.tasks.includes(task));
    });
    const models = recipeModels.length ? recipeModels : base.models;
    const payload = { ...base, recipe, models };
    if (payload.model_params) {
      const filtered = {};
      for (const [name, value] of Object.entries(payload.model_params)) {
        if (models.includes(name)) filtered[name] = value;
      }
      if (Object.keys(filtered).length) payload.model_params = filtered;
      else delete payload.model_params;
    }
    const signature = variantSignature(payload);
    if (state.variants.some((variant) => variantSignature(variant.payload) === signature)) {
      skipped += 1;
      continue;
    }
    state.variants.push({ id: variantId(), label: autoVariantLabel(payload), payload });
    added += 1;
  }
  renderVariants();
  const skippedNote = skipped ? ` Пропущено дублей: ${skipped}.` : "";
  toast(`Добавлено pipelines: ${added}.${skippedNote} Всего вариантов: ${state.variants.length}.`);
}

async function startBatch() {
  const error = runConfigError();
  if (error) {
    toast(error, true);
    return;
  }
  if (!state.variants.length) {
    toast("Добавьте хотя бы один вариант.", true);
    return;
  }
  const project = projectName();
  const variants = state.variants.map((variant) => ({ ...variant.payload, project, label: variant.label }));
  localStorage.setItem("kconv.project", project);
  state.currentProject = project;
  state.lastLog = "";
  try {
    const started = await postJson("/api/run/batch", { variants });
    state.currentStatus = started.status || "queued";
    state.batchActive = true;
    state.queue = started.items || [];
    state.variants = [];
    renderVariants();
    renderQueue(state.queue);
    setStatusBadge($("#run-status-badge"), state.currentStatus);
    $("#run-hint").textContent = `В очереди ${variants.length} запуск(ов). Они выполнятся по одному.`;
    state.followLog = true;
    $("#log-wrap").open = true;
    toast(`Очередь создана: ${variants.length}.`);
    pollStatus(true);
  } catch (requestError) {
    toast(requestError.hint ? `${requestError.message} ${requestError.hint}` : requestError.message, true);
  }
}

function renderQueue(items) {
  state.queue = Array.isArray(items) ? items : [];
  const node = $("#queue-list");
  node.replaceChildren();
  const pending = queuePendingCount();
  $("#queue-state").textContent = pending
    ? `${pending} в очереди`
    : state.queue.length
      ? `${state.queue.length} запусков`
      : "0 в очереди";
  const filter = state.queueFilter || "all";
  const visible = filter === "all"
    ? state.queue
    : state.queue.filter((item) => String(item.status || "queued") === filter);
  if (!state.queue.length) {
    node.classList.add("hidden");
    updateRunEnabled();
    return;
  }
  node.classList.remove("hidden");
  if (!visible.length) {
    node.append(el("div", "hint", "Нет запусков с выбранным статусом."));
    updateRunEnabled();
    return;
  }
  for (const item of visible) {
    const status = String(item.status || "queued");
    const row = el("div", `queue-row ${status}`);
    const main = el("div", "queue-main");
    main.append(el("b", "", item.label || item.run_id || "run"));
    const bits = [];
    if (item.profile) bits.push(`profile ${item.profile}`);
    else if (item.strategy) bits.push(item.strategy);
    if (item.models && item.models.length) bits.push(item.models.join("+"));
    if (item.until) bits.push(item.until === "report" ? "full" : "CV");
    if (item.llm) bits.push("LLM");
    if (item.run_id) bits.push(item.run_id);
    main.append(el("span", "", bits.join(" · ")));
    if (item.error) main.title = item.error;
    row.append(main, el("span", `queue-badge ${status}`, STATUS_LABELS[status] || status));
    if (item.score !== null && item.score !== undefined) row.append(el("span", "queue-score", fmtScore(item.score)));
    if (["failed", "stopped", "cancelled"].includes(status)) {
      const retry = el("button", "button ghost compact queue-retry", "Retry");
      retry.type = "button";
      retry.dataset.itemId = item.id;
      retry.title = item.error || "Повторить запуск";
      row.append(retry);
    }
    node.append(row);
  }
  updateRunEnabled();
}

function comparisonLabel(run) {
  const queued = state.queue.find((item) => item.run_id === run.run_id);
  return queued ? queued.label : run.run_id;
}

function comparisonDataset() {
  const select = $("#comparison-dataset");
  return select ? select.value : "";
}

function renderComparisonDatasets(datasets, selected) {
  const select = $("#comparison-dataset");
  if (!select) return;
  const previous = selected || select.value;
  select.replaceChildren();
  const all = el("option", "", "все датасеты");
  all.value = "";
  select.append(all);
  for (const dataset of datasets) {
    const option = el("option", "", `${dataset.label} · ${dataset.runs} run`);
    option.value = dataset.id;
    option.title = `dataset id ${dataset.id}`;
    select.append(option);
  }
  if (Array.from(select.options).some((option) => option.value === previous)) {
    select.value = previous;
  } else {
    select.value = "";
  }
}

function renderComparison(payload) {
  const summary = $("#comparison-summary");
  const table = $("#comparison-table");
  const head = table.querySelector("thead");
  const body = table.querySelector("tbody");
  const runs = payload.runs || [];
  const datasets = payload.datasets || [];
  summary.replaceChildren();
  body.replaceChildren();
  renderComparisonDatasets(datasets, payload.dataset || "");
  if (!runs.length) {
    summary.textContent = payload.dataset
      ? "Для выбранного датасета ещё нет runs."
      : "Для этого проекта ещё нет runs.";
    const empty = document.createElement("tr");
    const cell = el("td", "empty-cell", "Запустите очередь или полный прогон — сравнение появится здесь.");
    cell.colSpan = 9;
    empty.append(cell);
    body.append(empty);
    return;
  }
  const best = runs.find((run) => run.best);
  const metrics = new Set(datasets.map((dataset) => dataset.metric).filter(Boolean));
  if (payload.dataset) {
    const meta = datasets.find((dataset) => dataset.id === payload.dataset);
    summary.append(
      el("span", "", `metric ${payload.metric || "—"}`),
      el("span", "", `${payload.direction === "minimize" ? "minimize" : "maximize"}`),
      el("span", "", `best ${best ? comparisonLabel(best) : "—"}`),
      el("span", "", `${runs.length} runs`),
      el("span", "", meta ? meta.label : "датасет"),
    );
  } else {
    summary.append(
      el("span", "", metrics.size > 1 ? "metric mixed" : `metric ${payload.metric || "—"}`),
      el("span", "", `${datasets.length} датасетов`),
      el("span", "", `${runs.length} runs`),
    );
    for (const dataset of datasets) {
      const winner = runs.find((run) => run.run_id === dataset.best_run_id);
      summary.append(
        el("span", "", `${dataset.label}: ${winner ? comparisonLabel(winner) : "—"}`),
      );
    }
  }
  head.replaceChildren();
  const headRow = document.createElement("tr");
  for (const label of ["#", "run", "датасет", "strategy", "models", "OOF", "best single", "time", "status"]) {
    headRow.append(el("th", "", label));
  }
  head.append(headRow);
  for (const run of runs) {
    const row = el("tr", run.best ? "best-run" : "");
    row.append(el("td", "", String(run.index).padStart(2, "0")));
    const runCell = el("td", "", comparisonLabel(run));
    runCell.title = run.run_id;
    if (run.has_report) {
      const link = el("a", "table-link", " report");
      link.href = `/api/report?project=${encodeURIComponent(payload.project)}&run_id=${encodeURIComponent(run.run_id)}`;
      runCell.append(link);
    }
    row.append(runCell);
    const datasetCell = el("td", "", run.dataset_label || "—");
    datasetCell.title = run.dataset_id ? `dataset id ${run.dataset_id}` : "";
    row.append(datasetCell);
    row.append(el("td", "", run.profile ? `profile ${run.profile}` : run.strategy || "—"));
    row.append(el("td", "", (run.models || []).join("+") || "—"));
    row.append(el("td", "score-cell", run.oof_score === null || run.oof_score === undefined ? "—" : fmtScore(run.oof_score)));
    const single = run.best_single && run.best_single.name
      ? `${run.best_single.name} ${fmtScore(run.best_single.score)}`
      : "—";
    row.append(el("td", "", single));
    row.append(el("td", "", fmtDuration(run.duration_sec) || "—"));
    row.append(el("td", "", run.status || "—"));
    body.append(row);
  }
}

function refreshComparison() {
  const project = projectName();
  if (!project) return Promise.resolve();
  const params = new URLSearchParams({ project });
  const dataset = comparisonDataset();
  if (dataset) params.set("dataset", dataset);
  return api(`/api/comparison?${params.toString()}`).then((payload) => {
    if (projectName() === project) renderComparison(payload);
  }).catch((error) => {
    const summary = $("#comparison-summary");
    summary.textContent = error.message;
  });
}

function refreshComparisonIfNeeded(data) {
  const completed = state.queue
    .filter((item) => item.status === "completed")
    .map((item) => item.run_id || item.id)
    .join(",");
  const key = `${data.project || projectName()}|${data.run_id || ""}|${completed}|${state.queue.length}|${comparisonDataset()}`;
  if (key === state.comparisonKey) return;
  state.comparisonKey = key;
  refreshComparison().catch(() => {});
}

function trainRows() {
  const rows = state.dataset && state.dataset.previews && state.dataset.previews.train
    ? state.dataset.previews.train.rows
    : null;
  return typeof rows === "number" ? rows : null;
}

function updateProfileState() {
  const profile = $("#profile").value;
  const strategy = $("#strategy");
  const note = $("#profile-note");
  const resourceNote = $("#resource-note");
  const resourceInputs = [$("#device"), $("#cpu-threads"), $("#ram-gb"), $("#vram-gb")].filter(Boolean);
  const device = $("#device");
  const gpuOption = device ? device.querySelector('option[value="cuda"]') : null;
  if (profile) {
    strategy.disabled = true;
    note.textContent = "Профиль выбран: CLI применит его лимиты и переопределит стратегию.";
    note.classList.add("active");
    resourceInputs.forEach((input) => { input.disabled = false; });
    const level = Number(profile);
    if (gpuOption) {
      gpuOption.disabled = level <= 3;
      if (gpuOption.disabled && device.value === "cuda") device.value = "";
    }
    if (resourceNote) {
      resourceNote.textContent = level <= 3
        ? "Профиль CPU-only: устройство CPU/auto, GPU/CUDA недоступен. RAM/VRAM — ориентир."
        : "Можно переопределить CPU threads, RAM и VRAM. RAM/VRAM — ориентир, не жёсткий cgroup.";
    }
  } else {
    strategy.disabled = false;
    note.textContent = "Профиль 1–5 переопределяет стратегию и подбирает лимиты CPU/GPU.";
    note.classList.remove("active");
    resourceInputs.forEach((input) => { input.disabled = true; });
    if (device) device.value = "";
    if (resourceNote) resourceNote.textContent = "Выберите профиль, чтобы переопределить CPU threads, RAM и VRAM.";
  }
}

function resourceConfigError() {
  if (!$("#profile").value) return null;
  const profile = Number($("#profile").value);
  if ($("#device").value === "cuda" && profile <= 3) {
    return `Профиль ${profile} — CPU-only; выберите профиль 4–5 для GPU.`;
  }
  const specs = [
    ["#cpu-threads", "CPU threads", 0, 256],
    ["#ram-gb", "RAM GB", 1, 1024],
    ["#vram-gb", "VRAM GB", 0, 256],
  ];
  for (const [selector, label, minimum, maximum] of specs) {
    const raw = $(selector).value.trim();
    if (!raw) continue;
    const value = Number(raw);
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
      return `${label}: укажите целое число ${minimum}–${maximum} или оставьте пусто.`;
    }
  }
  return null;
}

function updateRunEnabled() {
  const hasTrain = Boolean(state.dataset && state.dataset.files && state.dataset.files.train);
  const hasTarget = Boolean($("#target").value);
  const info = targetSummary();
  const n = info && Number.isFinite(Number(info.nunique)) ? Number(info.nunique) : null;
  const task = currentTask();
  const taskMismatch = n !== null
    && ((task === "binary" && n !== 2) || (task === "multiclass" && n === 2));
  const llmBlocked = Boolean(llmRunBlocked());
  const blocked = state.uploadBusy || !hasTrain || !hasTarget || taskMismatch || llmBlocked;
  $("#run").disabled = blocked || isServerBusy();
  $("#stop").disabled = !isBusy();
  $("#variant-add").disabled = blocked;
  const batch = $("#recipe-batch");
  $("#variant-add-pipelines").disabled = blocked
    || !batch
    || batch.disabled
    || !batch.selectedOptions.length;
  $("#variant-start").disabled = blocked || !state.variants.length;
  $("#variant-clear").disabled = !state.variants.length;
  updatePipelinePresetNote();
  updateKaggleEnabled();
}

function isBusy() {
  return state.serverActive || state.batchActive;
}

function isServerBusy() {
  return state.serverBusy || state.batchActive;
}

function hasPendingQueue() {
  return state.queue.some((item) => ["queued", "starting", "running"].includes(String(item.status)));
}

function queuePendingCount() {
  return state.queue.filter((item) => ["queued", "starting", "running"].includes(String(item.status))).length;
}

function runConfigError() {
  if (state.config && state.config.recipe_error) {
    return "Recipe-каталог недоступен — исправьте ошибку в предупреждении и перезапустите сервер.";
  }
  if (!state.dataset || !state.dataset.files || !state.dataset.files.train) return "Сначала загрузите train.";
  const target = $("#target").value;
  if (!target) return "Выберите target-колонку.";
  if (!state.selected.size) return "Выберите хотя бы одну модель.";
  const modelParams = parseModelParams();
  if (modelParams.error) return modelParams.error;
  const resourceError = resourceConfigError();
  if (resourceError) return resourceError;
  const sample = optionalNumber("#sample-pct");
  if (sample !== null && (!Number.isInteger(sample) || sample < 1 || sample > 100)) {
    return "% датасета: целое число от 1 до 100.";
  }
  const project = projectName();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(project)) {
    return "Имя проекта: латиница, цифры, точка, дефис или подчёркивание.";
  }
  const info = targetSummary();
  const n = info && Number.isFinite(Number(info.nunique)) ? Number(info.nunique) : null;
  const task = currentTask();
  if (n !== null && task === "binary" && n !== 2) return `У target ${n} классов — binary не подходит.`;
  if (n !== null && task === "multiclass" && n === 2) return "У target 2 класса — выберите binary.";
  const llmError = llmRunBlocked();
  if (llmError) return llmError;
  return null;
}

function optionalNumber(selector) {
  const raw = $(selector).value.trim();
  return raw === "" ? null : Number(raw);
}

function buildRunPayload() {
  const parsedParams = parseModelParams();
  const profile = $("#profile").value || null;
  const payload = {
    project: projectName(),
    target: $("#target").value,
    id_col: $("#id-col").value || null,
    task: $("#task").value,
    metric: $("#metric").value,
    recipe: $("#recipe").value,
    strategy: $("#strategy").value,
    profile,
    sample_pct: optionalNumber("#sample-pct") ?? 100,
    device: profile ? ($("#device").value || null) : null,
    cpu_threads: profile ? optionalNumber("#cpu-threads") : null,
    ram_gb: profile ? optionalNumber("#ram-gb") : null,
    vram_gb: profile ? optionalNumber("#vram-gb") : null,
    models: Array.from(state.selected),
    until: $("#until-full").checked && state.dataset && state.dataset.files.test ? "report" : "train",
    force: true,
  };
  if (!parsedParams.error && Object.keys(parsedParams.params).length) {
    payload.model_params = parsedParams.params;
  }
  if (llmEnabled()) {
    payload.llm_plan_id = state.llm.plan ? state.llm.plan.plan_id : null;
    payload.llm_approved = Boolean($("#llm-approved").checked);
  }
  return payload;
}

function recipeShortName(recipe) {
  return String(recipe || "recipe").replace(/^(pipeline|preset)\//, "");
}

function autoVariantLabel(payload) {
  const effort = payload.profile ? `profile ${payload.profile}` : payload.strategy;
  const scope = payload.until === "report" ? "full" : "CV";
  const resources = variantResources(payload);
  const resource = resources.length ? ` · ${resources.join(" ")}` : "";
  const sample = payload.sample_pct && payload.sample_pct !== 100 ? ` · ${payload.sample_pct}%` : "";
  const llm = payload.llm_plan_id ? " · LLM" : "";
  const tuned = payload.model_params && Object.keys(payload.model_params).length ? " · tuned" : "";
  return `${effort} · ${recipeShortName(payload.recipe)} · ${payload.models.join("+")} · ${scope}${sample}${resource}${tuned}${llm}`;
}

function targetSummary() {
  return state.dataset && state.dataset.target_summary ? state.dataset.target_summary : null;
}

function syncTaskOptions(announce = false) {
  const task = $("#task");
  const info = targetSummary();
  const binary = task.querySelector('option[value="binary"]');
  const multiclass = task.querySelector('option[value="multiclass"]');
  if (!info || !Number.isFinite(Number(info.nunique))) {
    if (binary) binary.disabled = false;
    if (multiclass) multiclass.disabled = false;
    return;
  }
  const n = Number(info.nunique);
  if (binary) binary.disabled = n !== 2;
  if (multiclass) multiclass.disabled = n < 3;
  let changed = false;
  if (n === 2 && task.value === "multiclass") {
    task.value = "binary";
    changed = true;
  } else if (n >= 3 && task.value === "binary") {
    task.value = "multiclass";
    changed = true;
  }
  if (changed && announce) {
    toast(`У target ${n} класса — переключил задачу на ${task.value}.`);
  }
}

function renderTaskNote() {
  const note = $("#task-note");
  const info = targetSummary();
  if (!info || info.error) {
    note.textContent = "Определяется по target; binary для 2 классов.";
    note.classList.remove("active");
    return;
  }
  const n = Number(info.nunique);
  const noun = n === 1 ? "класс" : n >= 2 && n <= 4 ? "класса" : "классов";
  let text = `target: ${n} ${noun}`;
  if (info.missing) text += ` · ${Number(info.missing).toLocaleString("ru-RU")} NA`;
  const entries = Object.entries(info.counts || {}).slice(0, 4);
  const total = entries.reduce((sum, [, count]) => sum + Number(count), 0);
  if (total > 0) {
    const parts = entries.map(([label, count]) => `${label} ${(Number(count) / total * 100).toFixed(1)}%`);
    text += ` · ${parts.join(" · ")}`;
  }
  note.textContent = text;
  const task = currentTask();
  const mismatch = (task === "binary" && n !== 2) || (task === "multiclass" && n === 2);
  note.classList.toggle("active", Boolean(mismatch));
}

function renderFileCards(data) {
  const pairs = [
    ["train", data.files.train, "#train-file"],
    ["test", data.files.test, "#test-file"],
    ["sample", data.files.sample, "#sample-file"],
  ];
  for (const [kind, file, selector] of pairs) {
    const node = $(selector);
    if (file) {
      node.textContent = `${file.name} · ${fmtBytes(file.size)}`;
      node.classList.add("loaded");
      $(`#drop-${kind}`).classList.add("loaded");
    } else {
      node.textContent = "не загружен";
      node.classList.remove("loaded");
      $(`#drop-${kind}`).classList.remove("loaded");
    }
  }
  $("#data-dir-label").textContent = data.data_dir || "";
}

function renderPreview(data) {
  const preview = data.previews && data.previews.train;
  const table = $("#preview-table");
  const head = table.querySelector("thead");
  const body = table.querySelector("tbody");
  head.replaceChildren();
  body.replaceChildren();
  if (!preview || preview.error || !preview.columns || !preview.columns.length) {
    const emptyHead = document.createElement("tr");
    emptyHead.append(el("th", "", "Загрузите train"));
    head.append(emptyHead);
    const emptyBody = document.createElement("tr");
    emptyBody.append(el("td", "empty-cell", preview && preview.error ? preview.error : "Здесь появится превью колонок."));
    body.append(emptyBody);
    $("#preview-meta").textContent = "—";
    return;
  }
  const row = document.createElement("tr");
  for (const column of preview.columns) row.append(el("th", "", column.name));
  head.append(row);
  const records = preview.head || [];
  if (!records.length) {
    const empty = document.createElement("tr");
    const cell = el("td", "empty-cell", "В файле нет строк.");
    cell.colSpan = preview.columns.length;
    empty.append(cell);
    body.append(empty);
  } else {
    for (const record of records) {
      const tr = document.createElement("tr");
      for (const column of preview.columns) {
        const value = record[column.name];
        tr.append(el("td", "", value === null || value === undefined ? "∅" : value));
      }
      body.append(tr);
    }
  }
  const rows = preview.rows === null || preview.rows === undefined ? "?" : Number(preview.rows).toLocaleString("ru-RU");
  $("#preview-meta").textContent = `${rows} строк · ${preview.columns.length} колонок`;
  $("#hero-rows").textContent = preview.rows === null || preview.rows === undefined
    ? "—"
    : Number(preview.rows).toLocaleString("ru-RU");
}

function renderFacts(data) {
  const facts = $("#dataset-facts");
  facts.replaceChildren();
  const detected = data.detected;
  const preview = data.previews && data.previews.train;
  const items = [];
  if (detected) {
    items.push(["target", detected.target]);
    items.push(["task", detected.task]);
    items.push(["metric", detected.metric]);
    if (detected.id_col) items.push(["id", detected.id_col]);
    items.push(["features", detected.feature_columns ? detected.feature_columns.length : "—"]);
    items.push(["numeric", detected.numeric_columns ? detected.numeric_columns.length : "—"]);
    items.push(["categorical", detected.categorical_columns ? detected.categorical_columns.length : "—"]);
  } else if (preview && preview.columns) {
    items.push(["columns", preview.columns.length]);
    if (preview.rows !== null) items.push(["rows", Number(preview.rows).toLocaleString("ru-RU")]);
  }
  const info = data.target_summary;
  if (info && Number.isFinite(Number(info.nunique))) items.push(["classes", Number(info.nunique)]);
  for (const [label, value] of items) {
    const node = el("span", "fact", `${label} `);
    node.append(el("b", "", value));
    facts.append(node);
  }

  const warning = $("#detected-warning");
  warning.replaceChildren();
  const messages = [];
  if (data.detected_error) messages.push(data.detected_error);
  if (detected && detected.warnings && detected.warnings.length) messages.push(...detected.warnings);
  if (messages.length) {
    warning.classList.remove("hidden");
    warning.classList.toggle("bad", Boolean(data.detected_error));
    for (const message of messages) warning.append(el("div", "", message));
  } else {
    warning.classList.add("hidden");
  }
}

function applyDetection(data) {
  const detected = data.detected;
  const preview = data.previews && data.previews.train;
  const columns = preview && preview.columns ? preview.columns.map((column) => column.name) : [];
  setSelectOptions($("#target"), columns, null, true);
  setSelectOptions($("#id-col"), columns, "auto", true);
  if (detected) {
    if (!state.manualTarget || !$("#target").value) $("#target").value = detected.target;
    if (detected.id_col) $("#id-col").value = detected.id_col;
    if ($("#task").value === "auto" && detected.task) $("#task").value = detected.task;
    if ($("#metric").value === "auto" && detected.metric) $("#metric").value = detected.metric;
  }
  syncTaskOptions(false);
  renderTaskNote();
  const hasTest = Boolean(data.files.test);
  const full = $("#until-full");
  full.disabled = !hasTest;
  if (!hasTest) {
    full.checked = false;
  } else if (!state.fullRunTouched) {
    full.checked = true;
  }
  refreshModels();
  updateRunEnabled();
}

function renderDataset(data) {
  state.dataset = data;
  renderFileCards(data);
  renderPreview(data);
  renderFacts(data);
  applyDetection(data);
  if (state.llm.plan) {
    const trainInfo = data.files && data.files.train ? data.files.train : null;
    const plan = state.llm.plan;
    const stale = !trainInfo
      || trainInfo.modified !== plan.dataset_modified
      || plan.target !== $("#target").value
      || (plan.id_col || "") !== ($("#id-col").value || "");
    if (stale) {
      resetLlmPlan("Датасет или target изменились — сгенерируйте LLM-план заново.");
    } else {
      renderLlmState();
    }
  } else {
    renderLlmState();
  }
  $("#dataset-status").textContent = data.ready ? "готов" : "нет train";
  $("#dataset-status").className = `card-state ${data.ready ? "ok" : "warn"}`;
  $("#run-hint").textContent = data.ready
    ? "Проверьте target и выберите модели. Запуск создаст папку в competitions/."
    : "Сначала загрузите train. Остальное kconv определит сам.";
}

async function refreshDataset() {
  const data = await api("/api/dataset");
  renderDataset(data);
  if (!state.initialized) {
    state.initialized = true;
    const latest = localStorage.getItem("kconv.project");
    if (latest) $("#project").value = latest;
    state.currentProject = latest || projectName();
  }
  return data;
}

async function refreshStats() {
  state.stats = await api("/api/stats");
  const stats = state.stats;
  $("#stat-models").textContent = String(stats.models.total);
  $("#stat-recipes").textContent = String(stats.recipes);
  $("#stat-sources").textContent = String(stats.sources.total);
  $("#hero-models").textContent = String(stats.models.total);
  $("#hero-recipes").textContent = String(stats.recipes);
  const rows = stats.dataset && stats.dataset.previews && stats.dataset.previews.train
    ? stats.dataset.previews.train.rows
    : null;
  $("#stat-rows").textContent = rows === null || rows === undefined ? "—" : Number(rows).toLocaleString("ru-RU");
  $("#hero-rows").textContent = rows === null || rows === undefined ? "—" : Number(rows).toLocaleString("ru-RU");

  const detail = $("#stats-detail");
  detail.replaceChildren();
  const lines = [
    ["sources downloaded", `${stats.sources.downloaded} / ${stats.sources.total}`],
    ["recipe-backed sources", stats.sources.recipe_backed_sources],
    ["pipeline sources", stats.sources.pipeline_sources],
    ["experiments", stats.experiments],
    ["семейства моделей", Object.entries(stats.models.by_family || {}).map(([key, value]) => `${key}:${value}`).join(" · ")],
    ["binary / multiclass / regression", `${stats.models.by_task.binary} / ${stats.models.by_task.multiclass} / ${stats.models.by_task.regression}`],
  ];
  for (const [label, value] of lines) {
    const line = el("div", "stat-line");
    line.append(el("span", "", label), el("b", "", value));
    detail.append(line);
  }
}

function renderStages(stages) {
  const container = $("#stages");
  container.replaceChildren();
  for (const stage of stages || []) {
    const status = String(stage.status || "pending").toLowerCase();
    const node = el("div", `stage ${status}`);
    node.append(el("span", "stage-name", String(stage.name || "").replace(/^\d+_/, "").replaceAll("_", " ")));
    node.append(el("span", "stage-status", STAGE_LABELS[status] || status));
    if (stage.duration_sec !== null && stage.duration_sec !== undefined) {
      node.append(el("span", "stage-time", fmtDuration(stage.duration_sec)));
    }
    if (stage.error) node.title = stage.error;
    container.append(node);
  }
}

function filterLogLines(log) {
  const level = String(state.logLevel || "").toUpperCase();
  const query = String(state.logQuery || "").trim().toLowerCase();
  return log.filter((line) => {
    const upper = line.toUpperCase();
    if (level === "ERROR" && !upper.includes("ERROR") && !upper.includes("TRACEBACK")) return false;
    if (level === "WARNING" && !upper.includes("WARN")) return false;
    if (level === "INFO" && !["INFO", "STARTED", "DONE", "STAGE"].some((token) => upper.includes(token))) return false;
    return !query || line.toLowerCase().includes(query);
  });
}

function renderLog(data) {
  const log = data.log || [];
  const visible = filterLogLines(log);
  const text = visible.join("\n");
  if (text !== state.lastLog) {
    const pre = $("#log");
    const wrap = $("#log-wrap");
    const wasAtBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 24;
    pre.textContent = text || (log.length ? "Нет строк по фильтру." : "Лог пока пуст.");
    $("#log-count").textContent = `${visible.length} / ${log.length} строк`;
    if (wasAtBottom || state.followLog) pre.scrollTop = pre.scrollHeight;
    state.lastLog = text;
    if (wrap && log.length) wrap.open = true;
  }
}

async function retryQueueItem(itemId) {
  if (!itemId) return;
  try {
    await postJson("/api/queue/retry", { item_id: itemId, project: projectName() });
    toast("Запуск возвращён в очередь.");
    pollStatus(true);
  } catch (error) {
    toast(error.hint ? `${error.message} ${error.hint}` : error.message, true);
  }
}

async function continueRun(runId, until) {
  if (!runId) return;
  try {
    await postJson("/api/run/continue", {
      project: projectName(),
      run_id: runId,
      until: until || "report",
    });
    state.serverActive = true;
    state.serverBusy = true;
    state.lastLog = "";
    state.followLog = true;
    $("#log-wrap").open = true;
    toast("Продолжение запуска поставлено в очередь.");
    pollStatus(true);
  } catch (error) {
    toast(error.hint ? `${error.message} ${error.hint}` : error.message, true);
  }
}

async function renderFailure(data) {
  const card = $("#failure-card");
  if (!card) return;
  const status = String(data.status || "");
  const available = Boolean(data.failure_available || data.failure_report || data.failure_json)
    && !["running", "queued", "starting"].includes(status);
  card.classList.toggle("hidden", !available);
  if (!available) {
    state.failureKey = "";
    state.failure = null;
    return;
  }
  const key = `${data.run_id || ""}|${data.failure_report || ""}|${data.failure_json || ""}`;
  if (key === state.failureKey) return;
  state.failureKey = key;
  const project = encodeURIComponent(data.project || projectName());
  const runId = encodeURIComponent(data.run_id || "");
  try {
    const payload = await api(`/api/failure?project=${project}&run_id=${runId}`);
    state.failure = payload;
    const failure = payload.failure || {};
    const exc = failure.exception || {};
    const bits = [
      exc.type,
      exc.message,
      failure.stage ? `stage ${failure.stage}` : "",
      data.status ? `status ${data.status}` : "",
    ].filter(Boolean);
    $("#failure-state").textContent = data.status || failure.status || "failed";
    $("#failure-meta").textContent = bits.join(" · ") || "Postmortem без деталей исключения.";
    const report = $("#failure-report");
    if (payload.report_html) report.innerHTML = payload.report_html;
    else report.textContent = payload.report || "Postmortem недоступен.";
    const download = $("#failure-download");
    download.href = `/api/failure?project=${project}&run_id=${runId}`;
    const button = $("#failure-continue");
    button.dataset.runId = payload.run_id || data.run_id || "";
    button.dataset.until = failure.until || data.until || "report";
    button.classList.toggle(
      "hidden",
      !["failed", "partial", "interrupted", "stale"].includes(String(data.status || failure.status)),
    );
  } catch (error) {
    card.classList.add("hidden");
    state.failureKey = "";
  }
}

function renderMetricCards(data) {
  const container = $("#metric-cards");
  container.replaceChildren();
  const blend = data.metrics && data.metrics.blend;
  if (!blend) return;
  const metricName = blend.metric || data.metrics.metric || "score";
  const cards = [
    {
      label: `blend ${metricName}`,
      value: blend.oof_score,
      note: blend.method || "blend",
      accent: true,
    },
    {
      label: "best single",
      value: blend.best_single ? blend.best_single.score : null,
      note: blend.best_single ? blend.best_single.name : "—",
    },
    {
      label: "models",
      value: blend.models ? blend.models.length : (data.metrics.models || []).length,
      note: blend.fallback_used ? "fallback blend" : "weighted blend",
      neutral: true,
    },
  ];
  for (const card of cards) {
    const node = el("div", "metric-card");
    node.append(el("small", "", card.label));
    const strong = el("strong", card.neutral ? "neutral" : "", card.value === null || card.value === undefined ? "—" : (card.neutral ? String(card.value) : fmtScore(card.value)));
    node.append(strong);
    node.append(el("span", "", card.note || "—"));
    container.append(node);
  }
}

function renderBars(data) {
  const container = $("#bars");
  container.replaceChildren();
  const blend = data.metrics && data.metrics.blend;
  if (!blend) return;
  let rows = (data.metrics.models || []).filter((row) => row.oof_score !== null && row.oof_score !== undefined);
  if (!rows.length && blend.model_scores) {
    rows = Object.entries(blend.model_scores).map(([name, score]) => ({ name, oof_score: score }));
  }
  if (!rows.length) {
    container.append(el("div", "hint", "Нет OOF-метрик для отображения."));
    return;
  }
  const direction = data.metrics.direction === "minimize" ? "minimize" : "maximize";
  const values = rows.map((row) => Number(row.oof_score)).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  rows.sort((a, b) => direction === "maximize" ? Number(b.oof_score) - Number(a.oof_score) : Number(a.oof_score) - Number(b.oof_score));
  rows.forEach((row, index) => {
    const score = Number(row.oof_score);
    let ratio = 1;
    if (Number.isFinite(score) && max !== min) {
      ratio = direction === "maximize" ? (score - min) / (max - min) : (max - score) / (max - min);
      ratio = Math.max(0.04, Math.min(1, ratio));
    }
    const line = el("div", `bar-row ${index === 0 ? "best" : ""}`);
    line.append(el("span", "bar-name", row.name));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = `${Math.round(ratio * 100)}%`;
    track.append(fill);
    line.append(track, el("span", "bar-score", fmtScore(score)));
    container.append(line);
  });
}

function renderSubmission(data) {
  const table = $("#submission-preview");
  const head = table.querySelector("thead");
  const body = table.querySelector("tbody");
  head.replaceChildren();
  body.replaceChildren();
  const preview = data.submission_preview;
  if (!preview || !preview.columns || !preview.columns.length) {
    const emptyHead = document.createElement("tr");
    emptyHead.append(el("th", "", "—"));
    head.append(emptyHead);
    const emptyBody = document.createElement("tr");
    emptyBody.append(el("td", "empty-cell", "Появится после стадии submit."));
    body.append(emptyBody);
    return;
  }
  const row = document.createElement("tr");
  for (const column of preview.columns) row.append(el("th", "", column));
  head.append(row);
  for (const record of preview.rows || []) {
    const tr = document.createElement("tr");
    for (const column of preview.columns) {
      const value = record[column];
      tr.append(el("td", "", value === null || value === undefined ? "∅" : value));
    }
    body.append(tr);
  }
}

function renderResults(data) {
  const hasMetrics = Boolean(data.metrics && data.metrics.blend);
  const hasSubmission = Boolean(data.has_submission);
  const hasReport = Boolean(data.report);
  const hasAny = hasMetrics || hasSubmission || hasReport;
  $("#results-empty").classList.toggle("hidden", hasAny);
  $("#results-body").classList.toggle("hidden", !hasAny);
  $("#results-state").textContent = hasMetrics ? "готово" : data.stale ? "stale" : data.server_active ? "обучение" : "нет запуска";
  $("#results-state").className = `card-state ${hasMetrics ? "ok" : ""}`;

  const reportNode = $("#report");
  if (data.report_html) {
    // The server renders a strict Markdown subset and escapes every text
    // fragment; raw HTML from report.md is never passed through.
    reportNode.innerHTML = data.report_html;
  } else if (data.report) {
    reportNode.textContent = data.report;
  } else {
    reportNode.replaceChildren(el("p", "report-empty", "Отчёт появится после завершения полного прогона."));
  }

  if (!hasAny) return;

  renderMetricCards(data);
  renderBars(data);
  renderSubmission(data);

  const blendNote = $("#blend-note");
  const blend = data.metrics && data.metrics.blend;
  if (blend) {
    const weights = blend.weights && Object.keys(blend.weights).length
      ? Object.entries(blend.weights).map(([name, weight]) => `${name}=${Number(weight).toFixed(3)}`).join(", ")
      : "равные веса / single model";
    blendNote.textContent = `Blend: ${blend.method || "—"} · ${weights}${blend.fallback_used ? " · fallback" : ""}`;
    blendNote.classList.remove("hidden");
  } else {
    blendNote.classList.add("hidden");
  }

  const project = encodeURIComponent(data.project || projectName());
  const sub = $("#dl-sub");
  const report = $("#dl-report");
  sub.classList.toggle("hidden", !hasSubmission);
  report.classList.toggle("hidden", !hasReport);
  if (hasSubmission) sub.href = `/api/submission?project=${project}`;
  if (hasReport) report.href = `/api/report?project=${project}`;
}

function renderRunMeta(data) {
  const meta = $("#run-meta");
  meta.replaceChildren();
  const items = [];
  if (data.project) items.push(`project ${data.project}`);
  if (data.run_id) items.push(`run ${data.run_id}`);
  if (data.until) items.push(`until ${data.until}`);
  if (data.elapsed_sec !== null && data.elapsed_sec !== undefined) items.push(`time ${fmtDuration(data.elapsed_sec)}`);
  if (data.returncode !== null && data.returncode !== undefined) items.push(`exit ${data.returncode}`);
  if (data.disk_status && data.disk_status !== data.status) items.push(`disk ${data.disk_status}`);
  for (const item of items) meta.append(el("span", "", item));
  if (!items.length) meta.append(el("span", "", "Запуск ещё не создан."));
}

function renderStatus(data) {
  state.currentStatus = data.status;
  state.lastStatus = data;
  state.batchActive = Boolean(data.batch_active);
  state.serverActive = Boolean(data.server_active);
  state.serverBusy = Boolean(data.server_busy);
  state.stale = Boolean(data.stale);
  const nextHasSubmission = Boolean(data.has_submission);
  const nextRunId = data.run_id || null;
  if (nextHasSubmission !== state.kaggle.hasSubmission || nextRunId !== state.kaggle.runId) {
    state.kaggle.hasSubmission = nextHasSubmission;
    state.kaggle.runId = nextRunId;
    if (!nextHasSubmission) state.kaggle.submission = null;
    renderKaggle();
  }
  renderQueue(data.queue || []);
  setStatusBadge($("#run-status-badge"), data.stale ? "stale" : data.status);
  renderRunMeta(data);
  renderStages(data.stages);
  renderLog(data);
  renderResults(data);
  renderFailure(data).catch(() => {});
  updateRunEnabled();
  if (data.error) {
    $("#run-hint").textContent = data.error;
  } else if (data.stale) {
    $("#run-hint").textContent = "Run на диске в статусе running, но этот сервер его не запускал. Вероятно, он прерван — можно запустить новый.";
  }
}

async function pollStatus(schedule = true) {
  clearTimeout(state.pollTimer);
  const project = projectName();
  state.currentProject = project;
  try {
    const data = await api(`/api/run/status?project=${encodeURIComponent(project)}`);
    renderStatus(data);
    if (schedule && (data.server_active || data.server_busy || data.batch_active || hasPendingQueue())) {
      state.pollTimer = setTimeout(() => pollStatus(true), 1200);
    } else if (schedule) {
      state.pollTimer = null;
      refreshComparisonIfNeeded(data);
      refreshStats().catch(() => {});
    }
  } catch (error) {
    if (schedule) toast(error.message, true);
  }
}

async function startRun() {
  const error = runConfigError();
  if (error) {
    toast(error, true);
    return;
  }
  const payload = buildRunPayload();
  localStorage.setItem("kconv.project", payload.project);
  state.currentProject = payload.project;
  state.lastLog = "";
  try {
    const started = await postJson("/api/run", payload);
    state.currentStatus = started.status;
    state.serverActive = true;
    state.serverBusy = true;
    $("#run-hint").textContent = "Запуск создан. Первые строки лога появятся через несколько секунд.";
    state.followLog = true;
    $("#log-wrap").open = true;
    setStatusBadge($("#run-status-badge"), started.status);
    updateRunEnabled();
    pollStatus(true);
  } catch (error) {
    toast(error.hint ? `${error.message} ${error.hint}` : error.message, true);
  }
}

async function stopRun() {
  try {
    await api("/api/run/stop", { method: "POST" });
    toast("Останавливаю процесс.");
    pollStatus(true);
  } catch (error) {
    toast(error.message, true);
  }
}

function bindDropzone(kind, inputSelector, zoneSelector) {
  const input = $(inputSelector);
  const zone = $(zoneSelector);
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) uploadFile(kind, input.files[0]);
    input.value = "";
  });
  for (const eventName of ["dragenter", "dragover"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragging");
    });
  }
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) uploadFile(kind, file);
  });
}

function uploadFile(kind, file) {
  if (state.uploadBusy) {
    toast("Подождите, предыдущий файл ещё загружается.");
    return;
  }
  if (!/\.(csv|csv\.gz|parquet|pq)$/i.test(file.name)) {
    toast("Поддерживаются .csv, .csv.gz, .parquet и .pq.", true);
    return;
  }
  if (file.size > 256 * 1024 ** 2) {
    toast("Файл больше 256 MiB.", true);
    return;
  }
  if (!state.csrfToken) {
    toast("CSRF-токен не загружен. Обновите страницу.", true);
    return;
  }
  state.uploadBusy = true;
  updateRunEnabled();
  const status = $("#upload-status");
  status.textContent = `Загрузка ${file.name} (${fmtBytes(file.size)})…`;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/upload?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.setRequestHeader("X-Kconv-Token", state.csrfToken);
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      status.textContent = `Загрузка ${file.name}: ${Math.round((event.loaded / event.total) * 100)}%`;
    }
  };
  xhr.onload = async () => {
    state.uploadBusy = false;
    let payload = null;
    try { payload = JSON.parse(xhr.responseText); } catch (_) { payload = null; }
    if (xhr.status < 200 || xhr.status >= 300) {
      status.textContent = "Ошибка загрузки.";
      updateRunEnabled();
      toast(payload && payload.message ? payload.message : "Не удалось загрузить файл.", true);
      return;
    }
    status.textContent = `${payload.file ? payload.file.name : file.name} загружен. Обновляю датасет…`;
    try {
      state.manualTarget = false;
      state.llm.plan = null;
      state.llm.message = "Датасет обновлён — LLM-план нужно сгенерировать заново.";
      await refreshDataset();
      await refreshStats();
      state.comparisonKey = "";
      $("#comparison-dataset").value = "";
      refreshComparison().catch(() => {});
      status.textContent = "Датасет обновлён. Проверьте target и запускайте.";
    } catch (error) {
      status.textContent = "Файл загружен, но датасет не прочитался.";
      toast(error.message, true);
    }
    updateRunEnabled();
  };
  xhr.onerror = () => {
    state.uploadBusy = false;
    status.textContent = "Сетевая ошибка загрузки.";
    updateRunEnabled();
    toast("Сетевая ошибка загрузки.", true);
  };
  xhr.send(file);
}

function debounceDetect() {
  clearTimeout(state.detectTimer);
  state.detectTimer = setTimeout(async () => {
    const target = $("#target").value;
    if (!target) return;
    const params = new URLSearchParams({ target });
    if ($("#id-col").value) params.set("id", $("#id-col").value);
    if ($("#task").value !== "auto") params.set("task", $("#task").value);
    if ($("#metric").value !== "auto") params.set("metric", $("#metric").value);
    try {
      const data = await api(`/api/dataset?${params.toString()}`);
      state.dataset = data;
      renderFacts(data);
      if (data.detected) {
        if ($("#task").value === "auto") $("#task").value = data.detected.task;
        if ($("#metric").value === "auto") $("#metric").value = data.detected.metric;
        if (!$("#id-col").value && data.detected.id_col) $("#id-col").value = data.detected.id_col;
      }
      syncTaskOptions(false);
      renderTaskNote();
      refreshModels();
    } catch (error) {
      toast(error.message, true);
    }
  }, 280);
}

function bindEvents() {
  $("#refresh").addEventListener("click", async () => {
    try {
      await Promise.all([refreshDataset(), refreshStats()]);
      await pollStatus(false);
      await refreshKaggleStatus(true);
      toast("Обновлено.");
    } catch (error) {
      toast(error.message, true);
    }
  });
  $("#kaggle-save").addEventListener("click", saveKaggleCredentials);
  $("#kaggle-submit").addEventListener("click", submitKaggle);
  $("#kaggle-refresh").addEventListener("click", () => refreshKaggleStatus(false));
  $("#kaggle-competition").addEventListener("input", () => {
    localStorage.setItem("kconv.kaggle.competition", kaggleCompetition());
    updateKaggleEnabled();
  });
  $("#run").addEventListener("click", startRun);
  $("#stop").addEventListener("click", stopRun);
  $("#variant-add").addEventListener("click", addVariant);
  $("#variant-add-pipelines").addEventListener("click", addSelectedPipelines);
  $("#recipe-batch").addEventListener("change", () => {
    syncPipelinePreset();
    updateRunEnabled();
  });
  $("#pipeline-preset").addEventListener("change", (event) => {
    applyPipelinePreset(event.target.value);
  });
  $("#pipeline-select-all").addEventListener("click", () => {
    const select = $("#recipe-batch");
    if (!select) return;
    for (const option of select.options) option.selected = true;
    syncPipelinePreset();
    updateRunEnabled();
  });
  $("#pipeline-select-none").addEventListener("click", () => {
    const select = $("#recipe-batch");
    if (!select) return;
    for (const option of select.options) option.selected = false;
    syncPipelinePreset();
    updateRunEnabled();
  });
  $("#variant-start").addEventListener("click", startBatch);
  $("#variant-clear").addEventListener("click", () => {
    state.variants = [];
    renderVariants();
    toast("Список вариантов очищен.");
  });
  $("#variants").addEventListener("click", (event) => {
    const button = event.target.closest(".variant-remove");
    if (!button) return;
    state.variants = state.variants.filter((variant) => variant.id !== button.dataset.variantId);
    renderVariants();
  });
  $("#variants").addEventListener("input", (event) => {
    const input = event.target.closest(".variant-label");
    if (!input) return;
    const row = input.closest(".variant-row");
    const variant = state.variants.find((item) => item.id === row.dataset.variantId);
    if (variant) variant.label = input.value;
  });
  $("#comparison-refresh").addEventListener("click", () => {
    state.comparisonKey = "";
    refreshComparison().then(() => toast("Сравнение обновлено.")).catch((error) => toast(error.message, true));
  });
  $("#comparison-dataset").addEventListener("change", () => {
    state.comparisonKey = "";
    refreshComparison().catch((error) => toast(error.message, true));
  });
  $("#queue-filter").addEventListener("change", (event) => {
    state.queueFilter = event.target.value || "all";
    renderQueue(state.queue);
  });
  $("#queue-list").addEventListener("click", (event) => {
    const button = event.target.closest(".queue-retry");
    if (!button) return;
    retryQueueItem(button.dataset.itemId);
  });
  $("#failure-continue").addEventListener("click", (event) => {
    const button = event.currentTarget;
    continueRun(button.dataset.runId, button.dataset.until);
  });
  $("#log-level").addEventListener("change", (event) => {
    state.logLevel = event.target.value;
    state.lastLog = "";
    renderLog(state.lastStatus || {});
  });
  $("#log-query").addEventListener("input", (event) => {
    state.logQuery = event.target.value;
    state.lastLog = "";
    renderLog(state.lastStatus || {});
  });
  $("#log-follow").addEventListener("change", (event) => {
    state.followLog = event.target.checked;
  });
  $("#profile").addEventListener("change", updateProfileState);
  $("#recipe").addEventListener("change", () => {
    renderRecipeCard();
    applyRecipeModels({ announce: true });
  });
  $("#llm-enabled").addEventListener("change", () => {
    if (!llmEnabled()) state.llm.message = "";
    renderLlmState();
  });
  $("#llm-provider").addEventListener("change", () => {
    syncLlmProvider();
    resetLlmPlan("Провайдер изменён — сгенерируйте план заново.");
    if ($("#llm-provider").value === "local") discoverLocalServer(true);
  });
  $("#llm-max-features").addEventListener("input", () => {
    if (state.llm.plan) resetLlmPlan("Настройки LLM изменены — сгенерируйте план заново.");
  });
  $("#llm-max-tokens").addEventListener("input", () => {
    if (state.llm.plan) resetLlmPlan("Max tokens изменён — сгенерируйте план заново.");
  });
  $("#llm-local-discover").addEventListener("click", () => discoverLocalServer(false));
  $("#llm-model").addEventListener("input", () => {
    if (state.llm.plan) resetLlmPlan("Модель изменена — сгенерируйте план заново.");
  });
  $("#llm-base-url").addEventListener("input", () => {
    if (state.llm.plan) resetLlmPlan("Base URL изменён — сгенерируйте план заново.");
  });
  $("#llm-json-mode").addEventListener("change", () => {
    if (state.llm.plan) resetLlmPlan("JSON mode изменён — сгенерируйте план заново.");
    renderLlmState();
  });
  $("#llm-generate").addEventListener("click", generateLlmPlan);
  $("#llm-approved").addEventListener("change", updateRunEnabled);
  $("#task").addEventListener("change", () => {
    syncTaskOptions(true);
    renderTaskNote();
    refreshModels();
    if ($("#target").value) debounceDetect();
  });
  $("#target").addEventListener("change", () => {
    state.manualTarget = true;
    if (state.llm.plan) resetLlmPlan("Target изменён — сгенерируйте LLM-план заново.");
    debounceDetect();
    updateRunEnabled();
  });
  $("#id-col").addEventListener("change", () => {
    if (state.llm.plan) resetLlmPlan("ID-колонка изменена — сгенерируйте LLM-план заново.");
    debounceDetect();
  });
  $("#metric").addEventListener("change", debounceDetect);
  $("#model-search").addEventListener("input", refreshModels);
  $("#model-params-rows").addEventListener("input", (event) => {
    const input = event.target.closest(".model-params-input");
    if (!input) return;
    state.modelParams.set(input.dataset.model, input.value);
  });
  $("#models").addEventListener("change", (event) => {
    const checkbox = event.target.closest("input[type=checkbox]");
    if (!checkbox) return;
    const label = checkbox.closest(".model-item");
    if (checkbox.checked) {
      state.selected.add(checkbox.value);
      label.classList.add("selected");
    } else {
      state.selected.delete(checkbox.value);
      label.classList.remove("selected");
    }
    updateModelCount();
    renderSelectedChips();
    renderModelParams();
    updateRunEnabled();
  });
  $("#project").addEventListener("input", () => {
    localStorage.setItem("kconv.project", projectName());
    state.currentProject = projectName();
    updateRunEnabled();
    clearTimeout(state.projectTimer);
    state.projectTimer = setTimeout(() => {
      state.comparisonKey = "";
      $("#comparison-dataset").value = "";
      pollStatus(false);
      refreshComparison().catch(() => {});
    }, 500);
  });
  $("#log").addEventListener("scroll", () => {
    const pre = $("#log");
    state.followLog = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 28;
  });
  $("#until-full").addEventListener("change", () => {
    state.fullRunTouched = true;
    updateRunEnabled();
  });

  bindDropzone("train", "#file-train", "#drop-train");
  bindDropzone("test", "#file-test", "#drop-test");
  bindDropzone("sample", "#file-sample", "#drop-sample");
}

async function init() {
  bindEvents();
  renderVariants();
  try {
    await loadSession();
  } catch (error) {
    const detail = error.hint ? `${error.message} ${error.hint}` : error.message;
    setConfigError(`Не удалось получить CSRF-токен сессии: ${detail} Перезапустите сервер (start.cmd) и обновите страницу.`);
    toast("Не удалось получить CSRF-токен сессии.", true);
    $("#run-hint").textContent = "Сервер не подтвердил сессию — действия недоступны.";
    return;
  }
  try {
    await loadConfig();
    await Promise.all([refreshDataset(), refreshStats()]);
    await restoreLlmPlan();
    const savedCompetition = localStorage.getItem("kconv.kaggle.competition");
    if (savedCompetition && !$("#kaggle-competition").value) {
      $("#kaggle-competition").value = savedCompetition;
    }
    await pollStatus(false);
    await refreshKaggleStatus(true);
    refreshComparison().catch(() => {});
    if (state.serverActive || state.serverBusy || state.batchActive || hasPendingQueue()) pollStatus(true);
  } catch (error) {
    const detail = error.hint ? `${error.message} ${error.hint}` : error.message;
    setConfigError(`Не удалось загрузить конфигурацию: ${detail} Перезапустите сервер (start.cmd), чтобы загрузить актуальный код.`);
    toast(detail, true);
    $("#run-hint").textContent = error.message;
  }
}

document.addEventListener("DOMContentLoaded", init);
