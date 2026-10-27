"use strict";

// ---------- tiny helpers ----------
const $ = (sel) => document.querySelector(sel);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.setAttribute("style", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const pct = (x, d = 1) => (x == null ? "–" : `${(100 * x).toFixed(d)}%`);
const pp = (x, d = 1) => {
  const v = Number((100 * x).toFixed(d));
  return `${v > 0 ? "+" : v < 0 ? "" : "±"}${(v === 0 ? 0 : v).toFixed(d)} pp`;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const COND_COLORS = {
  none: "--off", reflexion: "--muted", raw_memory: "--warn", verified: "--accent",
  verified_rerank: "--accent-2", shuffled: "--bad",
};
const condColor = (k) => css(COND_COLORS[k] || "--muted");

const state = { runs: [], run: null, rows: null, charts: {}, tasks: [] };

// ---------- theme ----------
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  try { t ? localStorage.setItem("mf-theme", t) : localStorage.removeItem("mf-theme"); } catch (_) {}
  if (state.run) renderAll();
}
(() => {
  let saved = null;
  try { saved = localStorage.getItem("mf-theme"); } catch (_) {}
  if (saved) document.documentElement.dataset.theme = saved;
})();
$("#theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(dark ? "light" : "dark");
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.run && renderAll());

// ---------- routing ----------
const TABS = ["overview", "experiments", "lessons", "patterns", "explorer", "live"];
function route() {
  const tab = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  for (const t of TABS) $("#" + t).hidden = t !== tab;
  document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-selected", a.getAttribute("href") === "#" + tab));
  const noRun = !state.run && tab !== "live";
  $("#empty").hidden = !noRun;
  if (noRun) $("#" + tab).hidden = true;
  if (tab === "lessons" && state.run) loadLessons();
  if (tab === "explorer" && state.run) loadExplorer();
}
window.addEventListener("hashchange", route);

// ---------- Chart.js defaults ----------
const errorBars = {
  id: "errorBars",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    chart.data.datasets.forEach((ds, i) => {
      if (!ds.errors) return;
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden) return;
      const horizontal = chart.options.indexAxis === "y";
      const scale = horizontal ? chart.scales.x : chart.scales.y;
      ctx.save();
      ctx.strokeStyle = css("--text");
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1.5;
      meta.data.forEach((bar, j) => {
        const e = ds.errors[j];
        if (!e) return;
        const a = scale.getPixelForValue(e[0]), b = scale.getPixelForValue(e[1]);
        ctx.beginPath();
        if (horizontal) {
          ctx.moveTo(a, bar.y); ctx.lineTo(b, bar.y);
          ctx.moveTo(a, bar.y - 4); ctx.lineTo(a, bar.y + 4);
          ctx.moveTo(b, bar.y - 4); ctx.lineTo(b, bar.y + 4);
        } else {
          ctx.moveTo(bar.x, a); ctx.lineTo(bar.x, b);
          ctx.moveTo(bar.x - 5, a); ctx.lineTo(bar.x + 5, a);
          ctx.moveTo(bar.x - 5, b); ctx.lineTo(bar.x + 5, b);
        }
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};
function chart(id, config) {
  if (!window.Chart) return;
  if (state.charts[id]) state.charts[id].destroy();
  Chart.defaults.font.family = css("--font").replace(/"/g, "") || "Inter";
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--border");
  config.plugins = [...(config.plugins || []), errorBars];
  state.charts[id] = new Chart(document.getElementById(id), config);
}
const pctAxis = { min: 0, max: 1, ticks: { callback: (v) => `${Math.round(v * 100)}%` }, grid: { color: () => css("--border") } };

// ---------- data loading ----------
async function loadRuns() {
  state.runs = await api("/api/runs").catch(() => []);
  const sel = $("#run-select");
  sel.replaceChildren(...state.runs.map((r) =>
    h("option", { value: r.run_id }, `${r.run_id} · ${r.simulated ? "sim" : r.backend} · ${r.seeds.length} seeds`)));
  if (!state.runs.length) { state.run = null; route(); return; }
  const want = new URLSearchParams(location.search).get("run");
  await selectRun(state.runs.some((r) => r.run_id === want) ? want : state.runs[0].run_id);
}
async function selectRun(id) {
  $("#run-select").value = id;
  state.run = await api(`/api/runs/${id}`);
  state.rows = null;
  renderAll();
  route();
}
$("#run-select").addEventListener("change", (e) => selectRun(e.target.value));

function renderAll() {
  renderOverview();
  renderExperiments();
  renderPatterns();
  setupLessonFilters();
  setupExplorer();
}

// ---------- overview ----------
function renderOverview() {
  const r = state.run, s = r.summary, m = r.meta;
  const off = s.none, on = s.verified || s[m.conditions.at(-1).key];
  const lift = r.contrasts.find((c) => c.a === "verified" && c.b === "none" && c.metric === "pass1");
  $("#ov-eyebrow").textContent = `Held-out evaluation · ${m.n_eval} unseen tasks × ${m.seeds.length} seeds`;
  $("#ov-sub").textContent = `${m.n_train} experience tasks build the memory, which is then frozen. Up to ${m.max_trials} trials per task, top-${m.top_k} lessons, ${m.embedder} embeddings, ${m.store} store.`;
  const badges = [h("span", { class: "badge " + (m.simulated ? "sim" : "real") }, m.simulated ? "⚠ simulated model" : "● " + m.backend)];
  if (m.git_sha) badges.push(h("span", { class: "badge" }, "commit " + m.git_sha));
  if (m.duration_s) badges.push(h("span", { class: "badge" }, `${m.duration_s}s`));
  $("#ov-badges").replaceChildren(...badges);
  if (m.simulated) {
    $("#ov-sub").append(h("br"), h("span", {}, "Simulated runs validate the pipeline: lesson effects are simulator parameters, not measurements of a real model."));
  }

  $("#k-off").textContent = off ? pct(off.pass1.mean) : "–";
  $("#k-off-ci").textContent = off ? `95% CI ${pct(off.pass1.ci95[0])} – ${pct(off.pass1.ci95[1])}` : "";
  $("#k-on").textContent = on ? pct(on.pass1.mean) : "–";
  $("#k-on-ci").textContent = on ? `95% CI ${pct(on.pass1.ci95[0])} – ${pct(on.pass1.ci95[1])}` : "";
  if (lift) {
    const el = $("#k-lift");
    el.textContent = pp(lift.diff);
    el.className = "v " + (lift.ci95[0] > 0 ? "pos" : lift.ci95[1] < 0 ? "neg" : "");
    $("#k-lift-ci").textContent = `CI [${pp(lift.ci95[0])}, ${pp(lift.ci95[1])}] · p=${lift.mcnemar.p.toFixed(3)}`;
    const lo = -0.5, hi = 0.5, x = (v) => `${(100 * (Math.min(hi, Math.max(lo, v)) - lo)) / (hi - lo)}%`;
    const bar = $("#k-lift-bar");
    bar.replaceChildren(
      h("span", { class: "span", style: `left:${x(lift.ci95[0])};right:calc(100% - ${x(lift.ci95[1])})` }),
      h("span", { class: "zero", style: `left:${x(0)}` }));
  }
  const mem = r.memory.lessons;
  const raw = mem.raw_memory, ver = mem.verified;
  if (ver) {
    const caught = (ver.harmful_total || 0) - (ver.harmful_live || 0);
    $("#k-harm").textContent = m.simulated ? `${caught}/${ver.harmful_total || 0}` : `${ver.by_status.rejected || 0}`;
    $("#k-harm-sub").textContent = m.simulated
      ? `raw memory kept ${raw ? raw.harmful_live : "–"} retrievable`
      : `lessons rejected by replay`;
  }

  const conds = m.conditions;
  chart("ch-ov-bars", {
    type: "bar",
    data: {
      labels: conds.map((c) => c.label),
      datasets: [{
        label: "pass@1", data: conds.map((c) => s[c.key].pass1.mean),
        errors: conds.map((c) => s[c.key].pass1.ci95),
        backgroundColor: conds.map((c) => condColor(c.key)), borderRadius: 8, maxBarThickness: 46,
      }],
    },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: (c) => `pass@1 ${pct(c.raw)}  CI ${pct(c.dataset.errors[c.dataIndex][0])}–${pct(c.dataset.errors[c.dataIndex][1])}` } } },
      scales: { y: pctAxis, x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 30 } } } },
  });
  renderCurve("ch-ov-curve");
}

function renderCurve(id) {
  const lc = state.run.learning_curve, labels = Object.fromEntries(state.run.meta.conditions.map((c) => [c.key, c.label]));
  const keys = Object.keys(lc);
  if (!keys.length) return;
  const x = lc[keys[0]].x;
  const datasets = [];
  for (const k of keys) {
    const col = condColor(k);
    datasets.push({ label: labels[k] || k, data: lc[k].y, borderColor: col, backgroundColor: col, tension: .3, pointRadius: 0, borderWidth: 2.5,
      borderDash: k === "none" ? [6, 4] : undefined });
    datasets.push({ label: "_band", data: lc[k].hi, borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: col + "22" });
    datasets.push({ label: "_band", data: lc[k].lo, borderWidth: 0, pointRadius: 0, fill: false });
  }
  chart(id, {
    type: "line", data: { labels: x, datasets },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { filter: (i) => i.text !== "_band", usePointStyle: true, boxWidth: 8 } },
        tooltip: { filter: (i) => i.dataset.label !== "_band", callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.raw)}` } } },
      scales: { y: pctAxis, x: { title: { display: true, text: "task # in experience stream" }, grid: { display: false } } } },
  });
}

// ---------- experiments ----------
function renderExperiments() {
  const r = state.run, m = r.meta, s = r.summary;
  $("#ex-meta").textContent = `${m.seeds.length} seeds · ${m.n_eval} tasks · K=${m.max_trials}`;
  $("#ex-solved-h").textContent = `solved@${m.max_trials}`;
  $("#ex-table tbody").replaceChildren(...m.conditions.map((c) => {
    const x = s[c.key];
    return h("tr", { class: c.key === "verified" ? "hl" : "" },
      h("td", {}, h("span", { style: `color:${condColor(c.key)}` }, "● "), c.label, h("span", { class: "desc" }, c.description)),
      h("td", {}, pct(x.pass1.mean)),
      h("td", {}, `${pct(x.pass1.ci95[0])} – ${pct(x.pass1.ci95[1])}`),
      h("td", {}, pct(x.solved.mean)),
      h("td", {}, x.mean_trials_censored ? x.mean_trials_censored.toFixed(2) : "–"),
      h("td", {}, Math.round(x.tokens_per_task)));
  }));
  chart("ch-ex-bars", {
    type: "bar",
    data: {
      labels: m.conditions.map((c) => c.label),
      datasets: [
        { label: "pass@1", data: m.conditions.map((c) => s[c.key].pass1.mean), errors: m.conditions.map((c) => s[c.key].pass1.ci95),
          backgroundColor: css("--accent"), borderRadius: 6 },
        { label: `solved@${m.max_trials}`, data: m.conditions.map((c) => s[c.key].solved.mean), errors: m.conditions.map((c) => s[c.key].solved.ci95),
          backgroundColor: css("--accent-2") + "aa", borderRadius: 6 },
      ],
    },
    options: { maintainAspectRatio: false, scales: { y: pctAxis, x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 30 } } },
      plugins: { legend: { labels: { usePointStyle: true, boxWidth: 8 } } } },
  });

  // forest plot of contrasts
  const cs = r.contrasts.filter((c) => c.metric === "pass1");
  const lim = Math.max(0.25, ...cs.map((c) => Math.max(Math.abs(c.ci95[0]), Math.abs(c.ci95[1])))) * 1.1;
  const X = (v) => `${((v + lim) / (2 * lim)) * 100}%`;
  const label = Object.fromEntries(m.conditions.map((c) => [c.key, c.label]));
  $("#forest").replaceChildren(
    ...cs.map((c) => h("div", { class: "f-row" },
      h("div", {}, h("b", {}, label[c.a] || c.a), h("span", { class: "desc" }, "vs " + (label[c.b] || c.b))),
      h("div", { class: "f-track", title: `CI [${pp(c.ci95[0])}, ${pp(c.ci95[1])}]  McNemar b=${c.mcnemar.b} c=${c.mcnemar.c} p=${c.mcnemar.p.toFixed(3)}` },
        h("span", { class: "f-zero", style: `left:${X(0)}` }),
        h("span", { class: "f-ci" + (c.ci95[0] > 0 || c.ci95[1] < 0 ? " sig" : ""), style: `left:${X(c.ci95[0])};width:calc(${X(c.ci95[1])} - ${X(c.ci95[0])})` }),
        h("span", { class: "f-pt", style: `left:${X(c.diff)}` })),
      h("div", { class: "f-val " + (c.ci95[0] > 0 ? "pos" : c.ci95[1] < 0 ? "neg" : "") }, pp(c.diff)))),
    h("p", { class: "hint" }, `Axis ±${Math.round(lim * 100)} pp. Highlighted intervals exclude zero. Hover a row for McNemar counts.`));

  // memory composition
  const colors = { verified: css("--ok"), candidate: css("--warn"), rejected: css("--bad"), deprecated: css("--off"), merged: css("--muted") };
  const mem = r.memory.lessons;
  $("#ex-memory").replaceChildren(...Object.entries(mem).map(([k, cell]) => {
    const total = Object.values(cell.by_status).reduce((a, b) => a + b, 0) || 1;
    return h("div", { class: "mem-row" },
      h("div", {}, h("b", {}, label[k] || k), h("span", { class: "muted small" }, `  ·  ${cell.per_seed} lessons per seed · ${pct(cell.on_target / Math.max(1, cell.total), 0)} target a real pitfall of their source task`)),
      h("div", { class: "stack" }, ...Object.entries(cell.by_status).map(([st, n]) =>
        h("span", { style: `width:${(100 * n) / total}%;background:${colors[st] || css("--muted")}`, title: `${st}: ${n}` }))),
      h("div", { class: "legend" }, ...Object.entries(cell.by_status).map(([st, n]) => h("span", {}, h("i", { style: `background:${colors[st]}` }), `${st} ${n}`)),
        m.simulated ? h("span", {}, `harmful retrievable: ${cell.harmful_live} of ${cell.harmful_total}`) : null));
  }), h("p", { class: "muted small" }, "Retrieval precision on eval (retrieved lesson targets a pitfall the task really has): ",
    ...Object.entries(r.memory.retrieval_precision).map(([k, v]) => h("span", { class: "pat", style: "margin-right:6px" }, `${label[k] || k} ${pct(v, 0)}`))));

  const rr = r.reranker || [];
  $("#ex-rerank").replaceChildren(rr.length
    ? h("table", { class: "data" }, h("thead", {}, h("tr", {}, ...["seed", "events", "model AUC", "formula AUC", "blend"].map((t) => h("th", {}, t)))),
        h("tbody", {}, ...rr.map((x) => h("tr", {}, h("td", {}, x.seed), h("td", {}, x.n_events),
          h("td", {}, x.trained ? x.val_auc.toFixed(3) : "not trained"), h("td", {}, x.trained ? x.formula_auc.toFixed(3) : "–"),
          h("td", {}, x.trained ? x.blend_weight.toFixed(2) : "–")))))
    : h("p", { class: "muted" }, "The reranker condition was not part of this run."));
}

// ---------- patterns ----------
const VOCAB = {
  empty_input: "Empty or degenerate input (empty list or string, single element, zero).",
  off_by_one: "Boundary shifted by one: inclusive vs exclusive ends, loop bounds, slice ends.",
  tie_ordering: "Ties or equal keys come out in the wrong (unstable) order.",
  whitespace_case: "Parsing breaks on extra whitespace, tabs, case or unicode.",
  negative_numbers: "Wrong result for negative numbers or zero.",
  mutates_input: "Modifies an argument the caller still owns.",
  shared_reference: "Result aliases mutable objects from the input.",
  float_rounding: "Float arithmetic or rounding mode gives a slightly wrong number.",
  inefficient: "Correct but too slow on large inputs.",
  error_contract: "Wrong exception, or no exception where the spec demands one.",
};
function renderPatterns() {
  const r = state.run, bp = r.by_pattern, pats = Object.keys(bp);
  const show = ["none", "reflexion", "raw_memory", "verified", "verified_rerank"].filter((k) => r.summary[k]);
  const label = Object.fromEntries(r.meta.conditions.map((c) => [c.key, c.label]));
  chart("ch-patterns", {
    type: "bar",
    data: { labels: pats.map((p) => `${p} (${bp[p].n_tasks})`),
      datasets: show.map((k) => ({ label: label[k], data: pats.map((p) => bp[p][k]), backgroundColor: condColor(k), borderRadius: 4 })) },
    options: { indexAxis: "y", maintainAspectRatio: false, scales: { x: pctAxis, y: { grid: { display: false } } },
      plugins: { legend: { labels: { usePointStyle: true, boxWidth: 8 } }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${pct(c.raw)}` } } } },
  });
  const clf = r.classifier;
  $("#clf-acc").textContent = clf.n ? `accuracy ${pct(clf.accuracy)} on ${clf.n} failures with known ground truth` : "no ground truth (real model)";
  if (clf.n) {
    const truth = Object.keys(clf.confusion).sort();
    const predicted = [...new Set(Object.values(clf.confusion).flatMap((o) => Object.keys(o)))].sort();
    const max = Math.max(...Object.values(clf.confusion).flatMap((o) => Object.values(o)));
    const acc = css("--accent");
    $("#confusion").replaceChildren(h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "true ↓ / predicted →"), ...predicted.map((p) => h("th", { class: "rot" }, p)))),
      h("tbody", {}, ...truth.map((t) => h("tr", {}, h("th", { style: "text-align:right" }, t), ...predicted.map((p) => {
        const n = clf.confusion[t][p] || 0;
        const a = n ? 0.15 + 0.85 * (n / max) : 0;
        return h("td", { title: `${t} → ${p}: ${n}`, style: `background:${n ? acc : "transparent"};opacity:${n ? 1 : 1};color:${a > .55 ? css("--accent-ink") : css("--text")};background:${n ? `color-mix(in srgb, ${acc} ${Math.round(a * 100)}%, transparent)` : css("--surface-2")}` }, n || "");
      }))))));
  } else $("#confusion").replaceChildren(h("p", { class: "muted" }, "Classifier accuracy needs ground-truth bug labels, which only the simulator provides."));
  $("#vocab").replaceChildren(...Object.entries(VOCAB).map(([k, v]) => h("li", {}, h("span", { class: "pat" }, k), h("span", { class: "muted" }, v))));
}

// ---------- lessons ----------
const STATUSES = ["all", "verified", "candidate", "rejected", "deprecated"];
let lessonStatus = "all", lessonTimer = null;
function setupLessonFilters() {
  const seeds = state.run.meta.seeds;
  $("#l-seed").replaceChildren(h("option", { value: "" }, "all seeds"), ...seeds.map((s) => h("option", { value: s }, `seed ${s}`)));
  $("#l-memory").replaceChildren(...["verified", "raw_memory"].filter((k) => state.run.memory.lessons[k])
    .map((k) => h("option", { value: k }, k === "verified" ? "Verified memory" : "Raw memory")));
  $("#l-status").replaceChildren(...STATUSES.map((st) => h("button", { class: "chip", type: "button", "aria-pressed": st === lessonStatus,
    onclick: () => { lessonStatus = st; setupLessonFilters(); loadLessons(); } }, st)));
}
["#l-q", "#l-memory", "#l-seed"].forEach((sel) => $(sel).addEventListener("input", () => {
  clearTimeout(lessonTimer); lessonTimer = setTimeout(loadLessons, 200);
}));
async function loadLessons() {
  const p = new URLSearchParams({ memory: $("#l-memory").value || "verified" });
  if ($("#l-seed").value !== "") p.set("seed", $("#l-seed").value);
  if (lessonStatus !== "all") p.set("status", lessonStatus);
  if ($("#l-q").value) p.set("q", $("#l-q").value);
  $("#l-list").replaceChildren(...[1, 2, 3].map(() => h("div", { class: "skeleton", style: "height:70px" })));
  const res = await api(`/api/runs/${state.run.meta.run_id}/lessons?${p}`).catch(() => ({ items: [], total: 0, counts: {} }));
  $("#l-count").textContent = `${res.total} lessons · ` + Object.entries(res.counts).map(([k, v]) => `${v} ${k}`).join(" · ");
  if (!res.items.length) { $("#l-list").replaceChildren(h("p", { class: "muted" }, "No lessons match these filters.")); return; }
  const items = res.items.sort((a, b) => b.uses - a.uses || a.lesson_id - b.lesson_id);
  $("#l-list").replaceChildren(...items.map((l) => {
    const u = l.utility, rp = l.replay || {};
    const canvas = h("canvas", { class: "spark", width: 240, height: 56, "aria-label": "lift over time" });
    const el = h("article", { class: `lesson ${l.status}` },
      h("p", {}, l.text),
      h("div", { class: "stats" },
        h("div", { class: u.lift_mean > 0 ? "pos" : u.lift_mean < 0 ? "neg" : "" }, `lift ${pp(u.lift_mean)}`),
        h("div", { class: "muted" }, `80% [${pp(u.lift_lo, 0)}, ${pp(u.lift_hi, 0)}]`), canvas),
      h("div", { class: "meta" }, h("span", { class: "pat" }, l.pattern), h("span", { class: "status " + l.status }, l.status),
        h("span", {}, `used ${l.uses}×`), h("span", {}, `from ${l.source_task}`), h("span", {}, `seed ${l.seed}`),
        rp.n ? h("span", { title: "counterfactual replay: b = improved with lesson, c = worse with lesson" }, `replay b=${rp.b} c=${rp.c} over ${rp.n}`) : null,
        h("span", {}, `P(lift>0) ${pct(u.p_positive, 0)}`)));
    requestAnimationFrame(() => spark(canvas, (l.history || []).map((x) => x[1])));
    return el;
  }));
}
function spark(canvas, ys) {
  const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = css("--border"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  if (ys.length < 2) return;
  const lim = Math.max(0.2, ...ys.map(Math.abs));
  ctx.strokeStyle = css("--accent"); ctx.lineWidth = 3; ctx.beginPath();
  ys.forEach((y, i) => { const x = (i / (ys.length - 1)) * W, py = H / 2 - (y / lim) * (H / 2 - 4); i ? ctx.lineTo(x, py) : ctx.moveTo(x, py); });
  ctx.stroke();
}

// ---------- explorer ----------
async function setupExplorer() {
  if (!state.tasks.length) state.tasks = await api("/api/tasks");
  const m = state.run.meta;
  const phase = $("#x-phase").value;
  const tasks = state.tasks.filter((t) => t.split === (phase === "eval" ? "eval" : "train"));
  const keep = $("#x-task").value;
  $("#x-task").replaceChildren(...tasks.map((t) => h("option", { value: t.id }, t.id)));
  if (tasks.some((t) => t.id === keep)) $("#x-task").value = keep;
  const condKeep = $("#x-cond").value;
  const conds = phase === "eval" ? m.conditions : m.conditions.filter((c) => !["verified_rerank", "shuffled"].includes(c.key));
  $("#x-cond").replaceChildren(...conds.map((c) => h("option", { value: c.key }, c.label)));
  $("#x-cond").value = conds.some((c) => c.key === condKeep) ? condKeep : (conds.find((c) => c.key === "verified") || conds[0]).key;
  const seedKeep = $("#x-seed").value;
  $("#x-seed").replaceChildren(...m.seeds.map((s) => h("option", { value: s }, s)));
  if (seedKeep) $("#x-seed").value = seedKeep;
}
["#x-task", "#x-cond", "#x-seed"].forEach((s) => $(s).addEventListener("change", loadExplorer));
$("#x-phase").addEventListener("change", async () => { await setupExplorer(); loadExplorer(); });
async function loadExplorer() {
  if (!$("#x-task").value) await setupExplorer();
  const p = new URLSearchParams({ task_id: $("#x-task").value, condition: $("#x-cond").value, seed: $("#x-seed").value || 0, phase: $("#x-phase").value });
  const res = await api(`/api/runs/${state.run.meta.run_id}/attempts?${p}`).catch(() => null);
  if (!res) return;
  const t = res.task;
  $("#x-task-card").replaceChildren(
    h("div", { class: "card-h" }, h("h3", {}, t.id), h("span", { class: "hint" }, `${t.split} · ${t.n_tests} hidden tests · pitfalls: ${t.pitfalls.join(", ")}`)),
    h("pre", { class: "code" }, t.signature), h("p", {}, t.spec));
  if (!res.trials.length) { $("#x-timeline").replaceChildren(h("li", { class: "muted" }, "No attempts recorded for this combination.")); return; }
  $("#x-timeline").replaceChildren(...res.trials.map((tr) => h("li", { class: "trial" },
    h("span", { class: "dot " + (tr.passed ? "pass" : "fail") }, tr.trial),
    h("div", { class: "card" },
      h("h4", {}, `Trial ${tr.trial}`, h("span", { class: "status " + (tr.passed ? "verified" : "rejected") }, tr.passed ? "passed" : "failed"),
        h("span", { class: "muted small" }, `${tr.n_passed}/${tr.n_tests} tests · ${tr.tokens} tokens`),
        tr.pattern ? h("span", { class: "pat" }, `diagnosed: ${tr.pattern}${tr.pattern_conf ? ` (${Math.round(tr.pattern_conf * 100)}%)` : ""}`) : null,
        tr.true_pattern ? h("span", { class: "pat", title: "ground truth from the simulator" }, `true: ${tr.true_pattern}`) : null),
      tr.retrieved.length ? h("div", {}, h("div", { class: "muted small" }, "Retrieved lessons"), h("div", { class: "ret" }, ...tr.retrieved.map((l) =>
        h("div", { class: "ret-item" + (l.withheld ? " withheld" : ""), title: l.withheld ? "withheld from the prompt (randomised control arm)" : "" },
          h("span", { class: "pat" }, l.pattern), h("span", {}, l.text + (l.withheld ? "  (withheld)" : "")),
          h("div", { class: "scorebar" }, h("span", { style: `width:${Math.min(100, l.score * 100)}%` })))))) : null,
      tr.passed ? null : h("pre", { class: "code" }, tr.feedback || ""),
      tr.reflection ? h("blockquote", {}, tr.reflection) : null,
      tr.diff ? h("details", { open: true }, h("summary", {}, "Change from previous trial"), diffView(tr.diff))
        : h("details", {}, h("summary", {}, "Code"), h("pre", { class: "code" }, tr.code))))));
}
function diffView(text) {
  return h("pre", { class: "code diff" }, ...text.split("\n").map((line) => h("span", {
    class: line.startsWith("+") && !line.startsWith("+++") ? "add" : line.startsWith("-") && !line.startsWith("---") ? "del" : line.startsWith("@@") ? "hunk" : "",
  }, line + "\n")));
}

// ---------- live ----------
const form = $("#live-form");
form.harmful_rate.addEventListener("input", (e) => ($("#hr-out").textContent = e.target.value));
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(form));
  for (const k of Object.keys(body)) body[k] = Number(body[k]);
  $("#live-start").disabled = true;
  $("#live-open").hidden = true;
  $("#live-log").textContent = "";
  try {
    const { job_id } = await api("/api/live", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    poll(job_id);
  } catch (err) {
    $("#live-state").textContent = String(err.message || err);
    $("#live-start").disabled = false;
  }
});
async function poll(id) {
  const j = await api(`/api/live/${id}`).catch(() => null);
  if (!j) return setTimeout(() => poll(id), 1500);
  $("#live-state").textContent = j.status === "running" ? `running · ${j.done}/${j.total || "?"}` : j.status;
  $("#live-bar").style.width = j.total ? `${(100 * j.done) / j.total}%` : "2%";
  $("#live-done").textContent = j.done;
  $("#live-rate").textContent = j.attempted ? pct(j.passed / j.attempted, 0) : "–";
  const log = $("#live-log");
  log.textContent = j.log.join("\n");
  log.scrollTop = log.scrollHeight;
  if (j.status === "running") return setTimeout(() => poll(id), 1000);
  $("#live-start").disabled = false;
  if (j.status === "done") {
    await loadRuns();
    await selectRun(j.run_id);
    $("#live-open").hidden = false;
  } else if (j.error) log.textContent += `\n\nerror: ${j.error}`;
}

loadRuns().then(route);
