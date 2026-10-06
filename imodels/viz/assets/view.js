/* Additive-model pages: rule sets, scorecards, shape functions (GAMs) and linear models. */
(function () {
  "use strict";
  const S = DTI.init(), D = S.D, T = D.terms, esc = S.esc, fmt = S.fmt;
  const svg = document.getElementById("canvas"), NS = "http://www.w3.org/2000/svg";
  const OPS = {"<=": (a, b) => a <= b, "<": (a, b) => a < b, ">": (a, b) => a > b, ">=": (a, b) => a >= b, "==": (a, b) => a === b,
    "!=": (a, b) => a !== b, "isnan": (a) => Number.isNaN(a), "notnan": (a) => !Number.isNaN(a)};
  const NEG = {"<=": ">", "<": ">=", ">": "<=", ">=": "<", "==": "!=", "!=": "==", "isnan": "notnan", "notnan": "isnan"};

  // ---------- model evaluation (mirrors AdditiveView.score in Python)
  function interp(v, g, y) {  // np.interp: linear between points, flat beyond the ends
    if (v <= g[0]) return y[0];
    if (v >= g[g.length - 1]) return y[y.length - 1];
    let i = 1;
    while (g[i] < v) i++;
    return y[i - 1] + (y[i] - y[i - 1]) * (v - g[i - 1]) / (g[i] - g[i - 1]);
  }
  function contrib(t, x) {
    if (t.k === "rule") return t.c.every(([f, op, v]) => OPS[op](x[f], v)) ? t.w : 0;
    const v = x[t.f];
    if (t.k === "linear") return t.w * ((t.clip ? Math.min(t.clip[1], Math.max(t.clip[0], v)) : v) - t.ctr);
    if (t.k === "curve") {
      if (!t.tied || t.tied.every(Boolean) || !t.tied.some(Boolean)) return interp(v, t.g, t.vals);
      let b = 0;  // bin of v: searchsorted(edges, v, "right")
      while (b < t.e.length && t.e[b] <= v) b++;
      if (t.tied[b]) return interp(v, t.g, t.vals);
      const keep = t.tied.map((q) => !q);
      return interp(v, t.g.filter((_, i) => keep[i]), t.vals.filter((_, i) => keep[i]));
    }
    const e = t.e.map((q, i) => (q == null ? (i === 0 ? -Infinity : Infinity) : q));
    let i = 0;  // bins are (e[i], e[i+1]], matching tree splits
    while (i < t.vals.length - 1 && v > e[i + 1]) i++;
    return t.vals[i];
  }
  function output(s) {
    if (D.link === "logistic") return 1 / (1 + Math.exp(-(D.linkScale * s + D.linkOffset)));
    if (D.link === "clip") return Math.min(1, Math.max(0, s));
    if (D.link === "exp") return Math.exp(s);
    if (D.link === "threshold") return s > D.threshold ? 1 : 0;
    return s;
  }
  function predict(x) {
    const parts = T.map((t) => contrib(t, x));
    const s = D.intercept + parts.reduce((a, c) => a + c, 0), out = output(s);
    if (!S.isClf) return {value: out, parts, s};
    const k = out >= 0.5 ? D.classes.length - 1 : 0;
    if (D.link === "threshold") {
      const fired = T.filter((t, i) => t.k === "rule" && parts[i] !== 0).length;
      return {cls: k, parts, s, note: `${fired} rule${fired === 1 ? "" : "s"} hold${fired === 1 ? "s" : ""}`};
    }
    const note = D.scoreName.startsWith("P(") ? "" : `${esc(D.scoreName)} ${fmt(s)}`;
    return {cls: k, proba: [1 - out, out], parts, s, note};
  }
  function explain(x, p) {
    const items = p.parts.map((c, i) => [T[i].txt, c]).filter(([, c]) => Math.abs(c) > 1e-12)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    if (D.intercept) items.unshift(["baseline", D.intercept]);
    return S.waterfall(items, D.scoreName, p.s, D.pos, D.neg) + (D.note ? `<div class="tt-m" style="font-size:11px;margin-top:6px">${esc(D.note)}</div>` : "");
  }

  // ---------- rows (rule sets / scorecards) and panels (GAMs)
  const rows = [...svg.querySelectorAll(".term")], panels = [...svg.querySelectorAll(".panel")];
  const condHolds = (c, x) => OPS[c[1]](x[c[0]], c[2]);
  // smallest input change that makes condition (f, op, v) hold
  function satisfy(f, op, v) {
    const ft = S.feat(f), span = (ft.hi - ft.lo) || 1, eps = span * 1e-6;
    if (op === "isnan") return NaN;
    if (op === "notnan") return ft.v;
    if (ft.levels && op === "!=") return v === 0 ? 1 : 0;  // any other level
    if (ft.kind !== "continuous") {
      return {"<=": Math.floor(v), "<": Math.ceil(v) - 1, ">": Math.floor(v) + 1, ">=": Math.ceil(v), "==": v,
        "!=": ft.kind === "binary" ? 1 - v : v + 1}[op];
    }
    return {"<=": v, "<": v - eps, ">": v + eps, ">=": v, "==": v, "!=": v + span * 0.01}[op];
  }
  function toggleRule(t) {
    const x = S.x();
    if (t.c.every((c) => condHolds(c, x))) {  // break it with the smallest single change
      let best = null;
      for (const [f, op, v] of t.c) {
        const nv = satisfy(f, NEG[op], v);
        const d = Number.isNaN(nv) || Number.isNaN(x[f]) || S.feat(f).levels ? 0.5 : Math.abs(nv - x[f]) / ((S.feat(f).hi - S.feat(f).lo) || 1);
        if (!best || d < best.d) best = {f, nv, d};
      }
      if (best) S.setValues({[best.f]: best.nv});
    } else {  // make every failing condition hold
      const vals = {};
      for (const c of t.c) if (!condHolds(c, Object.assign({}, x, vals))) vals[c[0]] = satisfy(c[0], c[1], c[2]);
      S.setValues(vals);
    }
  }
  rows.forEach((g) => {
    const t = T[+g.dataset.t];
    const act = t.k !== "rule" ? "" : D.samples.length ? "Click to load a training row this rule covers (while predicting, click to switch the rule on or off)."
      : "While predicting, click to switch the rule on or off.";
    g.addEventListener("pointerenter", () => { S.tip(t.tip + (act ? `<div class="tt-act">${act}</div>` : "")); S.hotFeatures(t.fs); });
    g.addEventListener("pointerleave", () => { S.hideTip(); S.hotFeatures([]); });
    if (t.k === "rule") g.addEventListener("click", () => {
      if (S.predicting()) toggleRule(t);
      else if (!S.loadSample((x) => t.c.every((c) => condHolds(c, x)), `training row covered by rule ${g.dataset.rank || ""}`.trim())) S.open(true);
    });
  });
  // GAM panels: click to set the feature to the clicked value
  panels.forEach((g) => {
    const m = (D.chart || []).find((c) => c.f === +g.dataset.f);
    g.addEventListener("pointerenter", () => { S.tip(`<div class="tt-h"><b>${esc(S.feat(m.f).name)}</b></div><div class="tt-act">Click to set ${esc(S.feat(m.f).name)} to the value under the pointer.</div>`); S.hotFeatures([m.f]); });
    g.addEventListener("pointerleave", () => { S.hideTip(); S.hotFeatures([]); });
    g.addEventListener("click", (e) => {
      const pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const local = pt.matrixTransform(g.getScreenCTM().inverse());
      const fr = Math.max(0, Math.min(1, (local.x - m.x0) / (m.x1 - m.x0)));
      let v = m.lo + fr * (m.hi - m.lo);
      if (S.feat(m.f).kind !== "continuous") v = Math.round(v);
      S.setValues({[m.f]: v});
    });
  });

  // ---------- sort control (rule sets only)
  const sortSel = document.getElementById("sort"), sortCtl = document.getElementById("sortctl");
  if (D.family !== "ruleset" || rows.length < 2) sortCtl.style.display = "none";
  sortSel.onchange = () => {
    const key = sortSel.value;
    const order = rows.slice().sort((a, b) => {
      const ta = T[+a.dataset.t], tb = T[+b.dataset.t];
      if (key === "coverage") return (tb.sup ?? -1) - (ta.sup ?? -1);
      if (key === "model") return +a.dataset.t - +b.dataset.t;
      return +a.dataset.rank - +b.dataset.rank;
    });
    let y = D.rowsTop;
    for (const g of order) { g.style.transform = `translate(0px, ${y}px)`; y += +g.dataset.h; }
  };

  // ---------- markers
  // label: [text, x, y, anchor] placed clear of chart titles
  function marker(g, x, y0, y1, yDot, label) {
    let mk = g.querySelector(":scope > .mk");
    if (!mk) { mk = document.createElementNS(NS, "g"); mk.setAttribute("class", "mk"); g.appendChild(mk); }
    mk.innerHTML = x == null ? "" : `<line x1="${x}" x2="${x}" y1="${y0}" y2="${y1}" style="stroke:var(--dti-hl);stroke-width:2"/>` +
      (yDot != null ? `<circle cx="${x}" cy="${yDot}" r="4.5" style="fill:var(--dti-hl);stroke:var(--dti-card);stroke-width:1.5"/>` : "") +
      (label ? `<text x="${label[1]}" y="${label[2]}" text-anchor="${label[3]}" style="font-size:10.5px;font-weight:700;fill:var(--dti-hl);paint-order:stroke;stroke:var(--dti-card);stroke-width:3px">${label[0]}</text>` : "");
  }
  function show(x, p) {
    svg.classList.toggle("predicting", !!x);
    if (!x) {
      svg.querySelectorAll(".mk").forEach((m) => (m.innerHTML = ""));
      svg.querySelectorAll(".rk").forEach((d) => d.setAttribute("r", 4.5));
      return;
    }
    rows.forEach((g) => {
      const i = +g.dataset.t, t = T[i];
      if (t.k !== "rule") { g.classList.add("on"); return; }
      const ok = t.c.map((c) => condHolds(c, x)), n = ok.filter(Boolean).length;
      g.classList.toggle("on", n === ok.length);
      g.classList.toggle("near", n === ok.length - 1 && ok.length > 1);
      g.querySelectorAll(".chip").forEach((ch) => {
        const j = +ch.dataset.c;
        ch.classList.toggle("ok", !!ok[j]);
        ch.classList.toggle("no", !ok[j]);
      });
    });
    if (D.family === "gam" && D.chart) for (const m of D.chart) {
      const g = svg.querySelector(`.panel[data-f="${m.f}"]`);
      if (Number.isNaN(x[m.f])) { marker(g, null); continue; }
      const v = Math.max(m.lo, Math.min(m.hi, x[m.f])), px = m.x0 + (v - m.lo) / (m.hi - m.lo) * (m.x1 - m.x0);
      const c = T.filter((t) => t.f === m.f).reduce((a, t) => a + contrib(t, x), 0);
      const py = m.y0 + (m.y1 - m.y0) / 2 - c / m.ymax * (m.y1 - m.y0) / 2;
      const right = px > (m.x0 + m.x1) / 2;
      marker(g, px.toFixed(1), m.y0, m.y1, py.toFixed(1), [S.signed(c), (px + (right ? -8 : 8)).toFixed(1), (py - 8).toFixed(1), right ? "end" : "start"]);
    }
    if (D.family === "scorecard" && D.chart) {
      const c = D.chart, v = Math.max(c.smin, Math.min(c.smax, p.s)), px = c.x0 + (v - c.smin) / (c.smax - c.smin || 1) * (c.x1 - c.x0);
      marker(svg, px.toFixed(1), c.y0, c.y1, null, [`score ${fmt(p.s)}`, (px + 6).toFixed(1), (c.y0 + 14).toFixed(1), "start"]);
      svg.querySelectorAll(".rk").forEach((d) => d.setAttribute("r", Math.abs(+d.dataset.s - p.s) < 1e-9 ? 7 : 4.5));
    }
  }

  S.start({
    word: D.family === "gam" ? "shape function" : D.family === "scorecard" ? "item" : "term",
    hint: D.family === "gam" ? "Set feature values, or click inside a panel, to see each feature's contribution. They add up to the prediction."
      : D.family === "scorecard" ? "Set feature values to see which items score. Click an item to switch it on or off."
        : "Set feature values to see which rules hold and how the terms add up. Click a rule to switch it on or off.",
    predict, explain, show,
    highlight(f) {
      svg.classList.toggle("feat-mode", f != null);
      rows.forEach((g) => g.classList.toggle("feat", f != null && T[+g.dataset.t].fs.includes(f)));
      panels.forEach((g) => g.classList.toggle("feat", f != null && +g.dataset.f === f));
    },
    exportSvg() {
      const clone = svg.cloneNode(true);
      clone.removeAttribute("id");
      clone.querySelectorAll(".mk").forEach((m) => m.remove());
      const text = new XMLSerializer().serializeToString(clone);
      return text.replace(/(<svg[^>]*>)/, `$1<rect x="-4" y="-4" width="100%" height="100%" fill="${S.surface()}"/>`);
    },
  });
})();
