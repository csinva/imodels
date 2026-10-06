/* Shared runtime for imodels.viz interactive pages.
 * A model script calls DTI.init() for utilities, draws its view, then calls S.start(hooks):
 *   predict(x)        -> {cls, proba} for classifiers or {value} for regressors (pure)
 *   explain(x, pred)  -> HTML shown under the prediction (path, waterfall, ...)
 *   show(x, pred)     -> update the drawing for input x (x = null clears it)
 *   highlight(f)      -> emphasize what uses feature f (null clears)
 *   exportSvg()       -> SVG text of the current view
 *   onOpen / onClose  -> optional layout hooks when the Predict panel opens / closes
 */
window.DTI = (function () {
  "use strict";

  function init() {
    const D = JSON.parse(document.getElementById("dti-data").textContent);
    const S = {D};
    const $ = (id) => document.getElementById(id);
    const sig = D.sig || 3;

    // ---------- formatting
    S.esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
    S.fmt = (v, digits) => {
      if (v == null || !isFinite(v)) return String(v);
      if (Math.abs(v) < 1e-12) return "0";
      const g = digits || sig, a = Math.abs(v);
      if (a >= 1e6 || a < 1e-3) return v.toExponential(g - 1).replace("e+", "e").replace("-", "−");
      const d = Math.max(g - Math.floor(Math.log10(a)) - 1, 0);
      return Number(v.toFixed(d)).toLocaleString("en-US", {maximumFractionDigits: d}).replace("-", "−");
    };
    S.signed = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + S.fmt(Math.abs(v));
    S.pct = (p) => (p >= 0.995 && p < 1 ? ">99" : p > 0 && p <= 0.005 ? "<1" : (100 * p).toFixed(0)) + "%";
    const isClf = D.task === "classification";
    S.isClf = isClf;
    S.classDot = (k) => `<i style="background:${D.classes[k].col}"></i>${S.esc(D.classes[k].name)}`;

    // ---------- legend (same look on every page)
    $("legend").innerHTML = (D.legend || []).map((it) => {
      if (it.k === "dot") return `<span><i style="background:${it.col}"></i>${S.esc(it.label)}</span>`;
      if (it.k === "grad") return `<span>${S.esc(it.label)}</span><span class="tt-m">${S.fmt(it.lo)}</span>` +
        `<span class="grad" style="background:linear-gradient(90deg,${it.stops.join(",")})"></span><span class="tt-m">${S.fmt(it.hi)}</span>`;
      return `<span class="lnote">${S.esc(it.label)}</span>`;
    }).join("");

    // ---------- tooltip
    const tip = $("tip");
    let tipOn = false;
    S.tip = (html) => { tip.innerHTML = html; tip.classList.add("show"); tipOn = true; };
    S.hideTip = () => { tip.classList.remove("show"); tipOn = false; };
    addEventListener("pointermove", (e) => {
      if (!tipOn) return;
      let x = e.clientX + 16, y = e.clientY + 16;
      if (x + tip.offsetWidth > innerWidth - 8) x = e.clientX - tip.offsetWidth - 16;
      if (y + tip.offsetHeight > innerHeight - 8) y = Math.max(8, innerHeight - tip.offsetHeight - 8);
      tip.style.left = x + "px"; tip.style.top = y + "px";
    });

    // ---------- theme
    const root = document.documentElement;
    $("b-theme").onclick = () => {
      const dark = root.dataset.theme === "dark" || (root.dataset.theme === "auto" && matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
    };
    S.resolveVars = (text) => {
      const cs = getComputedStyle(root);
      return text.replace(/var\((--dti-[a-z0-9]+)\)/g, (_, v) => cs.getPropertyValue(v).trim());
    };
    S.surface = () => getComputedStyle(root).getPropertyValue("--dti-surface").trim();

    // ---------- features column
    const hud = $("hud"), hudBody = $("hud-body");
    const FW = 264, FH = 26;
    let hoverF = null, pinned = null, H = {};
    function sparkline(f) {
      const sx = (v) => Math.max(0, Math.min(1, (v - f.hlo) / (f.hhi - f.hlo || 1))) * FW;
      let out = `<line x1="0" x2="${FW}" y1="${FH + 0.5}" y2="${FH + 0.5}" style="stroke:var(--dti-axis)"/>`;
      if (f.hist) {
        const nb = f.hist.length, bw = FW / nb, top = Math.max(1, ...f.hist.map((b) => b.reduce((a, c) => a + c, 0)));
        const w = Math.min(bw - 1, 22);
        f.hist.forEach((b, i) => {
          let y = FH;
          b.forEach((c, k) => {
            if (!c) return;
            const h = (c / top) * (FH - 2);
            y -= h;
            const col = isClf && b.length > 1 ? D.classes[k].col : "var(--dti-s4)";
            out += `<rect x="${(i * bw + (bw - w) / 2).toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" style="fill:${col}"/>`;
          });
        });
      }
      for (const t of f.thr) out += `<line x1="${sx(t).toFixed(1)}" x2="${sx(t).toFixed(1)}" y1="${FH + 2}" y2="${FH + 8}" style="stroke:var(--dti-ink);stroke-width:1.5"/>`;
      out += `<line class="xv" y1="-1" y2="${FH}" style="stroke:var(--dti-hl);stroke-width:2"/>`;
      return `<svg viewBox="0 -2 ${FW} ${FH + 12}" preserveAspectRatio="none" aria-hidden="true">${out}</svg>`;
    }
    function buildHud(word) {
      const tot = D.feats.reduce((a, f) => a + f.imp, 0) || 1, maxImp = Math.max(...D.feats.map((f) => f.imp), 1e-12);
      $("hud-sub").textContent = `${D.feats.length} used · sorted by importance`;
      $("hud-note").innerHTML = D.impNote || "";
      $("hud-f").textContent = `Hover a feature to highlight the ${word}s that use it. Click to pin.`;
      hudBody.innerHTML = D.feats.map((f) => {
        const ends = f.ends || [S.fmt(f.hlo), S.fmt(f.hhi)], n = f.nsplit;
        return `<div class="frow" data-f="${f.i}">
          <div class="top"><span class="nm" title="${S.esc(f.name)}">${S.esc(f.name)}</span><span class="im">${(100 * f.imp / tot).toFixed(1)}%</span></div>
          <div class="ib"><span style="width:${(100 * f.imp / maxImp).toFixed(1)}%"></span></div>
          ${sparkline(f)}
          <div class="meta"><span>${S.esc(ends[0])}</span><span>${n} ${word}${n === 1 ? "" : "s"}</span><span>${S.esc(ends[1])}</span></div></div>`;
      }).join("");
      hudBody.querySelectorAll(".frow").forEach((row) => {
        const f = +row.dataset.f;
        row.addEventListener("pointerenter", () => { hoverF = f; applyFeat(); });
        row.addEventListener("pointerleave", () => { hoverF = null; applyFeat(); });
        row.addEventListener("click", () => { pinned = pinned === f ? null : f; applyFeat(); });
      });
      $("hud-x").onclick = () => hud.classList.toggle("min");
      if (innerWidth < 1100) hud.classList.add("min");  // keep the model in view on smaller screens
    }
    function applyFeat() {
      const f = hoverF ?? pinned;
      if (H.highlight) H.highlight(f);
      hudBody.querySelectorAll(".frow").forEach((r) => r.classList.toggle("pin", +r.dataset.f === pinned));
    }
    S.reapplyHighlight = applyFeat;
    S.hotFeatures = (fs) => hudBody.querySelectorAll(".frow").forEach((r) => r.classList.toggle("hot", fs.includes(+r.dataset.f)));

    // ---------- predict panel
    const panel = $("panel"), fields = $("fields"), res = $("res"), wi = $("whatif"), predBtn = $("b-pred");
    let x = null, truth = null, truthLabel = "";
    const featByIdx = {};
    D.feats.forEach((f) => (featByIdx[f.i] = f));
    S.feat = (i) => featByIdx[i];
    // a feature value as read by people: level name, "missing", or the number
    S.val = (i, v, digits) => {
      const f = featByIdx[i];
      if (v == null || Number.isNaN(v)) return "missing";
      if (f && f.levels) return f.levels[Math.round(v)] ?? S.fmt(v);
      return S.fmt(v, digits);
    };
    const isMissing = (v) => v == null || Number.isNaN(v);
    // samples store missing values as null
    const fromSample = (sx) => { const o = {}; for (const k in sx) o[k] = sx[k] == null ? NaN : sx[k]; return o; };
    S.x = () => x;
    S.predicting = () => !!x;
    function defaults() { const o = {}; D.feats.forEach((f) => (o[f.i] = f.v)); return o; }
    function buildFields() {
      fields.innerHTML = D.feats.map((f) => {
        const nm = S.esc(f.name);
        if (f.levels) {  // categorical: pick a level (or missing)
          return `<div class="fld" data-f="${f.i}"><label><span title="${nm}">${nm}</span>
            <select aria-label="${nm}">${f.levels.map((l, k) => `<option value="${k}">${S.esc(l)}</option>`).join("")}` +
            `${f.missing ? `<option value="nan">(missing)</option>` : ""}</select></label></div>`;
        }
        return `<div class="fld" data-f="${f.i}"><label><span title="${nm}">${nm}</span><span class="ctl2">` +
          (f.missing ? `<label class="miss" title="Treat ${nm} as missing"><input type="checkbox"> missing</label>` : "") +
          `<input type="number" step="${f.step}" value="${+(+f.v).toPrecision(6)}" aria-label="${nm}"></span></label>
          <input type="range" min="${f.lo}" max="${f.hi}" step="${f.step}" value="${f.v}" aria-label="${nm} slider"></div>`;
      }).join("");
      fields.querySelectorAll(".fld").forEach((el) => {
        const f = +el.dataset.f, ft = featByIdx[f];
        const num = el.querySelector("[type=number]"), rng = el.querySelector("[type=range]");
        const sel = el.querySelector("select"), miss = el.querySelector(".miss input");
        const changed = () => { truth = null; refresh(); };
        if (sel) sel.addEventListener("change", () => { x[f] = sel.value === "nan" ? NaN : +sel.value; changed(); });
        if (num) num.addEventListener("input", () => {
          if (num.value === "") return;
          x[f] = +num.value; rng.value = num.value; if (miss) miss.checked = false; changed();
        });
        if (rng) rng.addEventListener("input", () => {
          x[f] = +rng.value; num.value = +(+rng.value).toPrecision(6); if (miss) miss.checked = false; changed();
        });
        if (miss) miss.addEventListener("change", () => { x[f] = miss.checked ? NaN : +num.value || ft.v; changed(); });
        el.addEventListener("pointerenter", () => { hoverF = f; applyFeat(); });
        el.addEventListener("pointerleave", () => { hoverF = null; applyFeat(); });
      });
    }
    function setFields() {
      fields.querySelectorAll(".fld").forEach((el) => {
        const f = +el.dataset.f, v = x[f];
        const num = el.querySelector("[type=number]"), rng = el.querySelector("[type=range]");
        const sel = el.querySelector("select"), miss = el.querySelector(".miss input");
        if (sel) { sel.value = isMissing(v) ? "nan" : String(Math.round(v)); return; }
        if (miss) miss.checked = isMissing(v);
        if (isMissing(v)) { num.value = ""; return; }
        num.value = +(+v).toPrecision(6);
        rng.value = v;
      });
    }
    function resultHtml(p) {
      let html = `<div class="tt-k">Prediction</div>`;
      if (isClf) {
        const k = p.cls;
        html += `<div class="big">${S.classDot(k)}</div>`;
        if (p.proba) {
          const shown = D.classes.length === 2 ? 1 : k;
          html += `<div class="meta">P(${S.esc(D.classes[shown].name)}) = ${S.pct(p.proba[shown])}${p.note ? " \u00b7 " + p.note : ""}</div>`;
          html += `<div class="probs">${p.proba.map((q, j) => q > 0 ? `<span style="width:${(100 * q).toFixed(2)}%;background:${D.classes[j].col}" title="${S.esc(D.classes[j].name)} ${S.pct(q)}"></span>` : "").join("")}</div>`;
          html += `<div class="plabs">${p.proba.map((q, j) => q > 0.005 ? `<span><i style="background:${D.classes[j].col}"></i>${S.esc(D.classes[j].name)} ${S.pct(q)}</span>` : "").join("")}</div>`;
        } else if (p.note) html += `<div class="meta">${p.note}</div>`;
      } else {
        html += `<div class="big">${S.fmt(p.value, Math.max(sig, 4))}</div>`;
        if (p.note) html += `<div class="meta">${p.note}</div>`;
      }
      if (truth != null) html += `<div class="truth">${S.esc(truthLabel)} <b>${S.esc(truth)}</b></div>`;
      return html + (H.explain ? H.explain(x, p) : "");
    }
    function refresh() {
      const p = H.predict(x);
      res.innerHTML = resultHtml(p);
      if (H.show) H.show(x, p);
      for (const f of D.feats) {
        const ln = hudBody.querySelector(`.frow[data-f="${f.i}"] .xv`);
        if (!ln) continue;
        ln.style.visibility = isMissing(x[f.i]) ? "hidden" : "";
        const px = (Math.max(0, Math.min(1, (x[f.i] - f.hlo) / (f.hhi - f.hlo || 1))) * FW).toFixed(1);
        ln.setAttribute("x1", px); ln.setAttribute("x2", px);
      }
      hud.classList.add("predicting");
      whatIf(p);
      applyFeat();
    }
    S.refresh = () => { if (x) refresh(); };

    // ---------- what would change the prediction
    // the roundest number just past threshold t (above when up, else below) that crosses no other
    // threshold, so an applied change reads cleanly (1.8 rather than 1.750001)
    function nicePast(t, up, span, thr) {
      const tol = span * 1e-3;
      for (let p = 1; p <= 12; p++) {
        const step = Math.pow(10, Math.floor(Math.log10(Math.abs(t) || 1)) - p + 1);
        const v = +((up ? Math.floor(t / step) + 1 : Math.ceil(t / step) - 1) * step).toPrecision(p + 1);
        const clear = thr.every((o) => !(up ? o > t && o < v : o < t && o > v));
        if ((up ? v > t : v < t) && Math.abs(v - t) <= tol && clear) return v;
      }
      return up ? t + span * 1e-6 : t - span * 1e-6;
    }
    function candidates(f, cur) {
      const span = (f.hi - f.lo) || 1, out = [];
      if (f.missing && !isMissing(cur)) out.push({v: NaN, txt: "missing"});
      if (f.levels) {
        f.levels.forEach((l, k) => { if (isMissing(cur) || k !== Math.round(cur)) out.push({v: k, txt: l}); });
        return out;
      }
      if (f.kind === "binary") return out.concat([{v: 0, txt: "0"}, {v: 1, txt: "1"}].filter((c) => c.v !== cur));
      for (const t of f.thr) {
        if (f.kind === "integer") {
          out.push({v: Math.floor(t), txt: S.fmt(Math.floor(t), 12)}, {v: Math.floor(t) + 1, txt: S.fmt(Math.floor(t) + 1, 12)});
        } else {
          let dg = sig;  // enough digits that the threshold reads differently from the current value
          // (only when they really differ: a value sitting on the cut keeps the short form)
          while (dg < 8 && !isMissing(cur) && Math.abs(t - cur) > span * 1e-5 && S.fmt(t, dg) === S.fmt(cur, dg)) dg++;
          out.push({v: nicePast(t, false, span, f.thr), txt: `\u2264 ${S.fmt(t, dg)}`, t},
                   {v: nicePast(t, true, span, f.thr), txt: `> ${S.fmt(t, dg)}`, t});
        }
      }
      for (let q = 0; q <= 8; q++) {
        let v = f.lo + span * q / 8;
        if (f.kind === "integer") v = Math.round(v);
        out.push({v, txt: S.fmt(v)});
      }
      return out.filter((c) => isMissing(c.v) || isMissing(cur) || Math.abs(c.v - cur) > 1e-12);
    }
    // how big a change is, as a share of the feature's range (switching a level or to / from missing counts as half)
    function dist(f, c, cur) {
      if (f.levels || isMissing(c.v) || isMissing(cur)) return 0.5;
      return Math.abs(c.v - cur) / ((f.hi - f.lo) || 1);
    }
    function whatIf(base) {
      const rows = [];
      for (const f of D.feats) {
        const cur = x[f.i];
        let best = null;
        for (const c of candidates(f, cur)) {
          const xx = Object.assign({}, x, {[f.i]: c.v});
          const p = H.predict(xx), d = dist(f, c, cur);
          if (isClf) {
            const flips = p.cls !== base.cls;
            // prefer the smallest change that flips the class; otherwise the biggest drop in the current class
            const drop = p.proba && base.proba ? base.proba[base.cls] - p.proba[base.cls] : 0;
            const key = flips ? [0, d] : [1, -drop];
            if (!flips && drop < 0.005) continue;  // ignore changes that barely move the probability
            if (!best || key[0] < best.key[0] || (key[0] === best.key[0] && key[1] < best.key[1])) best = {f, c, p, key, flips, drop};
          } else {
            const delta = p.value - base.value;
            if (Math.abs(delta) <= 1e-9 * (1 + Math.abs(base.value))) continue;
            const key = [0, -Math.abs(delta)];
            if (!best || key[1] < best.key[1]) best = {f, c, p, key, delta};
          }
        }
        if (best) rows.push(best);
      }
      rows.sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1]);
      rows.forEach((r) => (r.changes = [{f: r.f, c: r.c}]));
      const flipping = rows.filter((r) => r.flips), lowering = rows.filter((r) => !r.flips);
      const pairs = isClf && !flipping.length ? pairFlips(base) : [];
      const cls = S.esc(D.classes[base.cls] ? D.classes[base.cls].name : "");
      const groups = isClf
        ? [["Smallest single-feature changes that flip the class:", flipping.slice(0, 4)],
           ["No single change flips the class. Smallest pairs of changes that do:", pairs],
           [`Changes that lower P(${cls}) most without flipping it:`, lowering.slice(0, Math.max(0, 4 - flipping.length - pairs.length))]]
        : [["Single-feature changes that move the prediction most:", rows.slice(0, 5)]];
      const show = [];
      let html = "";
      for (const [lead, list] of groups) {
        if (!list.length) continue;
        html += `<div class="none">${lead}</div>`;
        for (const r of list) {
          const to = isClf
            ? `${S.classDot(r.p.cls)}${r.p.proba ? " " + S.pct(r.p.proba[r.p.cls]) : ""}`
            : `${S.fmt(r.p.value, Math.max(sig, 4))} <span class="tt-m">(${S.signed(r.delta)})</span>`;
          const what = r.changes.map((ch) => {
            let dg = sig;
            while (dg < 8 && ch.c.t != null && !isMissing(x[ch.f.i]) &&
                   Math.abs(ch.c.t - x[ch.f.i]) > ((ch.f.hi - ch.f.lo) || 1) * 1e-5 &&
                   S.fmt(ch.c.t, dg) === S.fmt(x[ch.f.i], dg)) dg++;
            return `<b>${S.esc(ch.f.name)}</b> ${S.esc(S.val(ch.f.i, x[ch.f.i], dg))} \u2192 ${S.esc(ch.c.txt)}`;
          }).join(" and ");
          html += `<button data-k="${show.length}" title="Apply ${r.changes.length > 1 ? "these changes" : "this change"}"><span class="what">${what}</span>` +
            `<span class="to">${to}</span></button>`;
          show.push(r);
        }
      }
      wi.innerHTML = html || `<div class="none">No change to one or two features moves this prediction.</div>`;
      wi.querySelectorAll("button").forEach((b) => {
        const r = show[+b.dataset.k];
        b.onclick = () => { for (const ch of r.changes) x[ch.f.i] = ch.c.v; truth = null; setFields(); refresh(); };
      });
    }
    // two-feature changes that flip the class, nearest first (top features, nearest candidates only)
    function pairFlips(base) {
      const cand = D.feats.slice(0, 10).map((f) => {
        const cur = x[f.i];
        return {f, list: candidates(f, cur).map((c) => ({c, d: dist(f, c, cur)})).sort((a, b) => a.d - b.d).slice(0, 12)};
      });
      const best = [];
      for (let a = 0; a < cand.length; a++) for (let b = a + 1; b < cand.length; b++) {
        let top = null;
        for (const ca of cand[a].list) for (const cb of cand[b].list) {
          const d = ca.d + cb.d;
          if (top && d >= top.key[1]) continue;
          const p = H.predict(Object.assign({}, x, {[cand[a].f.i]: ca.c.v, [cand[b].f.i]: cb.c.v}));
          if (p.cls !== base.cls) top = {changes: [{f: cand[a].f, c: ca.c}, {f: cand[b].f, c: cb.c}], p, key: [0, d], flips: true};
        }
        if (top) best.push(top);
      }
      return best.sort((a, b) => a.key[1] - b.key[1]).slice(0, 3);
    }

    function open(on) {
      on = on ?? !panel.classList.contains("open");
      panel.classList.toggle("open", on);
      predBtn.classList.toggle("on", on);
      if (on) {
        if (!x) { x = defaults(); buildFields(); }
        refresh();
        if (H.onOpen) setTimeout(H.onOpen, 270);
      } else {
        x = null;
        hud.classList.remove("predicting");
        if (H.show) H.show(null, null);
        if (H.onClose) setTimeout(H.onClose, 270);
      }
    }
    S.open = open;
    // load a training row into Predict (optionally one matching `keep`)
    S.loadSample = (keep, why) => {
      const pool = keep ? D.samples.filter((s) => keep(Object.assign(defaults(), fromSample(s.x)))) : D.samples;
      if (!pool.length) return false;
      const s = pool[Math.floor(Math.random() * pool.length)];
      if (!x) { x = defaults(); buildFields(); }
      Object.assign(x, fromSample(s.x));
      truth = s.y;
      truthLabel = why ? `${why}; true label` : "true label";
      if (!panel.classList.contains("open")) open(true); else { setFields(); refresh(); }
      setFields();
      return true;
    };
    S.setValues = (vals) => { if (!x) open(true); Object.assign(x, vals); truth = null; setFields(); refresh(); };
    S.withX = (vals) => Object.assign({}, x || defaults(), vals);
    S.defaults = defaults;

    S.start = (hooks) => {
      H = hooks;
      buildHud(hooks.word || "term");
      $("p-hint").textContent = hooks.hint || "Set feature values to see the prediction. Features are ordered by importance.";
      predBtn.onclick = () => open();
      const rb = $("b-rand");
      if (!D.samples.length) rb.style.display = "none";
      rb.onclick = () => S.loadSample(null);
      $("b-reset").onclick = () => { x = defaults(); truth = null; setFields(); refresh(); };
      $("b-svg").onclick = () => {
        const text = S.resolveVars(hooks.exportSvg());
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([text], {type: "image/svg+xml"}));
        a.download = (D.fileName || "model") + ".svg";
        a.click();
      };
    };
    // a small horizontal waterfall used by additive models and sums of trees
    S.waterfall = (items, totalLabel, total, pos, neg, limit = 12) => {
      // the baseline is listed but not drawn: it would set the scale and flatten every contribution
      const maxc = Math.max(...items.filter(([lab]) => lab !== "baseline").map(([, c]) => Math.abs(c)), 1e-12);
      const line = (lab, c) => `<div class="lab" title="${S.esc(lab)}">${S.esc(lab)}</div><div class="val">${S.signed(c)}</div>` +
        (lab === "baseline" ? `<div class="bar"></div>` :
          `<div class="bar" style="width:${(100 * Math.abs(c) / maxc).toFixed(1)}%;background:${c >= 0 ? pos : neg}"></div>`);
      let html = items.slice(0, limit).map(([lab, c]) => line(lab, c)).join("");
      if (items.length > limit) html += `<div class="lab tt-m">+${items.length - limit} smaller terms</div><div></div>`;
      return `<div class="wf">${html}<div class="lab tot">${S.esc(totalLabel)}</div><div class="val tot">${S.fmt(total, Math.max(sig, 4))}</div></div>`;
    };
    return S;
  }
  return {init};
})();
