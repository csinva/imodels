/* Tree pages: decision trees, rule lists (cascade layout) and sums of trees (several roots). */
(function () {
  "use strict";
  const S = DTI.init(), D = S.D, esc = S.esc, fmt = S.fmt;
  const N = D.nodes, NS = "http://www.w3.org/2000/svg";
  const ROOT0 = D.roots[0], ROOTW = N[ROOT0].wt, CASCADE = D.layout === "cascade", SUM = D.combine === "sum";
  const MEAN = D.combine === "mean", SHOWN = D.shown || D.roots;
  const svg = document.getElementById("canvas"), vp = document.getElementById("vp");
  const gRib = document.getElementById("g-rib"), gNode = document.getElementById("g-node"), gLab = document.getElementById("g-lab");
  const stage = document.getElementById("stage");
  const GAP = 20, LEVEL_TB = 64, LEVEL_LR = 120, DUR = 380;
  let orient = D.orientation === "LR" ? "LR" : "TB";
  let collapsed = new Set(), view = {k: 1, x: 0, y: 0}, cur = {}, path = new Set();
  const els = {}, ribs = {}, pills = {};
  svg.insertAdjacentHTML("afterbegin", D.defs);
  // which tree each node belongs to (sums of trees have several roots)
  N.forEach((n) => { let r = n.id; while (N[r].p >= 0) r = N[r].p; n.root = r; });

  // detailed and simple cards; `simple` picks which one each node shows
  N.forEach((n) => { n.dw = n.w; n.dh = n.h; n.dsvg = n.svg; });
  let simple = !!D.simple, cascBand = 0;
  function applyMode() {
    N.forEach((n) => { n.w = simple ? n.sw : n.dw; n.h = simple ? n.sh : n.dh; n.svg = simple ? n.ssvg : n.dsvg; });
    cascBand = 0.8 * Math.min(...N.map((n) => n.h));  // rule lists: flows turn sideways into cards
  }
  applyMode();

  // ---------- model evaluation
  const OPS = {"<=": (a, b) => a <= b, "<": (a, b) => a < b, ">": (a, b) => a > b, ">=": (a, b) => a >= b, "==": (a, b) => a === b, "!=": (a, b) => a !== b};
  function holds(n, x) {  // a split sends a sample left when all its conditions hold
    if (n.nl != null && Number.isNaN(x[n.c[0][0]])) return n.nl;  // sklearn: missing values go to a fixed side
    return n.c.every(([f, op, v]) => OPS[op](x[f], v));
  }
  function walk(x, root) {
    const p = [root];
    let id = root;
    while (!N[id].leaf) { id = holds(N[id], x) ? N[id].l : N[id].r; p.push(id); }
    return p;
  }
  const leafOf = (x, root) => { const p = walk(x, root); return p[p.length - 1]; };
  function predict(x) {
    const leaves = D.roots.map((r) => leafOf(x, r));
    if (MEAN) {  // forests: average the trees' predictions
      if (S.isClf) {
        const pr = D.classes.map((_, k) => leaves.reduce((a, id) => a + N[id].pr[k], 0) / leaves.length);
        const cls = pr.indexOf(Math.max(...pr));
        return {cls, proba: pr, leaves, note: `average of ${leaves.length} trees`};
      }
      return {value: leaves.reduce((a, id) => a + N[id].v, 0) / leaves.length, leaves, note: `average of ${leaves.length} trees`};
    }
    if (SUM) {
      const total = D.intercept + leaves.reduce((a, id) => a + N[id].v, 0);
      if (S.isClf) {
        const p1 = D.link === "logistic" ? 1 / (1 + Math.exp(-total)) : Math.min(1, Math.max(0, total));
        return {cls: p1 >= 0.5 ? 1 : 0, proba: [1 - p1, p1], leaves, total, note: `log-odds ${fmt(total)}`};
      }
      return {value: total, leaves, total, note: `${leaves.length} trees added together`};
    }
    const lf = N[leaves[0]];
    const note = `${CASCADE ? (lf.ruleLabel || "else") : "leaf #" + lf.id} · ${lf.n.toLocaleString()} training samples`;
    return S.isClf ? {cls: lf.pc, proba: lf.pr, leaves, note} : {value: lf.v, leaves, note};
  }

  // ---------- node elements
  function stackSvg(n) {
    let s = "";
    for (const d of [8, 4]) s += `<rect class="stack" x="${d}" y="${d}" width="${n.w}" height="${n.h}" rx="10" style="fill:var(--dti-card);stroke:var(--dti-ring)"/>`;
    return s;
  }
  function badgeSvg(n) {
    const t = "+" + n.desc, w = 12 + t.length * 6.4;
    return `<g class="badge" transform="translate(${n.w / 2},${n.h + 9})"><rect x="${-w / 2}" y="-9" width="${w}" height="18" rx="9" style="fill:var(--dti-ink)"/>` +
      `<text y="3.8" text-anchor="middle" style="font-size:10.5px;font-weight:600;fill:var(--dti-card)">${t}</text></g>`;
  }
  function nodeEl(id) {
    if (els[id]) return els[id];
    const n = N[id], g = document.createElementNS(NS, "g");
    g.setAttribute("class", "node" + (n.leaf ? " leaf" : ""));
    g.innerHTML = stackSvg(n) + n.svg.replace(/<title>[\s\S]*?<\/title>/, "") + (n.leaf ? "" : badgeSvg(n)) +
      `<rect class="ring" x="-3" y="-3" width="${n.w + 6}" height="${n.h + 6}" rx="13" style="fill:none;stroke:var(--dti-hl);stroke-width:2"/>` +
      `<g class="mk"></g>`;
    g.dataset.id = id;
    gNode.appendChild(g);
    els[id] = g;
    return g;
  }
  function ribEl(id) {
    if (ribs[id]) return ribs[id];
    const p = document.createElementNS(NS, "g");
    p.setAttribute("class", "rib");
    p.innerHTML = N[id].mix.map(([c]) => `<path style="fill:${c}"/>`).join("");
    gRib.appendChild(p);
    ribs[id] = p;
    const g = document.createElementNS(NS, "g"), w = N[id].labw;
    g.setAttribute("class", "pill");
    g.innerHTML = `<rect x="${-w / 2}" y="-9" width="${w}" height="18" rx="9" style="fill:var(--dti-card);stroke:var(--dti-ring)"/>` +
      `<text y="3.7" text-anchor="middle" style="font-size:10.5px;font-weight:600;fill:var(--dti-ink);font-variant-numeric:tabular-nums">${esc(N[id].lab)}</text>`;
    gLab.appendChild(g);
    pills[id] = g;
    return p;
  }

  // ---------- layout (contour-based tidy tree, same as the Python renderer)
  function kids(id) { const n = N[id]; return n.leaf || collapsed.has(id) ? [] : [n.l, n.r]; }
  function sideChild(pid) { const n = N[pid]; return N[n.l].leaf ? n.l : n.r; }
  let plusAt = [];
  function tidyFrom(root, H) {
    const B = (id) => (H ? N[id].h : N[id].w), Z = (id) => (H ? N[id].w : N[id].h);
    const rel = {};
    function place(id) {
      const cs = kids(id), half = B(id) / 2;
      if (!cs.length) return [[-half, half]];
      const conts = cs.map(place);
      let offs = [0], acc = conts[0].slice();
      for (let k = 1; k < conts.length; k++) {
        const c = conts[k];
        let s = -Infinity;
        for (let i = 0; i < Math.min(acc.length, c.length); i++) s = Math.max(s, acc[i][1] - c[i][0]);
        s += GAP;
        offs.push(s);
        const m = [];
        for (let i = 0; i < Math.max(acc.length, c.length); i++) {
          const a = acc[i], b = c[i] ? [c[i][0] + s, c[i][1] + s] : null;
          m.push(a && b ? [Math.min(a[0], b[0]), Math.max(a[1], b[1])] : a || b);
        }
        acc = m;
      }
      const mid = (offs[0] + offs[offs.length - 1]) / 2;
      cs.forEach((c, k) => (rel[c] = offs[k] - mid));
      return [[-half, half]].concat(acc.map(([a, b]) => [a - mid, b - mid]));
    }
    const cont = place(root);
    const lo = Math.min(...cont.map((c) => c[0])), hi = Math.max(...cont.map((c) => c[1]));
    const pos = {[root]: -lo}, vis = [root], st = [root];
    while (st.length) { const id = st.pop(); for (const c of kids(id)) { pos[c] = pos[id] + rel[c]; vis.push(c); st.push(c); } }
    const lv = {};
    for (const id of vis) lv[N[id].d] = Math.max(lv[N[id].d] || 0, Z(id));
    const off = {};
    let acc = 0;
    Object.keys(lv).map(Number).sort((a, b) => a - b).forEach((d) => { off[d] = acc; acc += lv[d] + (H ? LEVEL_LR : LEVEL_TB); });
    const out = {};
    for (const id of vis) {
      const n = N[id];
      out[id] = H ? [off[n.d], pos[id] - n.h / 2, n.w, n.h] : [pos[id] - n.w / 2, off[n.d], n.w, n.h];
    }
    return [out, hi - lo];
  }
  function cascade() {
    const out = {}, chain = [];
    let id = ROOT0;
    for (;;) {
      const n = N[id];
      if (n.leaf || collapsed.has(id)) { chain.push([id, null]); break; }
      const side = sideChild(id);
      chain.push([id, side]);
      id = side === n.l ? n.r : n.l;
    }
    const colW = Math.max(...chain.map(([i]) => N[i].w)), sx = colW + 110;
    let y = 0;
    for (const [i, sd] of chain) {
      const n = N[i];
      let rh = n.h;
      out[i] = [(colW - n.w) / 2, y, n.w, n.h];
      if (sd != null) { const m = N[sd]; out[sd] = [sx, m.h < n.h ? y + (n.h - m.h) / 2 : y, m.w, m.h]; rh = Math.max(rh, m.h); }
      y += rh + 62;
    }
    return out;
  }
  function layout() {
    plusAt = [];
    if (CASCADE) return cascade();
    const H = orient === "LR", out = {};
    let off = 0;
    SHOWN.forEach((r, k) => {
      const [boxes, extent] = tidyFrom(r, H);
      if (k) plusAt.push(H ? [N[r].w / 2, off - 36] : [off - 36, N[r].h / 2]);
      for (const id in boxes) { const b = boxes[id]; out[id] = H ? [b[0], b[1] + off, b[2], b[3]] : [b[0] + off, b[1], b[2], b[3]]; }
      off += extent + 72;
    });
    return out;
  }

  // ---------- ribbons: one sub-ribbon per class, sized by that class's share of the samples entering the child
  function bandW(wt) {
    const r = N[ROOT0];
    let b = Math.min(D.bandMax, 0.8 * (orient === "LR" && !CASCADE ? r.h : r.w));
    if (CASCADE) b = Math.min(b, cascBand);
    return Math.max(1.5, b * wt / ROOTW);
  }
  function ribbon(id, pb, cb) {
    const n = N[id], p = N[n.p], H = CASCADE ? id === sideChild(n.p) : orient === "LR";
    const pw = bandW(p.wt), lw = pw * N[p.l].wt / (p.wt || 1), share = pw * n.wt / (p.wt || 1);
    // in a cascade the two children leave from different sides, so each flow is centered
    const a0 = CASCADE ? -share / 2 : n.id === p.l ? -pw / 2 : -pw / 2 + lw;
    const a1 = CASCADE ? share / 2 : n.id === p.l ? -pw / 2 + lw : pw / 2;
    const cw = bandW(n.wt), c0 = -cw / 2, c1 = cw / 2;
    const r = (v) => Math.round(v * 10) / 10;
    const pc = H ? pb[1] + pb[3] / 2 : pb[0] + pb[2] / 2, pe = H ? pb[0] + pb[2] : pb[1] + pb[3];
    const cc = H ? cb[1] + cb[3] / 2 : cb[0] + cb[2] / 2, ce = H ? cb[0] : cb[1], m = (pe + ce) / 2;
    const ds = [];
    let A = pc + a0, C = cc + c0;
    for (const [, f] of n.mix) {
      const A0 = A, A1 = A + (a1 - a0) * f, C0 = C, C1 = C + (c1 - c0) * f;
      ds.push(H
        ? `M${r(pe)},${r(A0)}C${r(m)},${r(A0)} ${r(m)},${r(C0)} ${r(ce)},${r(C0)}L${r(ce)},${r(C1)}C${r(m)},${r(C1)} ${r(m)},${r(A1)} ${r(pe)},${r(A1)}Z`
        : `M${r(A0)},${r(pe)}C${r(A0)},${r(m)} ${r(C0)},${r(m)} ${r(C0)},${r(ce)}L${r(C1)},${r(ce)}C${r(C1)},${r(m)} ${r(A1)},${r(m)} ${r(A1)},${r(pe)}Z`);
      A = A1; C = C1;
    }
    const mid = (2 * pc + a0 + a1 + 2 * cc) / 4;
    return H ? [ds, m, mid] : [ds, mid, m];
  }

  // ---------- drawing + animation
  function applyView() { vp.setAttribute("transform", `translate(${view.x},${view.y}) scale(${view.k})`); }
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpBox = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), b[2], b[3]];
  function anchor(id, boxes, own) {
    let p = N[id].p;
    while (p >= 0 && !boxes[p]) p = N[p].p;
    if (p < 0) return own;
    const b = boxes[p], n = N[id];
    return [b[0] + b[2] / 2 - n.w / 2, b[1] + b[3] / 2 - n.h / 2, n.w, n.h];
  }
  let anim = null;
  function render(tgt, opts = {}) {
    const ids = new Set([...Object.keys(cur), ...Object.keys(tgt)].map(Number));
    const from = {}, to = {};
    for (const id of ids) { to[id] = tgt[id] || anchor(id, tgt, cur[id]); from[id] = cur[id] || anchor(id, cur, to[id]); }
    const v0 = {...view}, v1 = opts.view || view;
    for (const id of ids) {
      const g = nodeEl(id);
      g.style.display = "";
      g.classList.toggle("collapsed", collapsed.has(id) && !N[id].leaf);
      if (N[id].p >= 0) { ribEl(id).style.display = ""; pills[id].style.display = ""; }
    }
    gLab.querySelectorAll(".plus").forEach((e) => e.remove());
    for (const [x, y] of plusAt) gLab.insertAdjacentHTML("beforeend", `<text class="plus" x="${x}" y="${y + 9}" text-anchor="middle" style="font-size:26px;font-weight:300;fill:var(--dti-muted)">+</text>`);
    if (anim) cancelAnimationFrame(anim);
    const t0 = performance.now(), dur = opts.instant ? 0 : DUR;
    const frame = (now) => {
      const t = dur ? ease(Math.min(1, (now - t0) / dur)) : 1;
      const box = {};
      for (const id of ids) {
        const b = (box[id] = lerpBox(from[id], to[id], t));
        els[id].setAttribute("transform", `translate(${b[0].toFixed(1)},${b[1].toFixed(1)})`);
        const op = !cur[id] ? t : !tgt[id] ? 1 - t : 1;
        els[id].style.opacity = op < 1 ? op : "";
      }
      for (const id of ids) {
        const p = N[id].p;
        if (p < 0 || !box[p]) continue;
        const [ds, mx, my] = ribbon(id, box[p], box[id]);
        const paths = ribs[id].children;
        ds.forEach((d, k) => paths[k].setAttribute("d", d));
        const op = !cur[id] ? t : !tgt[id] ? 1 - t : 1;
        ribs[id].style.opacity = op < 1 ? op : "";
        pills[id].setAttribute("transform", `translate(${mx.toFixed(1)},${my.toFixed(1)})`);
        pills[id].style.opacity = op < 1 ? op : "";
      }
      view = {k: lerp(v0.k, v1.k, t), x: lerp(v0.x, v1.x, t), y: lerp(v0.y, v1.y, t)};
      applyView();
      if (t < 1) { anim = requestAnimationFrame(frame); return; }
      anim = null;
      for (const id of ids) if (!tgt[id]) {
        els[id].style.display = "none";
        if (N[id].p >= 0) { ribs[id].style.display = "none"; pills[id].style.display = "none"; }
      }
      cur = tgt;
    };
    if (dur) anim = requestAnimationFrame(frame); else frame(t0 + 1);
  }
  function bounds(boxes) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const b of Object.values(boxes)) { x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[0] + b[2] + 8); y1 = Math.max(y1, b[1] + b[3] + 18); }
    return [x0, y0, x1, y1];
  }
  function fitView(boxes) {
    const [x0, y0, x1, y1] = bounds(boxes), W = stage.clientWidth, Hh = stage.clientHeight, m = 36;
    const k = Math.max(0.08, Math.min(1.15, (W - 2 * m) / (x1 - x0), (Hh - 2 * m) / (y1 - y0)));
    return {k, x: (W - (x1 - x0) * k) / 2 - x0 * k, y: Math.max(m, (Hh - (y1 - y0) * k) / 2) - y0 * k};
  }
  // relayout, keeping node `keep` fixed on screen (or refitting)
  function update(keep, fit) {
    const tgt = layout();
    let v = view;
    if (fit) v = fitView(tgt);
    else if (keep != null && cur[keep] && tgt[keep]) {
      v = {k: view.k, x: view.x - (tgt[keep][0] - cur[keep][0]) * view.k, y: view.y - (tgt[keep][1] - cur[keep][1]) * view.k};
    }
    render(tgt, {view: v});
    syncPath();
  }

  // ---------- toolbar controls
  const depthEl = document.getElementById("depth"), depthV = document.getElementById("depthv");
  depthEl.max = Math.max(1, D.maxDepth);
  function setDepth(d, fit = true) {
    d = Math.max(1, Math.min(d, D.maxDepth || 1));
    collapsed = new Set(N.filter((n) => !n.leaf && n.d >= d).map((n) => n.id));
    depthEl.value = d; depthV.textContent = d;
    update(ROOT0, fit);
  }
  depthEl.addEventListener("input", () => setDepth(+depthEl.value));
  document.getElementById("b-fit").onclick = () => update(null, true);
  const orientBtn = document.getElementById("b-orient");
  function setOrientLabel() { orientBtn.querySelector("span").textContent = orient === "TB" ? "Vertical" : "Horizontal"; orientBtn.querySelector("svg").style.transform = orient === "TB" ? "" : "rotate(-90deg)"; }
  const simpleBtn = document.getElementById("b-simple");
  simpleBtn.classList.toggle("on", simple);
  simpleBtn.onclick = () => {
    simple = !simple;
    simpleBtn.classList.toggle("on", simple);
    applyMode();
    for (const id in els) { els[id].remove(); delete els[id]; }
    cur = {};
    const t = layout();
    render(t, {view: fitView(t), instant: true});
    syncPath();
  };
  orientBtn.onclick = () => { orient = orient === "TB" ? "LR" : "TB"; setOrientLabel(); update(null, true); };

  // ---------- pan, zoom, click
  let drag = null;
  stage.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    drag = {x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false, target: e.target.closest(".node")};
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    stage.classList.add("drag");
    S.hideTip();
    view.x = drag.vx + dx; view.y = drag.vy + dy;
    applyView();
  });
  stage.addEventListener("pointerup", () => {
    if (drag && !drag.moved && drag.target) {
      const id = +drag.target.dataset.id, n = N[id];
      if (n.leaf) {  // a leaf loads one of its training rows into Predict
        S.loadSample((x) => leafOf(x, n.root) === id, `training row from ${CASCADE ? (n.ruleLabel || "the else outcome") : "leaf #" + id}`);
      } else {
        if (collapsed.has(id)) collapsed.delete(id); else collapsed.add(id);
        update(id, false);
      }
    }
    drag = null;
    stage.classList.remove("drag");
  });
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = stage.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.max(0.05, Math.min(4, view.k * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015))));
    view = {k, x: mx - (mx - view.x) * k / view.k, y: my - (my - view.y) * k / view.k};
    applyView();
  }, {passive: false});

  // ---------- hover: tooltip + path highlight
  let hoverPath = [];
  const ancestors = (id) => { const out = []; while (id >= 0) { out.push(id); id = N[id].p; } return out; };
  gNode.addEventListener("pointerover", (e) => {
    const g = e.target.closest(".node");
    if (!g || drag) return;
    const id = +g.dataset.id, n = N[id];
    hoverPath.forEach((a) => ribs[a] && ribs[a].classList.remove("hl"));
    hoverPath = ancestors(id);
    hoverPath.forEach((a) => ribs[a] && ribs[a].classList.add("hl"));
    const act = n.leaf ? (D.samples.length ? "Click to load a training row from here into Predict." : "")
      : `Click to ${collapsed.has(id) ? "expand" : "collapse"} (${n.desc} nodes below).`;
    S.tip(n.tip + (act ? `<div class="tt-act">${act}</div>` : ""));
    S.hotFeatures(n.leaf ? [] : n.fs);
  });
  gNode.addEventListener("pointerout", (e) => {
    const g = e.target.closest(".node");
    if (g && !g.contains(e.relatedTarget)) {
      hoverPath.forEach((a) => ribs[a] && ribs[a].classList.remove("hl"));
      hoverPath = [];
      S.hideTip();
      S.hotFeatures([]);
    }
  });

  // ---------- Predict: highlight the path(s) and mark the value in each chart
  function syncPath() {
    const x = S.x();
    svg.classList.toggle("predicting", !!x);
    for (const id in els) {
      const on = !!x && path.has(+id);
      els[id].classList.toggle("onpath", on);
      if (N[id].p >= 0) { ribs[id] && ribs[id].classList.toggle("onpath", on); pills[id] && pills[id].classList.toggle("onpath", on); }
      const mk = els[id].querySelector(".mk"), c = N[id].chart;
      mk.innerHTML = "";
      if (on && c && !simple && !N[id].leaf) {
        const v = x[N[id].f], fr = Math.max(0, Math.min(1, (v - c.lo) / (c.hi - c.lo))), px = c.x0 + fr * (c.x1 - c.x0);
        mk.innerHTML = `<line x1="${px}" x2="${px}" y1="${c.y0 - 3}" y2="${c.y1}" style="stroke:var(--dti-hl);stroke-width:2"/>` +
          `<circle cx="${px}" cy="${c.y0 - 3}" r="3.5" style="fill:var(--dti-hl);stroke:var(--dti-card);stroke-width:1.5"/>`;
      }
    }
  }
  function show(x) {
    if (!x) { path = new Set(); syncPath(); return; }
    const p = D.roots.flatMap((r) => walk(x, r));
    path = new Set(p);
    let opened = false;
    for (const id of p) if (collapsed.has(id)) { collapsed.delete(id); opened = true; }
    if (opened) update(null, false); else syncPath();
  }
  function stepText(id, x) {  // one edge of the path, as read in the Predict panel
    const par = N[N[id].p];
    if (par.f >= 0) return `${esc(S.feat(par.f).name)} = ${esc(S.val(par.f, x[par.f]))} <span class="tt-m">(${esc(N[id].lab)})</span>`;
    return `${esc(par.st)} <span class="tt-m">(${esc(N[id].lab)})</span>`;
  }
  function explain(x, p) {
    if (MEAN) {  // each tree's vote, then the average
      const k = S.isClf ? (D.classes.length === 2 ? 1 : p.cls) : null;
      const vals = p.leaves.map((id) => (S.isClf ? N[id].pr[k] : N[id].v));
      const lab = S.isClf ? `P(${esc(D.classes[k].name)})` : "prediction";
      const rows = vals.slice(0, 12).map((v, i) => `<div class="lab">tree ${i + 1}</div><div class="val">${S.isClf ? S.pct(v) : fmt(v)}</div>`).join("");
      return `<div class="wf">${rows}${vals.length > 12 ? `<div class="lab tt-m">+${vals.length - 12} more trees</div><div></div>` : ""}` +
        `<div class="lab tot">average ${lab}</div><div class="val tot">${S.isClf ? S.pct(vals.reduce((a, b) => a + b, 0) / vals.length) : fmt(p.value)}</div></div>`;
    }
    if (SUM) {
      const items = p.leaves.map((id, k) => [`tree ${k + 1}`, N[id].v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      if (D.intercept) items.unshift(["baseline", D.intercept]);
      const pos = S.isClf ? D.classes[1].col : "var(--dti-c0)", neg = S.isClf ? D.classes[0].col : "var(--dti-c7)";
      return S.waterfall(items, S.isClf ? "log-odds" : "prediction", p.total, pos, neg);
    }
    const steps = walk(x, ROOT0).slice(1).map((id) => `<li>${stepText(id, x)}</li>`).join("");
    return `<ul class="tt-r path">${steps}</ul>`;
  }
  function exportSvg() {
    const [x0, y0, x1, y1] = bounds(cur), m = 24;
    const clone = svg.cloneNode(true);
    clone.querySelector("#vp").setAttribute("transform", `translate(${m - x0},${m - y0})`);
    clone.setAttribute("width", x1 - x0 + 2 * m); clone.setAttribute("height", y1 - y0 + 2 * m);
    clone.setAttribute("viewBox", `0 0 ${x1 - x0 + 2 * m} ${y1 - y0 + 2 * m}`);
    clone.removeAttribute("id");
    clone.querySelectorAll('[style*="display: none"]').forEach((e) => e.remove());
    let text = new XMLSerializer().serializeToString(clone);
    const style = `<style>.stack,.badge,.ring{display:none}.collapsed .stack,.collapsed .badge{display:inline}.rib{fill-opacity:.32}</style>`;
    return text.replace(/(<svg[^>]*>)/, `$1${style}<rect width="100%" height="100%" fill="${S.surface()}"/>`);
  }

  S.start({
    word: CASCADE ? "rule" : "split",
    hint: CASCADE ? "Set feature values to see which rule applies first. Click an outcome to load a training row it covers."
      : SUM ? "Set feature values to follow each tree's path; their leaf values add up. Click a leaf to load a training row from it."
        : "Set feature values to follow the decision path. Click a leaf to load a training row from it.",
    predict, explain, show, exportSvg,
    highlight(f) {
      svg.classList.toggle("feat-mode", f != null);
      for (const id in els) els[id].classList.toggle("feat", f != null && N[id].fs.includes(f));
    },
    onOpen: () => update(null, true),
    onClose: () => update(null, true),
  });

  // ---------- init
  setOrientLabel();
  if (CASCADE) orientBtn.style.display = "none";  // rule lists have one layout
  const d0 = Math.min(D.initialDepth, D.maxDepth || 1);
  collapsed = new Set(N.filter((n) => !n.leaf && n.d >= d0).map((n) => n.id));
  depthEl.value = d0; depthV.textContent = d0;
  const t0 = layout();
  render(t0, {view: fitView(t0), instant: true});
  addEventListener("resize", () => { if (!anim) update(null, true); });
})();
