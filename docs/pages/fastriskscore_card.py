"""Draw the risk-score cards of fastriskscore.html (the abstract's 3-condition score and the
quickstart's 5-condition score) as HTML, and splice each between its marker comments
<!-- CARD:name --> ... <!-- /CARD:name --> in the page. Only those regions are rewritten.

    uv run python docs/pages/fastriskscore_card.py
"""

import html
import itertools
import os
import re

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imodels import FastRiskScoreClassifier

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastriskscore.html")

STYLE = """<style>
  /* risk-score cards (drawn by fastriskscore_card.py) */
  .rs-card { max-width: 30rem; margin: 1.2em auto 1.4em; padding: 1.1em 1.3em 1em; background: var(--surface);
    border: 1px solid var(--line); border-radius: 12px; box-shadow: 0 1px 2px rgba(27,31,35,.04), 0 6px 18px rgba(27,31,35,.06);
    font-size: 0.92rem; line-height: 1.4; color: var(--ink); text-align: left; hyphens: none; }
  .rs-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1em; margin-bottom: .7em; }
  .rs-title { font-weight: 600; letter-spacing: .01em; }
  .rs-meta { color: var(--muted); font-size: .82em; white-space: nowrap; }
  .rs-rows { list-style: none; margin: 0; padding: 0; counter-reset: rs; }
  .rs-rows li { display: flex; align-items: center; gap: .7em; padding: .42em 0; border-top: 1px solid var(--line-soft); }
  .rs-rows li:first-child { border-top: none; }
  .rs-rows li::before { counter-increment: rs; content: counter(rs); flex: none; width: 1.5em; height: 1.5em; border-radius: 50%;
    background: var(--surface-alt); border: 1px solid var(--line); color: var(--muted); font-size: .75em; display: grid; place-items: center; }
  .rs-cond { flex: 1; min-width: 0; }
  .rs-cond b { font-weight: 500; }
  .rs-cond span { color: var(--muted); font-variant-numeric: tabular-nums; }
  .rs-pts { flex: none; min-width: 2.6em; text-align: center; padding: .12em .5em; border-radius: 999px; font-weight: 600;
    font-variant-numeric: tabular-nums; color: #fff; background: var(--accent); }
  .rs-pts.neg { background: #c05621; }
  .rs-sum { display: flex; justify-content: space-between; margin: .5em 0 .9em; padding-top: .5em; border-top: 2px solid var(--ink);
    font-size: .85em; color: var(--muted); }
  .rs-sum b { color: var(--ink); font-weight: 600; }
  .rs-chart { position: relative; display: flex; align-items: flex-end; gap: 3px; height: 5.2em; }
  .rs-half { position: absolute; left: 0; right: 0; bottom: 50%; border-top: 1px dashed var(--line); pointer-events: none; }
  .rs-half span { position: absolute; left: 0; bottom: 2px; font-size: .62em; color: var(--muted); }
  .rs-col { flex: 1 1 0; min-width: 0; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: stretch; }
  .rs-col i { display: block; font-style: normal; font-size: .68em; color: var(--ink-soft); text-align: center;
    font-variant-numeric: tabular-nums; margin-bottom: 2px; white-space: nowrap; }
  .rs-col s { display: block; text-decoration: none; border-radius: 3px 3px 0 0; min-height: 2px; }
  .rs-axis { display: flex; gap: 3px; border-top: 1px solid var(--line); padding-top: 3px; }
  .rs-axis span { flex: 1 1 0; min-width: 0; text-align: center; font-size: .7em; color: var(--muted); font-variant-numeric: tabular-nums; }
  .rs-foot { display: flex; justify-content: space-between; gap: 1em; margin-top: .35em; font-size: .75em; color: var(--muted); }
  .rs-card.dense .rs-col i { display: none; }
  .rs-card.wide { max-width: 50rem; }
  .rs-card.wide .rs-body { display: grid; grid-template-columns: 1.15fr 1fr; gap: 0 2em; align-items: end; }
  .rs-card.wide { padding: .75em 1.1em .7em; margin: .8em auto 1em; }
  .rs-card.wide .rs-head { margin-bottom: .3em; }
  .rs-card.wide .rs-rows li { padding: .22em 0; }
  .rs-card.wide .rs-pts { padding: .02em .5em; }
  .rs-card.wide .rs-chart { height: 4.6em; }
  .rs-card.wide .rs-sum { margin: .35em 0 0; padding-top: .35em; }
  .rs-card.wide .rs-foot { margin-top: .2em; }
  @media (max-width: 640px) { .rs-card.wide .rs-body { grid-template-columns: 1fr; } .rs-card.wide .rs-sum { margin-bottom: .9em; }
    .rs-card { padding: .9em .9em .8em; } .abstract .rs-card { margin-left: -2.4em; margin-right: -2.4em; } }
</style>"""


def pretty(cond):
    """'worst concave points <= 0.1479' -> name, operator, value."""
    m = re.match(r"(.*?)\s*(<=|>=|<|>|==|=)\s*(.*)$", cond)
    if not m:
        return f"<b>{html.escape(cond)}</b>"
    op = {"<=": "&le;", ">=": "&ge;", "==": "=", "=": "="}.get(m.group(2), html.escape(m.group(2)))
    return f"<b>{html.escape(m.group(1))}</b> <span>{op} {html.escape(m.group(3))}</span>"


def pct(p):
    return f"{100 * p:.0f}%" if 0.095 <= p <= 0.995 else f"{100 * p:.1f}%"


def card(model, title, outcome, auc, dense=False, wide=False):
    pts = {c: int(v) for c, v in model.points_.items() if v}
    totals = sorted({sum(s) for r in range(len(pts) + 1) for s in itertools.combinations(pts.values(), r)})
    rows = "\n".join(f'    <li><span class="rs-cond">{pretty(c)}</span><span class="rs-pts{" neg" if v < 0 else ""}">'
                     f'{v:+d}</span></li>' for c, v in sorted(pts.items(), key=lambda cv: -cv[1]))
    cols, axis = [], []
    for t in totals:
        p = float(model.risk(t))
        # bar height is the risk; colour runs from a pale to the full accent with the risk
        cols.append(f'<div class="rs-col" title="total {t}: {100 * p:.1f}%"><i>{pct(p)}</i><s style="height:{max(100 * p, 2):.1f}%;'
                    f'background:color-mix(in srgb, var(--accent) {25 + 75 * p:.0f}%, var(--surface))"></s></div>')
        axis.append(f"<span>{t}</span>")
    return f"""<div class="rs-card{" dense" if dense else ""}{" wide" if wide else ""}" role="img" aria-label="{html.escape(title)}: {len(pts)} conditions and the {html.escape(outcome)} for each total score">
  <div class="rs-head"><span class="rs-title">{html.escape(title)}</span><span class="rs-meta">test AUC {auc:.3f}</span></div>
  <div class="rs-body"><div class="rs-left">
  <ol class="rs-rows">
{rows}
  </ol>
  <div class="rs-sum"><span>Add the points of every condition that applies.</span></div>
  </div><div class="rs-right">
  <div class="rs-chart"><div class="rs-half"><span>50%</span></div>{"".join(cols)}</div>
  <div class="rs-axis">{"".join(axis)}</div>
  <div class="rs-foot"><span>&uarr; {html.escape(outcome)}</span><span>total score &rarr;</span></div>
  </div></div>
</div>"""


def splice(page, name, block):
    pat = re.compile(rf"(<!-- CARD:{name} -->\n).*?(\s*<!-- /CARD:{name} -->)", re.S)
    assert len(pat.findall(page)) == 1, name
    return pat.sub(lambda m: m.group(1) + block + m.group(2), page)


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
    page = open(PAGE).read()
    for name, k, dense, wide in [("abstract", 3, False, True), ("quickstart", 5, True, True)]:
        m = FastRiskScoreClassifier(k=k).fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        block = card(m, "Benign breast tumour score", "chance the tumour is benign", auc, dense, wide)
        if name == "abstract":
            block = STYLE + "\n" + block
        page = splice(page, name, block)
        print(name, k, f"AUC {auc:.3f}", dict(m.points_))
    open(PAGE, "w").write(page)
