"""Render the intro and methods table of the literature notes of the FastRiskScore project (evolve_slim/LITERATURE.md in
csinva/agentic-imodels) into the "Related work" collapsible of fastriskscore.html, between
<!-- LITERATURE --> and <!-- /LITERATURE -->. Only that region is rewritten.

    uv run python docs/pages/fastriskscore_literature.py --src ../agentic-imodels/evolve_slim/LITERATURE.md
"""

import argparse
import os
import re

import markdown

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastriskscore.html")
INDENT = " " * 22


def render(md_text):
    md_text = re.sub(r"\A# .*\n", "", md_text)  # the collapsible's summary is the title
    md_text = md_text.split("\n## ", 2)
    md_text = md_text[0] + "\n## " + md_text[1]  # the intro and the methods table only
    html = markdown.markdown(md_text, extensions=["tables"])
    html = html.replace("<h3>", "<h5>").replace("</h3>", "</h5>").replace("<h2>", "<h4>").replace("</h2>", "</h4>")
    # wide tables scroll sideways on their own instead of widening the page
    html = html.replace("<table>", '<div class="lit-scroll"><table class="ranktbl dataset-table">')
    html = html.replace("</table>", "</table></div>")
    html = html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    return "\n".join(INDENT + line if line.strip() else line for line in html.split("\n"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to evolve_slim/LITERATURE.md")
    src = ap.parse_args().src
    page = open(PAGE).read()
    pat = re.compile(r"(<!-- LITERATURE -->\n).*?(\s*<!-- /LITERATURE -->)", re.S)
    assert len(pat.findall(page)) == 1
    page = pat.sub(lambda m: m.group(1) + render(open(src).read()).rstrip("\n") + "\n" + m.group(2).lstrip("\n"), page)
    open(PAGE, "w").write(page)
    print("wrote", PAGE)
