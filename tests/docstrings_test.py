"""Every parameter documented in a docstring renders as a parameter on the docs pages (pdoc3).

pdoc formats a section as a list only when it uses a heading it knows (NumPy ``Parameters`` with a
dashed underline, or Google ``Args:``) and each entry follows the format: ``name : type`` on one line
with the description indented below it. An entry written another way silently renders as plain text.
"""

import inspect
import re
import warnings

import pytest

pdoc = pytest.importorskip("pdoc")
from pdoc.html_helpers import to_markdown  # noqa: E402

NUMPY = ("Parameters", "Attributes", "Returns", "Yields", "Raises", "Other Parameters", "Arguments", "Args")
GOOGLE = ("Args", "Arguments", "Attributes", "Returns", "Yields", "Raises")


def _expected_entries(doc):
    """Number of entries the docstring's parameter-like sections declare."""
    lines = inspect.cleandoc(doc).split("\n")
    n, i = 0, 0
    while i < len(lines):
        line = lines[i]
        numpy_head = line.strip() in NUMPY and i + 1 < len(lines) and re.fullmatch(r"\s*-{3,}\s*", lines[i + 1])
        google_head = re.fullmatch(r"\s*(%s):\s*" % "|".join(GOOGLE), line)
        if not (numpy_head or google_head):
            i += 1
            continue
        base = len(line) - len(line.lstrip()) + (0 if numpy_head else 4)
        j = i + (2 if numpy_head else 1)
        while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) >= base):
            if j + 1 < len(lines) and re.fullmatch(r"\s*-{3,}\s*", lines[j + 1]):
                break
            if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) == base:
                n += 1
            j += 1
        i = j
    return n


def _documented():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = pdoc.Module("imodels", skip_errors=True)

    def walk(m):
        yield m
        for d in m.doc.values():
            yield d
            if isinstance(d, pdoc.Class):
                yield from d.doc.values()
        for sm in m.submodules():
            yield from walk(sm)

    seen = set()
    for d in walk(root):
        if d.docstring and ".experimental." not in d.refname + "." and d.docstring not in seen:
            seen.add(d.docstring)
            yield d


def test_docstring_parameters_render():
    lost = []
    for d in _documented():
        expected = _expected_entries(d.docstring)
        rendered = len(re.findall(r"^:   ", to_markdown(d.docstring, module=d.module), re.M))
        if rendered < expected:
            lost.append(f"{d.refname}: {expected} entries, {rendered} rendered")
    assert not lost, "docstring entries that pdoc renders as plain text:\n" + "\n".join(lost)


def test_no_unknown_parameter_headings():
    bad = [d.refname for d in _documented()
           if re.search(r"^\s*Params?\s*\n\s*-{3,}|^\s*(Parameters|Returns|Attributes):\s*\n\s*-{3,}", d.docstring, re.M)]
    assert not bad, "use 'Parameters' with a dashed underline (no colon): " + ", ".join(bad)
