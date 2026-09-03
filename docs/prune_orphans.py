"""Delete generated pages for modules that no longer exist.

build_docs.sh copies pdoc's output over the top of docs/ with `cp -rf`, which
overwrites but never removes. A module deleted from the package therefore leaves
its page behind forever, slowly rotting on an ancient version of the template.
Run this against pdoc's staging tree before the copy so the site only ever
contains pages pdoc just built, plus the hand-written ones.
"""

import os
import shutil

STAGING = 'imodels'          # pdoc's fresh output, before it is copied over docs/
HANDWRITTEN_DIRS = {'pages', 'img', 'conda', 'paper', STAGING}

# the blog pages at the root are rendered from docs/pages/ by build_pages.py
keep = {f for f in os.listdir('pages') if f.endswith('.html')}
for root, dirs, files in os.walk(STAGING):
    rel = os.path.relpath(root, STAGING)
    for f in files:
        if f.endswith('.html'):
            keep.add(os.path.normpath(os.path.join(rel, f)))

removed = []
for root, dirs, files in os.walk('.', topdown=True):
    if root == '.':
        dirs[:] = [d for d in dirs if d not in HANDWRITTEN_DIRS and not d.startswith('.')]
    for f in files:
        if not f.endswith('.html'):
            continue
        rel = os.path.normpath(os.path.relpath(os.path.join(root, f), '.'))
        if rel not in keep:
            os.remove(os.path.join(root, f))
            removed.append(rel)

# a module directory that lost all its pages is itself gone from the package
for root, dirs, files in os.walk('.', topdown=False):
    if root == '.' or os.path.basename(root) in HANDWRITTEN_DIRS:
        continue
    if os.path.relpath(root, '.').split(os.sep)[0] in HANDWRITTEN_DIRS:
        continue
    if not os.listdir(root):
        shutil.rmtree(root)
        removed.append(os.path.relpath(root, '.') + os.sep)

if removed:
    print('  prune_orphans.py: removed %d page(s) for modules no longer in the package:' % len(removed))
    for r in sorted(removed):
        print('    ' + r)
