"""Regenerate the data behind Fig 1 on the gpgam page.

Fits one GPGam to a small synthetic dataset whose five features differ only in
how smooth their effect is, then rewrites the `var S = {...};` line in
gpgam.html in place. The sample is deliberately small: the posterior bands are
part of what the figure is showing, and on tens of thousands of rows they
collapse to the width of the line and show nothing.

    uv run python gpgam_synth.py
"""

import json
import re
import sys

import numpy as np

sys.path.insert(0, '../..')
from imodels import GPGamRegressor

N = 250
NOISE = 0.25
SEED = 0
STEP_AT, STEP_SIZE = 0.55, 1.4

# each entry is (panel name, the true effect of that feature)
EFFECTS = [
    ('smooth wave',   lambda x: np.sin(2 * np.pi * x)),
    ('rapid wave',    lambda x: 0.7 * np.sin(10 * np.pi * x)),
    ('sharp step',    lambda x: STEP_SIZE * (x > STEP_AT) - STEP_SIZE * (1 - STEP_AT)),
    ('straight line', lambda x: 2 * x - 1),
    ('nothing',       lambda x: np.zeros_like(x)),
]


def r(a, k=4):
    return [round(float(v), k) for v in np.asarray(a)]


rng = np.random.default_rng(SEED)
X = rng.random((N, len(EFFECTS)))
parts = np.column_stack([f(X[:, j]) for j, (_, f) in enumerate(EFFECTS)])
y = parts.sum(1) + rng.normal(0, NOISE, N)

model = GPGamRegressor(n_pairs=0, n_bins=64).fit(X, y)

grid_x = np.linspace(0, 1, 200)
features = []
for j, (name, f) in enumerate(EFFECTS):
    grid, mean, std = model.shape_function(j, return_std=True)
    w = model.kernel_weights(j)
    # the partial residual: y with every other feature's true effect removed,
    # which leaves this feature's effect plus the noise
    resid = y - (parts.sum(1) - parts[:, j])
    features.append({
        'name': name,
        'grid': r(grid), 'mean': r(mean), 'std': r(std),
        'rough': round(w['matern-0.05'], 6), 'smooth': round(w['rbf-0.25'], 6),
        'sx': r(X[:, j], 3), 'sy': r(resid, 3),
        'true_x': r(grid_x, 3), 'true_y': r(f(grid_x)),
    })

page = 'gpgam.html'
src = open(page).read()
new, n_sub = re.subn(r'var S = \{.*?\};',
                     'var S = ' + json.dumps({'features': features}) + ';',
                     src, count=1, flags=re.S)
assert n_sub == 1, 'could not find the "var S = {...};" line in ' + page
open(page, 'w').write(new)

# the numbers the surrounding prose quotes, so they can be kept honest
print('%d rows, %d bins, noise sd %.2f' % (N, len(features[0]['grid']), NOISE))
for feat in features:
    rough, smooth = feat['rough'], feat['smooth']
    ratio = ('smooth %.0fx rough' % (smooth / rough)) if smooth > rough else ('rough %.0fx smooth' % (rough / smooth))
    band = 2 * np.mean(feat['std'])
    print('  %-14s rough=%.5f smooth=%.5f  %-22s mean band +-%.2f'
          % (feat['name'], rough, smooth, ratio, band))

g = np.array(features[2]['grid']); v = np.array(features[2]['mean'])
lo, hi = np.median(v[g < 0.45]), np.median(v[g > 0.65])
k = int(np.argmax(np.abs(np.diff(v))))   # the jump falls between bins k and k+1
print('  step: spans %.2f (true %.2f), turns at %.3f (true %.2f)'
      % (hi - lo, STEP_SIZE, (g[k] + g[k + 1]) / 2, STEP_AT))
print('  rough weight on the step vs the slow wave: %.0fx'
      % (features[2]['rough'] / features[0]['rough']))
