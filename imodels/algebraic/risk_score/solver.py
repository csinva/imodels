"""Solver behind FastRiskScoreClassifier: sparse integer points by beam search, calibrated rounding
and integer local search.

Found by an autoresearch loop (agentic-imodels, evolve_slim, run sep26-run2, version v35) that
started from FasterRisk (Liu et al., NeurIPS 2022) and was scored by the calibrated training loss

    min over integer w (at most k nonzero, each in [-bound, bound]) of
        min over real a, b of  sum_i log(1 + exp(-y_i (a * x_i.w + b)))

Pipeline:

1. data: unique (x, y) rows with counts; per column the CSC nonzeros and codes of its distinct values;
2. continuous beam search (FasterRisk's: grow the support one column at a time from the 10 best parents by
   the 10 largest gradients), where every node keeps its rows grouped by (y, x_support): a child's groups are
   the parent's groups split by x_j, so its projected-Newton fit runs on a few hundred groups instead of all
   rows and only needs the nonzeros of column j;
3. rounding: for each of the 10 final supports, round m * beta over a grid of scales and keep the rounding
   with the smallest calibrated loss;
4. integer local search from the 5 best roundings: value changes and additions, then swaps when those fail.
   Moves are scored on cells (score bin, x_j value) by an estimate of the calibrated loss (exact at the
   current (a, b), then one Newton step in (a, b) with the touched cells exact); the best two are checked
   exactly. Swap-in columns are screened by a second-order model (two BLAS mat-vecs).

numba is needed to fit but not to import this module.
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np

#: numba is optional for importing imodels but required to fit this model
HAVE_NUMBA = importlib.util.find_spec("numba") is not None
#: compiled kernels are cached on disk (about 30 s to compile once per machine); set
#: RISKSCORE_NUMBA_CACHE=0 to disable, e.g. when the package directory is read-only
NUMBA_CACHE = os.environ.get("RISKSCORE_NUMBA_CACHE", "1") != "0"

if HAVE_NUMBA:
    import numba as nb
else:
    class nb:  # noqa: N801 - stand-in so the module imports without numba
        @staticmethod
        def njit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

COEF_BOUND = 5
TR_A, TR_B = 1.0, 2.0  # trust region of the (a, b) Newton step in the move estimate


# ---------------------------------------------------------------- kernels
@nb.njit(cache=NUMBA_CACHE, inline="always")
def _lrow(m):
    # log(1 + exp(-m)), stable
    if m > 0:
        return np.log1p(np.exp(-m))
    return -m + np.log1p(np.exp(m))


@nb.njit(cache=NUMBA_CACHE, inline="always")
def _sig(z):
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return e / (1.0 + e)


@nb.njit(cache=NUMBA_CACHE)
def total_loss(ym, c):
    s = 0.0
    for i in range(ym.shape[0]):
        s += c[i] * _lrow(ym[i])
    return s


@nb.njit(cache=NUMBA_CACHE)
def column_codes(XT, y, max_vals):
    """Per column: rank codes of all values, CSC nonzeros, codes among distinct nonzero values."""
    d, n = XT.shape
    ncode = np.ones(d, np.int64)
    rank = np.empty(n, np.int64)
    isbin = np.zeros(d, np.bool_)
    cnz = np.zeros(d, np.int64)
    has0 = np.zeros(d, np.bool_)
    for j in range(d):
        col = XT[j]
        b = True
        nz = 0
        for i in range(n):
            v = col[i]
            if v != 0.0:
                nz += 1
                if v != 1.0:
                    b = False
        isbin[j] = b
        cnz[j] = nz
        has0[j] = nz < n
    nnz = cnz.sum()
    ptr = np.zeros(d + 1, np.int64)
    for j in range(d):
        ptr[j + 1] = ptr[j] + cnz[j]
    idx = np.empty(nnz, np.int64)
    xval = np.empty(nnz)
    xcode = np.zeros(nnz, np.int64)
    raw = np.zeros(d, np.bool_)
    vals_all = np.empty(nnz + d)
    vptr = np.zeros(d + 1, np.int64)
    uvals = np.empty(n)
    for j in range(d):
        col = XT[j]
        t = ptr[j]
        if isbin[j]:
            h0 = has0[j]
            h1 = cnz[j] > 0
            ncode[j] = (1 if h0 else 0) + (1 if h1 else 0)
            if h1:
                vals_all[vptr[j]] = 1.0
                vptr[j + 1] = vptr[j] + 1
            else:
                vptr[j + 1] = vptr[j]
            for i in range(n):
                if col[i] != 0.0:
                    idx[t] = i
                    xval[t] = 1.0
                    t += 1
            continue
        order = np.argsort(col)
        nu = 0
        for r in range(n):
            i = order[r]
            if r == 0 or col[i] != uvals[nu - 1]:
                uvals[nu] = col[i]
                nu += 1
            rank[i] = nu - 1
        ncode[j] = nu
        zpos = -1
        for q in range(nu):
            if uvals[q] == 0.0:
                zpos = q
        nnzv = nu - (1 if zpos >= 0 else 0)
        if nnzv > max_vals:
            raw[j] = True
            vptr[j + 1] = vptr[j]
        else:
            for q in range(nu):
                if q != zpos:
                    vals_all[vptr[j] + (q if (zpos < 0 or q < zpos) else q - 1)] = uvals[q]
            vptr[j + 1] = vptr[j] + nnzv
        for i in range(n):
            if col[i] != 0.0:
                idx[t] = i
                xval[t] = col[i]
                if not raw[j]:
                    cq = rank[i]
                    xcode[t] = cq if (zpos < 0 or cq < zpos) else cq - 1
                t += 1
    return ncode, isbin, ptr, idx, xval, xcode, raw, vals_all[:vptr[d]].copy(), vptr


@nb.njit(cache=NUMBA_CACHE)
def col_moments(XT, w):
    d, n = XT.shape
    mean = np.zeros(d)
    var = np.zeros(d)
    for j in range(d):
        m1 = 0.0
        m2 = 0.0
        for i in range(n):
            v = XT[j, i]
            m1 += w[i] * v
            m2 += w[i] * v * v
        mean[j] = m1
        var[j] = m2 - m1 * m1
    return mean, var


class Data:
    def code(self, j):
        """Rank code of every row's value in column j (computed on first use)."""
        cj = self._codes.get(j)
        if cj is None:
            if self.isbin[j]:
                cj = self.XT[j].astype(np.int64) if self.ncode[j] > 1 else np.zeros(self.n, np.int64)
            else:
                cj = np.unique(self.XT[j], return_inverse=True)[1].astype(np.int64)
            self._codes[j] = cj
        return cj

    @property
    def XT2(self):
        if self._XT2 is None:
            self._XT2 = self.XT if self.isbin.all() else self.XT * self.XT
        return self._XT2

    def __init__(self, X, y01, bound=COEF_BOUND):
        self.bound = int(bound)
        ys = np.where(np.asarray(y01) > 0, 1.0, -1.0)
        # unique (x, y) rows via a random projection hash (deterministic seed)
        r = np.random.default_rng(12345).standard_normal(X.shape[1] + 1)
        h = X @ r[:-1] + ys * r[-1]
        _, first, inv = np.unique(h, return_index=True, return_inverse=True)
        cnt = np.bincount(inv.ravel())
        self.X = np.ascontiguousarray(X[first])
        self.y = np.ascontiguousarray(ys[first])
        self.c = cnt.astype(np.float64)
        self.n, self.d = self.X.shape
        self.N = float(self.c.sum())
        self.XT = np.ascontiguousarray(self.X.T)
        self._XT2 = None
        self.yc = self.y * self.c
        (self.ncode, self.isbin, self.ptr, self.idx, self.xval, self.xcode, self.raw, self.vals,
         self.vptr) = column_codes(self.XT, self.y, 32)
        self._codes = {}
        self.scratch = None
        self.maxnv = int(max(1, np.max(np.diff(self.vptr))))
        mean, var = col_moments(self.XT, self.c / self.N)
        self.norm = np.sqrt(np.maximum(var, 0.0) * self.N)  # centred column norm, as in FasterRisk
        self.valid = self.norm > 1e-9
        self.scale = np.where(self.valid, 1.0 / np.maximum(self.norm, 1e-12), 0.0)
        self.lb = -self.bound * np.ones(self.d)
        self.ub = self.bound * np.ones(self.d)


# ------------------------------------------------------------- beam search
@nb.njit(cache=NUMBA_CACHE)
def newton_fit(Z, c, w, lo, hi, maxit, tol):
    """Projected Newton for min sum_g c_g log(1 + exp(-Z_g . w)) with box bounds; w updated in place.
    Returns (loss, margins)."""
    n, p = Z.shape
    m = np.empty(n)
    for i in range(n):
        s = 0.0
        for q in range(p):
            s += Z[i, q] * w[q]
        m[i] = s
    cur = total_loss(m, c)
    g = np.empty(p)
    H = np.empty((p, p))
    wn = np.empty(p)
    mn = np.empty(n)
    for _ in range(maxit):
        g[:] = 0.0
        H[:, :] = 0.0
        for i in range(n):
            pr = _sig(-m[i])
            gi = -c[i] * pr
            hi_ = c[i] * pr * (1.0 - pr)
            for q in range(p):
                zq = Z[i, q]
                g[q] += gi * zq
                hz = hi_ * zq
                for r in range(q, p):
                    H[q, r] += hz * Z[i, r]
        free = np.ones(p, np.bool_)
        for q in range(p):
            if (w[q] <= lo[q] + 1e-12 and g[q] > 0) or (w[q] >= hi[q] - 1e-12 and g[q] < 0):
                free[q] = False
        fi = np.flatnonzero(free)
        nf = fi.shape[0]
        d = np.zeros(p)
        if nf > 0:
            A = np.empty((nf, nf))
            bb = np.empty(nf)
            for a in range(nf):
                bb[a] = -g[fi[a]]
                for b in range(nf):
                    qa, qb = fi[a], fi[b]
                    A[a, b] = H[min(qa, qb), max(qa, qb)]
                A[a, a] += 1e-10 * (1.0 + A[a, a])
            sol = np.linalg.solve(A, bb)
            for a in range(nf):
                d[fi[a]] = sol[a]
        t = 1.0
        new = cur
        ok = False
        while t > 1e-8:
            for q in range(p):
                v = w[q] + t * d[q]
                wn[q] = min(max(v, lo[q]), hi[q])
            for i in range(n):
                s = 0.0
                for q in range(p):
                    s += Z[i, q] * wn[q]
                mn[i] = s
            new = total_loss(mn, c)
            if new <= cur:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        dec = cur - new
        w[:] = wn
        m[:] = mn
        cur = new
        if dec <= tol * cur:
            break
    return cur, m


@nb.njit(cache=NUMBA_CACHE)
def regroup(ginv, ng, code, ncode, y, c):
    """Refine row groups by a column's value code. Returns (inv, ng2, group y, group weight, representative row)."""
    n = ginv.shape[0]
    inv = np.empty(n, np.int64)
    M = ng * ncode
    cnt = 0
    if M <= 4 * n + 4096:
        table = np.full(M, -1, np.int64)
        for i in range(n):
            key = ginv[i] * ncode + code[i]
            t = table[key]
            if t < 0:
                t = cnt
                table[key] = t
                cnt += 1
            inv[i] = t
    else:
        keys = np.empty(n, np.int64)
        for i in range(n):
            keys[i] = ginv[i] * ncode + code[i]
        order = np.argsort(keys)
        last = -1
        for t in range(n):
            i = order[t]
            if t == 0 or keys[i] != last:
                cnt += 1
                last = keys[i]
            inv[i] = cnt - 1
    gy = np.empty(cnt)
    gc = np.zeros(cnt)
    rep = np.full(cnt, -1, np.int64)
    for i in range(n):
        g = inv[i]
        gc[g] += c[i]
        if rep[g] < 0:
            rep[g] = i
            gy[g] = y[i]
    return inv, cnt, gy, gc, rep


class Node:
    """A beam state: support, continuous (b0, beta) and the rows grouped by (y, x_support)."""
    __slots__ = ("loss", "w", "S", "inv", "ng", "gy", "gc", "rep", "mg", "par", "j")


@nb.njit(cache=NUMBA_CACHE)
def group_design(X, gy, rep, S):
    ng = rep.shape[0]
    p = S.shape[0] + 1
    Z = np.empty((ng, p))
    for g in range(ng):
        Z[g, 0] = gy[g]
        for q in range(1, p):
            Z[g, q] = gy[g] * X[rep[g], S[q - 1]]
    return Z


@nb.njit(cache=NUMBA_CACHE)
def child_fit_sparse(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, j, ptr, idx, xcode, vptr, vals, c, X,
                     bound, tol):
    """Fit a child (parent support + column j) without touching every row: the child's groups are the parent's
    groups split by the value of x_j, and their weights come from the nonzeros of column j only."""
    nv = vptr[j + 1] - vptr[j] + 1  # slot 0: x_j = 0
    W = np.zeros(par_ng * nv)
    for t in range(ptr[j], ptr[j + 1]):
        i = idx[t]
        W[par_inv[i] * nv + 1 + xcode[t]] += c[i]
    for g in range(par_ng):
        rest = par_gc[g]
        for v in range(1, nv):
            rest -= W[g * nv + v]
        W[g * nv] = rest if rest > 0.5 else 0.0
    ncell = 0
    for g in range(par_ng):
        for v in range(nv):
            if W[g * nv + v] > 0.0:
                ncell += 1
    ps = par_S.shape[0]
    S = np.empty(ps + 1, np.int64)
    w = np.zeros(ps + 2)
    w[0] = par_w[0]
    pos = 0
    q = 0
    ins = False
    for r in range(ps + 1):
        if not ins and (q >= ps or j < par_S[q]):
            S[r] = j
            pos = r
            ins = True
        else:
            S[r] = par_S[q]
            w[r + 1] = par_w[q + 1]
            q += 1
    Z = np.empty((ncell, ps + 2))
    cw = np.empty(ncell)
    u = 0
    for g in range(par_ng):
        for v in range(nv):
            wt = W[g * nv + v]
            if wt <= 0.0:
                continue
            xj = 0.0 if v == 0 else vals[vptr[j] + v - 1]
            yg = par_gy[g]
            Z[u, 0] = yg
            for r in range(ps + 1):
                if r == pos:
                    Z[u, r + 1] = yg * xj
                else:
                    Z[u, r + 1] = yg * X[par_rep[g], S[r]]
            cw[u] = wt
            u += 1
    lo = np.full(ps + 2, -bound)
    hi = np.full(ps + 2, bound)
    lo[0] = -1e300
    hi[0] = 1e300
    loss, _ = newton_fit(Z, cw, w, lo, hi, 50, tol)
    return S, loss, w


@nb.njit(cache=NUMBA_CACHE)
def child_fit_batch(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, js, ptr, idx, xcode, vptr, vals, c, X,
                    bound, tol):
    """child_fit_sparse for several columns of one parent: losses and fitted (b0, beta_S) per child."""
    m = js.shape[0]
    losses = np.empty(m)
    W = np.empty((m, par_S.shape[0] + 2))
    for u in range(m):
        _, l, w = child_fit_sparse(par_inv, par_ng, par_gy, par_gc, par_rep, par_S, par_w, js[u], ptr, idx, xcode,
                                   vptr, vals, c, X, bound, tol)
        losses[u] = l
        W[u] = w
    return losses, W


@nb.njit(cache=NUMBA_CACHE)
def make_child(par_inv, par_ng, par_S, par_w, j, colcode_j, ncode_j, y, c, X, bound, tol):
    """Add column j to a parent: refine its row groups and refit (b0, beta_S) by projected Newton."""
    inv, ng, gy, gc, rep = regroup(par_inv, par_ng, colcode_j, ncode_j, y, c)
    ps = par_S.shape[0]
    S = np.empty(ps + 1, np.int64)
    w = np.zeros(ps + 2)
    w[0] = par_w[0]
    q = 0
    ins = False
    for r in range(ps + 1):
        if not ins and (q >= ps or j < par_S[q]):
            S[r] = j
            w[r + 1] = 0.0
            ins = True
        else:
            S[r] = par_S[q]
            w[r + 1] = par_w[q + 1]
            q += 1
    lo = np.full(ps + 2, -bound)
    hi = np.full(ps + 2, bound)
    lo[0] = -1e300
    hi[0] = 1e300
    Z = group_design(X, gy, rep, S)
    loss, mg = newton_fit(Z, gc, w, lo, hi, 50, tol)
    return S, inv, ng, gy, gc, rep, loss, mg, w


def beam_search(D, k, parent_size=10, child_size=10, deadline=np.inf):
    root = Node()
    root.S = ()
    root.inv, root.ng, root.gy, root.gc, root.rep = regroup(np.zeros(D.n, np.int64), 1, (D.y > 0).astype(np.int64), 2, D.y, D.c)
    npos = D.c[D.y > 0].sum()
    root.w = np.array([np.log(npos / (D.N - npos))])
    root.mg = root.gy * root.w[0]
    root.loss = total_loss(root.mg, root.gc)
    parents = [root]
    seen = set()
    bound = float(D.bound)
    for _ in range(min(k, int(D.valid.sum()))):
        if time.perf_counter() > deadline:
            parents = parents[:1]  # out of time: finish the support greedily from the best parent
        children = []
        # |gradient| of every column for all parents at once (one BLAS mat-mat), scaled by the centred norm
        R = np.empty((D.n, len(parents)))
        for q, par in enumerate(parents):
            R[:, q] = D.yc / (1.0 + np.exp(par.mg[par.inv]))
        Gall = np.abs(D.XT @ R) * D.scale[:, None]
        for q, par in enumerate(parents):
            g = Gall[:, q]
            if par.S:
                g[list(par.S)] = -1
            cand = np.argsort(-g)[:child_size]
            cand = cand[g[cand] > 0]
            new_js, keys = [], []
            for j in cand:
                key = tuple(sorted(par.S + (int(j),)))
                if key not in seen:
                    seen.add(key)
                    new_js.append(int(j))
                    keys.append(key)
            if not new_js:
                continue
            par_S = np.array(par.S, dtype=np.int64)
            js = np.array(new_js, dtype=np.int64)
            sparse = ~D.raw[js]
            if sparse.any():
                # children fitted from the parent's groups split by x_j (no pass over all rows)
                losses, W = child_fit_batch(par.inv, par.ng, par.gy, par.gc, par.rep, par_S, par.w, js[sparse],
                                            D.ptr, D.idx, D.xcode, D.vptr, D.vals, D.c, D.X, bound, 1e-7)
                for u, q in enumerate(np.flatnonzero(sparse)):
                    ch = Node()
                    ch.loss, ch.w, ch.S, ch.inv, ch.par, ch.j = losses[u], W[u], keys[q], None, par, new_js[q]
                    children.append(ch)
            for q in np.flatnonzero(~sparse):
                j = new_js[q]
                ch = Node()
                (S, ch.inv, ch.ng, ch.gy, ch.gc, ch.rep, ch.loss, ch.mg, ch.w) = make_child(
                    par.inv, par.ng, par_S, par.w, j, D.code(j), D.ncode[j], D.y, D.c, D.X, bound, 1e-7)
                ch.S = keys[q]
                children.append(ch)
        if not children:
            break
        children.sort(key=lambda t: t.loss)
        parents = children[:parent_size]
        for ch in parents:
            if ch.inv is None:  # materialise the row groups of the selected children only
                par, j = ch.par, ch.j
                ch.inv, ch.ng, ch.gy, ch.gc, ch.rep = regroup(par.inv, par.ng, D.code(j), D.ncode[j], D.y, D.c)
                ch.mg = group_design(D.X, ch.gy, ch.rep, np.array(ch.S, dtype=np.int64)) @ ch.w
                ch.par = None
    return parents


# ------------------------------------------------------- calibrated rounding
@nb.njit(cache=NUMBA_CACHE)
def calib_round_kernel(XS, gy, gc, beta, n_mult, bound):
    """Round m * beta for a grid of scales (largest point 0.5 .. bound + 0.49); return the rounding with the
    smallest calibrated loss on the row groups (XS: group rows of the support columns)."""
    ng, p = XS.shape
    top = 0.0
    for q in range(p):
        top = max(top, abs(beta[q]))
    best_l = np.inf
    best_r = np.zeros(p)
    prev = np.full(p, np.nan)
    r = np.empty(p)
    sg = np.empty(ng)
    if top < 1e-12:
        return best_r, best_l
    for t in range(n_mult):
        L = 0.5 + (bound - 0.01) * t / max(n_mult - 1, 1)
        same = True
        anynz = False
        for q in range(p):
            v = np.round(beta[q] * L / top)
            v = min(max(v, -bound), bound)
            r[q] = v
            if v != prev[q]:
                same = False
            if v != 0:
                anynz = True
        if same or not anynz:
            continue
        prev[:] = r
        mn = np.inf
        mx = -np.inf
        m1 = 0.0
        m2 = 0.0
        for g in range(ng):
            v = 0.0
            for q in range(p):
                v += XS[g, q] * r[q]
            sg[g] = v
            mn = min(mn, v)
            mx = max(mx, v)
            m1 += gc[g] * v
            m2 += gc[g] * v * v
        if mx == mn:
            continue
        tot = gc.sum()
        sd = np.sqrt(max(m2 / tot - (m1 / tot) ** 2, 1e-300))
        for g in range(ng):
            sg[g] /= sd
        loss, _, _ = calibrate(sg, gy, gc, 0.0, 0.0)
        if loss < best_l:
            best_l = loss
            best_r[:] = r
    return best_r, best_l


def calib_round(D, nd, n_mult=20):
    """Best rounding of the node's beta by the calibrated loss over a grid of scales."""
    S = np.array(nd.S, dtype=np.int64)
    XS = group_design(D.X, np.ones(nd.ng), nd.rep, S)[:, 1:]
    r, l = calib_round_kernel(np.ascontiguousarray(XS), nd.gy, nd.gc, nd.w[1:].copy(), n_mult, float(D.bound))
    w = np.zeros(D.d)
    w[S] = r
    return w, l / D.N


# ---------------------------------------------------- calibrated integer loss
@nb.njit(cache=NUMBA_CACHE)
def calibrate(s, y, c, a, b):
    """min over (a, b) of sum c log(1 + exp(-y (a s + b))), damped Newton from (a, b)."""
    n = s.shape[0]
    cur = 0.0
    for i in range(n):
        cur += c[i] * _lrow(y[i] * (a * s[i] + b))
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for i in range(n):
            z = y[i] * (a * s[i] + b)
            p = _sig(-z)
            w = c[i] * p * (1.0 - p)
            gi = -c[i] * y[i] * p
            ga += gi * s[i]
            gb += gi
            haa += w * s[i] * s[i]
            hab += w * s[i]
            hbb += w
        haa += 1e-12
        hbb += 1e-12
        det = haa * hbb - hab * hab
        if det <= 1e-18 * (haa * hbb + 1e-300):
            da = 0.0
            db = gb / hbb
        else:
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
        dec = ga * da + gb * db
        t = 1.0
        new = cur
        while t > 1e-10:
            na, nb_ = a - t * da, b - t * db
            new = 0.0
            for i in range(n):
                new += c[i] * _lrow(y[i] * (na * s[i] + nb_))
            if new <= cur - 1e-4 * t * dec:
                break
            t *= 0.5
        if t <= 1e-10:
            break
        a, b = na, nb_
        improv = cur - new
        cur = new
        if improv < 1e-12 * (1.0 + cur):
            break
    return cur, a, b


# ------------------------------------------------ integer local search (ILS)
@nb.njit(cache=NUMBA_CACHE)
def bin_scores(s, y, c):
    """Group rows by distinct score: inv (row -> bin), bin scores sv, weights of y=+1 (Wp) and y=-1 (Wn)."""
    n = s.shape[0]
    lo = s[0]
    hi = s[0]
    integral = True
    for i in range(n):
        v = s[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        if integral and v != np.floor(v):
            integral = False
    if integral and hi - lo <= 4 * n + 1024:
        # integer scores in a small range: counting instead of sorting
        R = int(hi - lo) + 1
        cid = np.full(R, -1, np.int64)
        for i in range(n):
            cid[int(s[i] - lo)] = 0
        nb_ = 0
        for r in range(R):
            if cid[r] >= 0:
                cid[r] = nb_
                nb_ += 1
        inv = np.empty(n, np.int64)
        sv = np.empty(nb_)
        Wp = np.zeros(nb_)
        Wn = np.zeros(nb_)
        for r in range(R):
            if cid[r] >= 0:
                sv[cid[r]] = lo + r
        for i in range(n):
            q = cid[int(s[i] - lo)]
            inv[i] = q
            if y[i] > 0:
                Wp[q] += c[i]
            else:
                Wn[q] += c[i]
        return inv, sv, Wp, Wn
    order = np.argsort(s)
    inv = np.empty(n, np.int64)
    sv = np.empty(n)
    Wp = np.zeros(n)
    Wn = np.zeros(n)
    nb_ = -1
    last = 0.0
    for t in range(n):
        i = order[t]
        if t == 0 or s[i] != last:
            nb_ += 1
            sv[nb_] = s[i]
            last = s[i]
        inv[i] = nb_
        if y[i] > 0:
            Wp[nb_] += c[i]
        else:
            Wn[nb_] += c[i]
    nb_ += 1
    return inv, sv[:nb_].copy(), Wp[:nb_].copy(), Wn[:nb_].copy()


@nb.njit(cache=NUMBA_CACHE)
def calibrate_bins(sv, Wp, Wn, a, b):
    m = sv.shape[0]
    s2 = np.empty(2 * m)
    y2 = np.empty(2 * m)
    c2 = np.empty(2 * m)
    for i in range(m):
        s2[i] = sv[i]
        y2[i] = 1.0
        c2[i] = Wp[i]
        s2[m + i] = sv[i]
        y2[m + i] = -1.0
        c2[m + i] = Wn[i]
    return calibrate(s2, y2, c2, a, b)


@nb.njit(cache=NUMBA_CACHE)
def bin_stats(sv, Wp, Wn, a, b, lp, ln_, pp, pn, wq):
    T = np.zeros(6)
    for q in range(sv.shape[0]):
        z = a * sv[q] + b
        e = np.exp(-abs(z))
        l1 = np.log1p(e)
        if z > 0:
            lp[q] = l1
            ln_[q] = z + l1
            pp[q] = e / (1.0 + e)
            pn[q] = 1.0 / (1.0 + e)
        else:
            lp[q] = -z + l1
            ln_[q] = l1
            pp[q] = 1.0 / (1.0 + e)
            pn[q] = e / (1.0 + e)
        wq[q] = pp[q] * pn[q]
        g = -Wp[q] * pp[q] + Wn[q] * pn[q]
        wt = (Wp[q] + Wn[q]) * wq[q]
        T[0] += Wp[q] * lp[q] + Wn[q] * ln_[q]
        T[1] += g * sv[q]
        T[2] += g
        T[3] += wt * sv[q] * sv[q]
        T[4] += wt * sv[q]
        T[5] += wt
    return T


@nb.njit(cache=NUMBA_CACHE)
def eval_binned(cols, deltas, inv, sv, lp, ln_, pp, pn, wq, T, a, b, y, c, ptr, idx, xv, xcode, vptr, vals,
                raw, out, sp, sn, touched, tr_a, tr_b):
    """out[q, r] = estimated calibrated loss after s += deltas[q, r] * x_cols[q]. Rows are grouped into cells
    (score bin, value of x_j); each cell costs one exp per delta. The estimate is the exact loss at the
    current (a, b) minus a trust-region Newton step in (a, b)."""
    nd = deltas.shape[1]
    cb = np.empty(touched.shape[0], np.int64)
    cx = np.empty(touched.shape[0])
    cwp = np.empty(touched.shape[0])
    cwn = np.empty(touched.shape[0])
    stepa = np.empty(nd)
    stepb = np.empty(nd)
    hyb = np.empty(nd)
    dL = np.empty(nd)
    dGa = np.empty(nd)
    dGb = np.empty(nd)
    dHaa = np.empty(nd)
    dHab = np.empty(nd)
    dHbb = np.empty(nd)
    for q in range(cols.shape[0]):
        j = cols[q]
        dL[:] = 0.0
        dGa[:] = 0.0
        dGb[:] = 0.0
        dHaa[:] = 0.0
        dHab[:] = 0.0
        dHbb[:] = 0.0
        nt = 0
        if raw[j]:
            for t in range(ptr[j], ptr[j + 1]):
                touched[nt] = t
                nt += 1
        else:
            nv = vptr[j + 1] - vptr[j]
            for t in range(ptr[j], ptr[j + 1]):
                i = idx[t]
                key = inv[i] * nv + xcode[t]
                if sp[key] == 0.0 and sn[key] == 0.0:
                    touched[nt] = key
                    nt += 1
                if y[i] > 0:
                    sp[key] += c[i]
                else:
                    sn[key] += c[i]
        # gather the touched cells: bin, x value, weights of y = +1 / -1
        for u in range(nt):
            if raw[j]:
                t = touched[u]
                i = idx[t]
                cb[u] = inv[i]
                cx[u] = xv[t]
                cwp[u] = c[i] if y[i] > 0 else 0.0
                cwn[u] = c[i] - cwp[u]
            else:
                key = touched[u]
                nv = vptr[j + 1] - vptr[j]
                bq = key // nv
                cb[u] = bq
                cx[u] = vals[vptr[j] + key - bq * nv]
                cwp[u] = sp[key]
                cwn[u] = sn[key]
                sp[key] = 0.0
                sn[key] = 0.0
        oL = oGa = oGb = oHaa = oHab = oHbb = 0.0
        for u in range(nt):
            bq = cb[u]
            x = cx[u]
            wpc = cwp[u]
            wnc = cwn[u]
            so = sv[bq]
            lo = wpc * lp[bq] + wnc * ln_[bq]
            go = -wpc * pp[bq] + wnc * pn[bq]
            wo = (wpc + wnc) * wq[bq]
            oL += lo
            oGa += go * so
            oGb += go
            oHaa += wo * so * so
            oHab += wo * so
            oHbb += wo
            for r in range(nd):
                sn_ = so + deltas[q, r] * x
                z = a * sn_ + b
                e = np.exp(-abs(z))
                l1 = np.log1p(e)
                if z > 0:
                    lpn = l1
                    lnn = z + l1
                    ppn = e / (1.0 + e)
                    pnn = 1.0 / (1.0 + e)
                else:
                    lpn = -z + l1
                    lnn = l1
                    ppn = 1.0 / (1.0 + e)
                    pnn = e / (1.0 + e)
                gn = -wpc * ppn + wnc * pnn
                wn = (wpc + wnc) * ppn * pnn
                dL[r] += wpc * lpn + wnc * lnn - lo
                dGa[r] += gn * sn_ - go * so
                dGb[r] += gn - go
                dHaa[r] += wn * sn_ * sn_ - wo * so * so
                dHab[r] += wn * sn_ - wo * so
                dHbb[r] += wn - wo
        for r in range(nd):
            ga = T[1] + dGa[r]
            gb = T[2] + dGb[r]
            haa = T[3] + dHaa[r] + 1e-12
            hab = T[4] + dHab[r]
            hbb = T[5] + dHbb[r] + 1e-12
            det = haa * hbb - hab * hab
            if det > 1e-14 * haa * hbb:
                da = (hbb * ga - hab * gb) / det
                db = (haa * gb - hab * ga) / det
            else:
                da = 0.0
                db = gb / hbb
            t = 1.0
            if abs(da) * t > tr_a * abs(a) + 1e-300:
                t = tr_a * abs(a) / abs(da)
            if abs(db) * t > tr_b:
                t = tr_b / abs(db)
            stepa[r] = -t * da
            stepb[r] = -t * db
            out[1, q, r] = T[0] + dL[r]
            hyb[r] = 0.0
        # the untouched rows at the Newton point, by their quadratic model around (a, b)
        UGa = T[1] - oGa
        UGb = T[2] - oGb
        UHaa = T[3] - oHaa
        UHab = T[4] - oHab
        UHbb = T[5] - oHbb
        for r in range(nd):
            hyb[r] = (T[0] - oL + UGa * stepa[r] + UGb * stepb[r]
                      + 0.5 * (UHaa * stepa[r] * stepa[r] + 2 * UHab * stepa[r] * stepb[r] + UHbb * stepb[r] * stepb[r]))
        # the touched cells exactly at the Newton point
        for u in range(nt):
            so = sv[cb[u]]
            x = cx[u]
            wpc = cwp[u]
            wnc = cwn[u]
            for r in range(nd):
                z = (a + stepa[r]) * (so + deltas[q, r] * x) + b + stepb[r]
                l1 = np.log1p(np.exp(-abs(z)))
                if z > 0:
                    hyb[r] += wpc * l1 + wnc * (z + l1)
                else:
                    hyb[r] += wpc * (l1 - z) + wnc * l1
        for r in range(nd):
            out[0, q, r] = min(out[1, q, r], hyb[r])
    return out


class ScoreState:
    """Scores s = X w grouped into bins, with the calibrated (a, b) and per-bin statistics."""

    def __init__(self, D, s, a=None, b=None):
        self.s = s
        self.inv, self.sv, self.Wp, self.Wn = bin_scores(s, D.y, D.c)
        if a is None:
            if np.ptp(s) == 0:
                npos = D.c[D.y > 0].sum()
                self.L, self.a, self.b = calibrate_bins(self.sv, self.Wp, self.Wn, 0.0, np.log(npos / (D.N - npos)))
            else:
                sd = s.std()
                L, a, b = calibrate_bins(self.sv / sd, self.Wp, self.Wn, 0.0, 0.0)
                self.L, self.a, self.b = L, a / sd, b
        elif a == "keep":
            pass
        else:
            self.L, self.a, self.b = calibrate_bins(self.sv, self.Wp, self.Wn, a, b)

    def stats(self, a, b):
        m = self.sv.shape[0]
        self.lp, self.ln, self.pp, self.pn, self.wq = (np.empty(m) for _ in range(5))
        self.T = bin_stats(self.sv, self.Wp, self.Wn, a, b, self.lp, self.ln, self.pp, self.pn, self.wq)

    def screen_rows(self, D):
        """Per-row first and second derivative of the loss in the score at the current (a, b)."""
        return np.where(D.y > 0, -self.pp[self.inv], self.pn[self.inv]) * D.c, self.wq[self.inv] * D.c

    def eval(self, D, cols, deltas, a, b, maxnv, n_screen=None, GH=None):
        """deltas: one row of score changes per column, or one row shared by all columns."""
        if deltas.ndim == 1:
            deltas = np.ascontiguousarray(np.broadcast_to(deltas, (len(cols), len(deltas))))
        if n_screen is not None and len(cols) > n_screen:
            # rank columns by a second-order model at fixed (a, b) (no exp per row), evaluate the best
            if GH is None:
                r_row, h_row = self.screen_rows(D)
                GH = ((D.XT @ r_row)[cols], (D.XT2 @ h_row)[cols])
            G, H = GH
            u = a * deltas
            sc = np.min(np.minimum(0.0, u * G[:, None] + 0.5 * (u * u) * H[:, None]), axis=1)
            pick = np.sort(np.argsort(sc)[:n_screen])
            out = np.full((2,) + deltas.shape, np.inf)
            out[:, pick] = self.eval(D, cols[pick], deltas[pick], a, b, maxnv)
            return out
        out = np.empty((2,) + deltas.shape)  # [estimated calibrated loss, loss at fixed (a, b)]
        size = max(self.sv.shape[0] * maxnv, 1)
        if D.scratch is None or D.scratch[0].shape[0] < size:
            # zeroed scratch shared by all evaluations (the kernel resets every cell it touches)
            m = max(size, D.n)
            D.scratch = (np.zeros(m), np.zeros(m), np.empty(m, np.int64))
        sp, sn, touched = D.scratch
        eval_binned(cols, deltas, self.inv, self.sv, self.lp, self.ln, self.pp, self.pn, self.wq, self.T, a, b,
                    D.y, D.c, D.ptr, D.idx, D.xval, D.xcode, D.vptr, D.vals, D.raw, out, sp, sn, touched, TR_A, TR_B)
        return out


class ILS:
    """Best-improvement local search over integer points, scored by the calibrated loss."""

    def __init__(self, D, k, n_exact=2, n_screen=4, n_rank=1):
        self.D, self.k, self.n_exact, self.n_screen = D, k, n_exact, n_screen
        self.visited = set()
        self.n_rank = n_rank
        self.est_margin = 1e-4
        self.nevals = 0

    def check(self, st, w, cands, a, b):
        """Exact calibrated loss of the best estimated candidates; returns the best improving one or None."""
        D = self.D
        m = self.n_exact * 2
        by_est = sorted(cands, key=lambda t: t[0])[:m]
        by_bound = sorted(cands, key=lambda t: t[4])[:m] if self.n_rank > 1 else []
        todo, seen = [], set()
        for t in by_est + by_bound:
            if (t[1], t[2], t[3]) not in seen:
                seen.add((t[1], t[2], t[3]))
                todo.append(t)
        best = None
        for est, rj, aj, v, _ in todo:
            if est > st.L * (1.0 + self.est_margin):
                continue  # the estimate says the move does not help
            s2 = st.s.copy()
            if rj >= 0:
                s2 -= w[rj] * D.XT[rj]
            s2 += (v - w[aj]) * D.XT[aj]
            if np.ptp(s2) == 0:
                continue
            self.nevals += 1
            st2 = ScoreState(D, s2, a, b)
            if st2.L < st.L - 1e-9 * st.L and (best is None or st2.L < best[0].L):
                best = (st2, rj, aj, v)
        return best

    def run(self, w, max_iter=100, deadline=np.inf):
        D, k = self.D, self.k
        w = w.astype(np.float64).copy()
        st = ScoreState(D, D.X @ w)
        allv = np.arange(-D.bound, D.bound + 1, dtype=np.float64)
        nzv = allv[allv != 0]
        shifts = np.arange(-2 * D.bound, 2 * D.bound + 1, dtype=np.float64)
        for _ in range(max_iter):
            key = w.tobytes()
            if key in self.visited or time.perf_counter() > deadline:
                break  # the search from here is deterministic and was already done
            self.visited.add(key)
            a, b = st.a, st.b
            st.stats(a, b)
            S = np.flatnonzero(w)
            cands = []  # (est, remove_j, add_j, new_value_of_add_j)
            ne = self.n_exact

            def top(out, rj, cols, vals):
                # best moves by the estimate and by the upper bound (loss at the current (a, b))
                picked = set()
                for o in out[:self.n_rank]:
                    flat = o.ravel()
                    m = min(ne, flat.size)
                    for f in np.argpartition(flat, m - 1)[:m]:
                        f = int(f)
                        if f in picked or not np.isfinite(flat[f]):
                            continue
                        picked.add(f)
                        q, r = divmod(f, o.shape[1])
                        cands.append((out[0].flat[f], rj, cols[q], vals[r], out[1].flat[f]))

            if len(S):
                # value changes of support features: every other value in [-5, 5] (0 removes the feature)
                newv = np.array([allv[allv != w[j]] for j in S])
                out = st.eval(D, S, newv - w[S][:, None], a, b, D.maxnv)
                for q, j in enumerate(S):
                    top(out[:, q:q + 1], -1, np.array([j]), newv[q])
            nonS = np.flatnonzero((w == 0) & D.valid)
            if len(S) < k and len(nonS):
                top(st.eval(D, nonS, nzv, a, b, D.maxnv), -1, nonS, nzv)
            best = self.check(st, w, cands, a, b)
            if best is None and len(nonS):
                # only when no value change / addition helps: swaps (remove j, add j2 with a value)
                cands = []
                sts = []
                R = np.empty((D.n, 2 * len(S)))
                for q, j in enumerate(S):
                    st2 = ScoreState(D, st.s - w[j] * D.XT[j], "keep")
                    st2.stats(a, b)
                    sts.append(st2)
                    R[:, 2 * q], R[:, 2 * q + 1] = st2.screen_rows(D)
                # second-order screen of the swap-in columns for all removals at once (one BLAS mat-mat)
                GH = D.XT @ R if D.XT2 is D.XT else None
                for q, j in enumerate(S):
                    if GH is not None:
                        G, H = GH[nonS, 2 * q], GH[nonS, 2 * q + 1]
                    else:
                        G, H = D.XT[nonS] @ R[:, 2 * q], D.XT2[nonS] @ R[:, 2 * q + 1]
                    top(sts[q].eval(D, nonS, nzv, a, b, D.maxnv, self.n_screen, GH=(G, H)), j, nonS, nzv)
                best = self.check(st, w, cands, a, b)
            if best is None:
                break
            st, rj, aj, v = best
            if rj >= 0:
                w[rj] = 0
            w[aj] = v
        return st.L / D.N, w


# ------------------------------------------------------------------ model
def solve(X, y, k, bound=COEF_BOUND, time_limit=60.0, parent_size=10, child_size=10, n_starts=5):
    """Integer points for the columns of ``X`` (y in {0, 1}): at most ``k`` nonzero, each in
    [-bound, bound]. Returns (points, calibrated mean log loss, seconds per stage: data, beam,
    rounding, local search, stopped_early)."""
    tm = [time.perf_counter()]
    t0 = tm[0]
    X = np.asarray(X, dtype=float)
    D = Data(X, y, bound)
    tm.append(time.perf_counter())
    parents = beam_search(D, k, parent_size, child_size, t0 + 0.4 * time_limit)
    tm.append(time.perf_counter())
    seen = set()
    starts = []
    best_w, best_l = np.zeros(D.d), np.inf
    for nd in parents:
        w, l = calib_round(D, nd)
        key = w.tobytes()
        if key in seen or not np.isfinite(l):
            continue
        seen.add(key)
        starts.append((l, w))
        if l < best_l:
            best_l, best_w = l, w
    tm.append(time.perf_counter())
    # integer local search from the best few distinct rounded solutions
    ils = ILS(D, k)
    starts.sort(key=lambda t: t[0])
    stopped = False
    for l0, w0 in starts[:n_starts]:
        if time.perf_counter() > t0 + 0.8 * time_limit:
            stopped = True
            break
        l, w = ils.run(w0, deadline=t0 + 0.9 * time_limit)
        if l < best_l:
            best_l, best_w = l, w
    tm.append(time.perf_counter())
    stopped = stopped or tm[-1] - t0 > 0.9 * time_limit
    return np.clip(np.round(best_w), -bound, bound), float(best_l), np.diff(tm), stopped
