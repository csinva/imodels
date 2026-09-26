"""Branch-and-bound engine behind `FastSmallTreeClassifier`.

The public class lives in `fast_small_tree.py`; everything here is the solver it
calls. The objective is the one from `GOSDT
<https://arxiv.org/abs/2006.08690>`_ (Lin et al., ICML 2020),

    minimise   misclassification rate + regularization * (number of leaves)

over every decision tree on the binarized features. The search is a memoised
branch-and-bound over capture sets, where a capture set (the subset of training
rows reaching a node) is a bitmask over packed 64-bit words. Each subproblem
carries an interval ``[lb, ub]`` on its optimal risk, and the recursion either
closes that interval or proves it exceeds the budget it was given, so a returned
tree is certified optimal rather than merely the best one found.

Every bound here is admissible: they change which subproblems are visited, never
which tree is optimal. The ones that do the work are the equivalent-points and
leaf-support bounds of the original paper, a MurTree-style pairwise stage that
gives each child its exact best two-leaf tree, a shape relaxation that bounds
every tree with four or more leaves, and a similar-support bound propagated
along each numeric column.

The whole search is one numba function rather than a Python recursion: nodes are
rows of arrays behind an open-addressing index, and the recursion is an explicit
stack of frames with a phase machine for the candidate loop, since numba cannot
link a self-recursive function of this size. Python re-enters it in short
chunks to enforce the time and memory limits, so an interrupted search still
returns the best tree it found. `CompiledOptimizer` is that engine and
`Optimizer` is an equivalent pure-Python one, kept because it is far easier to
read and is what the exactness tests check the compiled engine against.

Reference implementation and derivations: https://github.com/csinva/agentic-imodels
"""

import importlib.util
import json
import os
import queue
import subprocess
import threading
import time

import numpy as np
import pandas as pd

EPS = 1e-10

#: numba is optional for importing imodels but required to fit this model: the
#: search itself is compiled, and interpreting it costs orders of magnitude.
#: `FastSmallTreeClassifier.fit` raises with an install hint when it is missing;
#: `njit` falls back to a no-op decorator so this module still imports.
HAVE_NUMBA = importlib.util.find_spec("numba") is not None

if HAVE_NUMBA:
    from numba import njit
else:
    def njit(*args, **kwargs):
        """Stand-in decorator so the module imports without numba installed."""
        def decorate(func):
            return func
        return decorate(args[0]) if args and callable(args[0]) else decorate

#: Compiling the search takes ~15 s, so the result is cached on disk and later
#: processes load it in a second or two. Set OPTTREE_NUMBA_CACHE=0 to disable,
#: which is what to do if the cache directory is read-only or shared oddly.
NUMBA_CACHE = os.environ.get("OPTTREE_NUMBA_CACHE", "1") != "0"

# ------------------------------------------------------------------ fastbits

@njit(cache=NUMBA_CACHE, nogil=True)
def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=NUMBA_CACHE, nogil=True)
def child_counts(F, masks, out):
    """out[j, r] = popcount(F[j] & masks[r]) for every feature j and mask r."""
    m, W = F.shape
    R = masks.shape[0]
    for j in range(m):
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[j, w] & masks[r, w])
            out[j, r] = acc
    return out


@njit(cache=NUMBA_CACHE, nogil=True)
def child_counts_subset(F, feats, masks, out):
    """Same as ``child_counts`` restricted to the rows ``feats`` of ``F``."""
    W = F.shape[1]
    R = masks.shape[0]
    for t in range(feats.shape[0]):
        j = feats[t]
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[j, w] & masks[r, w])
            out[t, r] = acc
    return out


@njit(cache=NUMBA_CACHE, nogil=True)
def segment_dp(S, costs, lam, best, back):
    """Optimal segmentation of ``M`` ordered bins into contiguous segments.

    ``S[i, k]`` is the cumulative count of class ``k`` in bins ``< i`` (``S[0] = 0``),
    ``costs[p, k]`` the cost of predicting ``p`` for a point of class ``k``.  A segment
    ``(j, i)`` costs its best single prediction plus ``lam``; ``best[i]`` is the
    optimal cost of the first ``i`` bins and ``back[i]`` the start of its last segment.
    """
    M = S.shape[0] - 1
    K = S.shape[1]
    best[0] = 0.0
    back[0] = -1
    for i in range(1, M + 1):
        bi = 1e300
        bj = -1
        for j in range(i):
            c = 1e300
            for p in range(K):
                acc = 0.0
                for k in range(K):
                    acc += costs[p, k] * (S[i, k] - S[j, k])
                if acc < c:
                    c = acc
            v = best[j] + c + lam
            if v < bi:
                bi = v
                bj = j
        best[i] = bi
        back[i] = bj
    return best[M]


@njit(cache=NUMBA_CACHE, nogil=True)
def segment_dp_uniform(S, w, lam, best, back):
    """``segment_dp`` for the uniform cost matrix (``w`` off the diagonal, 0 on it)."""
    M = S.shape[0] - 1
    K = S.shape[1]
    best[0] = 0.0
    back[0] = -1
    for i in range(1, M + 1):
        bi = 1e300
        bj = -1
        for j in range(i):
            tot = 0.0
            mx = 0.0
            for k in range(K):
                d = S[i, k] - S[j, k]
                tot += d
                if d > mx:
                    mx = d
            v = best[j] + w * (tot - mx) + lam
            if v < bi:
                bi = v
                bj = j
        best[i] = bi
        back[i] = bj
    return best[M]


@njit(cache=NUMBA_CACHE, nogil=True)
def node_stats(F, feats, rows, group_of, masks, weights, costs, diff, lam,
               out_feats, out_L, out_l_leaf, out_l_lb, out_l_solved, out_r_leaf, out_r_lb,
               out_r_solved, out_l_pot, out_dist, out_l_pred, out_r_pred, out_pos):
    """Per-split child statistics and bounds for one node, compacted to valid splits.

    ``masks`` holds the node's class masks (first ``K`` rows) followed by the
    equivalent-points masks weighted by ``weights``.  Writes the node's class counts to
    ``out_dist`` and returns the number of valid splits written.
    """
    W = F.shape[1]
    K = out_dist.shape[0] - 1
    R = masks.shape[0]
    nw = weights.shape[0]
    dist = out_dist
    total = 0.0
    total_pot = 0.0
    for k in range(K):
        acc = np.uint64(0)
        for w in range(W):
            acc += _popcount64(masks[k, w])
        dist[k] = acc
        total += dist[k]
        total_pot += diff[k] * dist[k]
    n_min = 0.0
    for r in range(nw):
        acc = np.uint64(0)
        for w in range(W):
            acc += _popcount64(masks[K + r, w])
        n_min += weights[r] * acc
    out_dist[K] = n_min
    nv = 0
    cnt = np.empty(R)
    prev = np.empty(K)
    prev_group = -2
    for t in range(feats.shape[0]):
        j = feats[t]
        fr = rows[t]
        for r in range(R):
            acc = np.uint64(0)
            for w in range(W):
                acc += _popcount64(F[fr, w] & masks[r, w])
            cnt[r] = acc
        # duplicate split: same column as the previous feature and identical class counts
        # (thresholds of one column are nested, so equal sizes mean equal sets)
        g = group_of[j]
        if g >= 0 and g == prev_group:
            same = True
            for k in range(K):
                if cnt[k] != prev[k]:
                    same = False
                    break
            if same:
                continue
        if g >= 0:
            prev_group = g
            for k in range(K):
                prev[k] = cnt[k]
        else:
            prev_group = -2
        lsum = 0.0
        lpot = 0.0
        for k in range(K):
            lsum += cnt[k]
            lpot += diff[k] * cnt[k]
        rpot = total_pot - lpot
        if lsum <= 0.0 or lsum >= total or lpot < lam - EPS or rpot < lam - EPS:
            continue
        rsum = total - lsum
        # leaf risks: best single prediction on each side
        lmax = 1e300
        rmax = 1e300
        lpred = 0
        rpred = 0
        for p in range(K):
            al = 0.0
            ar = 0.0
            for k in range(K):
                al += costs[p, k] * cnt[k]
                ar += costs[p, k] * (dist[k] - cnt[k])
            if al < lmax:
                lmax = al
                lpred = p
            if ar < rmax:
                rmax = ar
                rpred = p
        lmin = 0.0
        for r in range(nw):
            lmin += weights[r] * cnt[K + r]
        rmin = n_min - lmin
        l_leaf = lmax + lam
        r_leaf = rmax + lam
        l_solved = (lsum <= 1.0) or (lmax - lmin < lam) or (lpot < 2.0 * lam)
        r_solved = (rsum <= 1.0) or (rmax - rmin < lam) or (rpot < 2.0 * lam)
        out_feats[nv] = j
        for k in range(K):
            out_L[nv, k] = cnt[k]
        out_l_leaf[nv] = l_leaf
        out_r_leaf[nv] = r_leaf
        out_l_solved[nv] = l_solved
        out_r_solved[nv] = r_solved
        out_l_lb[nv] = l_leaf if l_solved else min(l_leaf, lmin + 2.0 * lam)
        out_r_lb[nv] = r_leaf if r_solved else min(r_leaf, rmin + 2.0 * lam)
        out_l_pot[nv] = lpot
        out_l_pred[nv] = lpred
        out_r_pred[nv] = rpred
        out_pos[nv] = fr
        nv += 1
    return nv


@njit(cache=NUMBA_CACHE, nogil=True)
def _leaf_cost(cnt, costs, K):
    c = 1e300
    for p in range(K):
        acc = 0.0
        for k in range(K):
            acc += costs[p, k] * cnt[k]
        if acc < c:
            c = acc
    return c



@njit(cache=NUMBA_CACHE, nogil=True)
def refilter_candidates(order, n_cand, split_lb, split_ub2, limit):
    """Keep the candidates in order[:n_cand] whose (raised) split_lb is within the limit,
    sorted by (split_lb, split_ub2), compacted in place; returns (count, min dropped lb)."""
    n = 0
    min_dropped = 1e300
    for t in range(n_cand):
        c = order[t]
        v = split_lb[c]
        if v <= limit:
            order[n] = c
            n += 1
        elif v < min_dropped:
            min_dropped = v
    # insertion sort (candidate counts are small)
    for a in range(1, n):
        c = order[a]
        va = split_lb[c]
        ua = split_ub2[c]
        b = a - 1
        while b >= 0:
            d = order[b]
            vb = split_lb[d]
            if vb < va or (vb == va and split_ub2[d] <= ua):
                break
            order[b + 1] = d
            b -= 1
        order[b + 1] = c
    return n, min_dropped


@njit(cache=NUMBA_CACHE, nogil=True)
def shape_bound(l_leaf, ml2_l, c3_l, r_leaf, ml2_r, c3_r, lam, l_lb, r_lb, l_ub2, r_ub2, split_lb, split_ub2):
    """Per-child bounds and the shape-relaxation bound from the pairwise counts.

    g_a(X) lower-bounds every tree with exactly a leaves on child X: g_1 = leaf risk
    (exact), g_2 = 2 lam + best 2-leaf loss (exact), g_3 = 3 lam + the cheapest cell
    any allowed split peels off as a leaf (the other side's loss is >= 0), g_a = a lam
    for a >= 4.  Writes l_ub2/r_ub2 = min(g_1, g_2) (achievable), raises l_lb/r_lb to
    min(g_1, g_2, g_3, 4 lam), split_lb = l_lb + r_lb, split_ub2 = l_ub2 + r_ub2, and
    returns (argmin split_ub2, min split_ub2, lb_ge4) where lb_ge4 = min over splits
    and (a, b) with a + b >= 4 of g_a(left) + g_b(right) bounds every tree with >= 4
    leaves (its root split has a- and b-leaf subtrees; larger a only add lam).
    """
    mf = l_leaf.shape[0]
    four = 4.0 * lam
    best_i = 0
    best_d2 = 1e300
    lb_ge4 = 8.0 * lam
    for i in range(mf):
        f1l = l_leaf[i]; f2l = 2.0 * lam + ml2_l[i]; f3l = 3.0 * lam + c3_l[i]
        f1r = r_leaf[i]; f2r = 2.0 * lam + ml2_r[i]; f3r = 3.0 * lam + c3_r[i]
        ul = min(f1l, f2l); ur = min(f1r, f2r)
        l_ub2[i] = ul; r_ub2[i] = ur
        ll = min(min(ul, f3l), four); rr = min(min(ur, f3r), four)
        if ll > l_lb[i]: l_lb[i] = ll
        if rr > r_lb[i]: r_lb[i] = rr
        split_lb[i] = l_lb[i] + r_lb[i]
        v = ul + ur
        split_ub2[i] = v
        if v < best_d2:
            best_d2 = v; best_i = i
        # a + b >= 4 with a, b <= 3, or one side >= 4 leaves (bounded by 4 lam)
        m3 = min(f2l, f3l)                      # cheapest of the >= 2-leaf options on the left
        g = min(min(f1l + f3r, f2l + f2r), f3l + f1r)
        g = min(g, min(f2l + f3r, f3l + f2r))
        g = min(g, f3l + f3r)
        g = min(g, min(min(f1l, m3), f3l) + four)
        g = min(g, four + min(min(f1r, f2r), f3r))
        if g < lb_ge4:
            lb_ge4 = g
    return best_i, best_d2, lb_ge4

@njit(cache=NUMBA_CACHE, nogil=True)
def depth2_pairs(F, feats, rows, group_of, masks, costs, lam, dist, L, out_ml2_l, out_ml2_r, out_j_l, out_j_r,
                 out_c3_l, out_c3_r):
    """Best 2-leaf loss of the left (feature true) and right child of every candidate split.

    ``masks`` are the node's ``K`` class masks, ``L[i]`` the class counts of the left
    child of split ``feats[i]``.  Pairwise class counts ``|left_i ∩ left_j ∩ class|``
    give the four cells of splitting either child of ``i`` by ``j`` (and of ``j`` by
    ``i``); each unordered pair is counted once.  Thresholds of one column are nested,
    so their intersection is the smaller set and needs no popcount.  A split with an
    empty side is skipped (it is the leaf).  Returns the best depth-2 value's root.
    """
    mf = feats.shape[0]
    W = F.shape[1]
    K = dist.shape[0]
    for i in range(mf):
        out_ml2_l[i] = 1e300
        out_ml2_r[i] = 1e300
        out_j_l[i] = -1
        out_j_r[i] = -1
        out_c3_l[i] = 1e300
        out_c3_r[i] = 1e300
    ij = np.empty(K)
    a = np.empty(K)
    b = np.empty(K)
    fij = np.empty(W, dtype=np.uint64)
    tot = 0.0
    for k in range(K):
        tot += dist[k]
    for i in range(mf):
        fi = feats[i]
        fri = rows[i]
        gi = group_of[fi]
        li = 0.0
        for k in range(K):
            li += L[i, k]
        for j in range(i + 1, mf):
            fj = feats[j]
            frj = rows[j]
            if gi >= 0 and group_of[fj] == gi:
                # nested: j is the higher threshold, so left_j ⊆ left_i
                for k in range(K):
                    ij[k] = L[j, k]
            else:
                for k in range(K):
                    acc = np.uint64(0)
                    for w in range(W):
                        acc += _popcount64(F[fri, w] & F[frj, w] & masks[k, w])
                    ij[k] = acc
            sij = 0.0
            lj = 0.0
            for k in range(K):
                sij += ij[k]
                lj += L[j, k]
            # the four cells of the pair; each cell's leaf cost is also the cost of the
            # leaf peeled off by the first split of a 3-leaf subtree on that child
            rj = lj - sij
            ri = li - sij
            rest = tot - li - lj + sij
            c_ij = 1e300
            c_inj = 1e300
            c_nij = 1e300
            c_rest = 1e300
            if sij > 0.0:
                for k in range(K):
                    a[k] = ij[k]
                c_ij = _leaf_cost(a, costs, K)
            if ri > 0.0:
                for k in range(K):
                    a[k] = L[i, k] - ij[k]
                c_inj = _leaf_cost(a, costs, K)
            if rj > 0.0:
                for k in range(K):
                    a[k] = L[j, k] - ij[k]
                c_nij = _leaf_cost(a, costs, K)
            if rest > 0.0:
                for k in range(K):
                    a[k] = dist[k] - L[i, k] - L[j, k] + ij[k]
                c_rest = _leaf_cost(a, costs, K)
            # left child of i split by j: cells ij and (L[i] - ij)
            if sij > 0.0 and ri > 0.0:
                v = c_ij + c_inj
                if v < out_ml2_l[i]:
                    out_ml2_l[i] = v
                    out_j_l[i] = j
                m = min(c_ij, c_inj)
                if m < out_c3_l[i]:
                    out_c3_l[i] = m
            # left child of j split by i: cells ij and (L[j] - ij)
            if sij > 0.0 and rj > 0.0:
                v = c_ij + c_nij
                if v < out_ml2_l[j]:
                    out_ml2_l[j] = v
                    out_j_l[j] = i
                m = min(c_ij, c_nij)
                if m < out_c3_l[j]:
                    out_c3_l[j] = m
            # right child of i split by j: cells (L[j] - ij) and (rest)
            if rj > 0.0 and rest > 0.0:
                v = c_nij + c_rest
                if v < out_ml2_r[i]:
                    out_ml2_r[i] = v
                    out_j_r[i] = j
                m = min(c_nij, c_rest)
                if m < out_c3_r[i]:
                    out_c3_r[i] = m
            # right child of j split by i: cells (L[i] - ij) and (rest)
            if ri > 0.0 and rest > 0.0:
                v = c_inj + c_rest
                if v < out_ml2_r[j]:
                    out_ml2_r[j] = v
                    out_j_r[j] = i
                m = min(c_inj, c_rest)
                if m < out_c3_r[j]:
                    out_c3_r[j] = m
    return 0


@njit(cache=NUMBA_CACHE, nogil=True)
def prep_candidates(gidx, l_leaf, l_lb, r_leaf, r_lb, bound, do_exchange, split_lb, split_ub, order):
    """Split bounds, threshold-exchange dominance, cheap filter and candidate order.

    Fills ``split_lb``/``split_ub``, writes the surviving candidate indices sorted by
    (split_lb, split_ub) into ``order`` and returns ``(n_cand, best_i, min_rejected)``:
    ``best_i`` is the split with the smallest ``split_ub`` and ``min_rejected`` the
    smallest ``split_lb`` among active splits above the bound (+inf if none).
    """
    mf = gidx.shape[0]
    best_i = 0
    best_ub = 1e300
    for i in range(mf):
        split_lb[i] = l_lb[i] + r_lb[i]
        split_ub[i] = l_leaf[i] + r_leaf[i]
        if split_ub[i] < best_ub:
            best_ub = split_ub[i]
            best_i = i
    active = np.ones(mf, dtype=np.bool_)
    if do_exchange:
        for i in range(mf - 1):
            k = i + 1
            if gidx[i] >= 0 and gidx[i] == gidx[k]:
                if r_lb[i] >= r_leaf[k] - EPS:
                    active[i] = False
                elif l_lb[k] >= l_leaf[i] - EPS:
                    active[k] = False
    n = 0
    min_rejected = 1e300
    for i in range(mf):
        if not active[i]:
            continue
        if split_lb[i] <= bound + EPS:
            order[n] = i
            n += 1
        elif split_lb[i] < min_rejected:
            min_rejected = split_lb[i]
    if n > 1:
        # exact order by (split_lb, split_ub): two stable sorts, the secondary key first.
        # The candidate loop stops at the first split above the bound, so the order must be
        # nondecreasing in split_lb exactly, not up to a rounding of a combined key.
        tmp = order[:n].copy()
        ub_n = np.empty(n)
        for t in range(n):
            ub_n[t] = split_ub[tmp[t]]
        i1 = np.argsort(ub_n, kind="mergesort")
        lb_n = np.empty(n)
        for t in range(n):
            lb_n[t] = split_lb[tmp[i1[t]]]
        i2 = np.argsort(lb_n, kind="mergesort")
        for t in range(n):
            order[t] = tmp[i1[i2[t]]]
    return n, best_i, min_rejected


def pack_columns(Xb: np.ndarray) -> np.ndarray:
    """(n, m) bool -> (m, W) uint64 with row i of the data in bit i."""
    n, m = Xb.shape
    W = (n + 63) // 64
    packed = np.packbits(np.ascontiguousarray(Xb.T), axis=1, bitorder="little")
    padded = np.zeros((m, W * 8), dtype=np.uint8)
    padded[:, :packed.shape[1]] = packed
    return np.ascontiguousarray(padded.view(np.uint64))


@njit(cache=NUMBA_CACHE, nogil=True)
def expand_kernel(F, features, group_of, kw, mask_matrix, weights, costs, diff, lam, bound, do_exchange,
                  io, fo, bo, L, dist):
    """node_stats + prep_candidates + (depth2_pairs + shape_bound) in one call.

    Workspace rows: ``io`` = feats, order, j_l, j_r; ``fo`` = l_leaf, l_lb, r_leaf, r_lb,
    l_pot, split_lb, split_ub, ml2_l, ml2_r, c3_l, c3_r, l_ub2, r_ub2, split_ub2;
    ``bo`` = l_solved, r_solved.  Returns (nv, n_cand, best_i, min_rejected, ran_depth2,
    i_d2, best_d2, lb_ge4); the depth-2 stage runs under the same size rule as before
    (cheap nodes always, otherwise only when most candidates survive the cheap filter).
    """
    K = dist.shape[0] - 1
    W = F.shape[1]
    R = mask_matrix.shape[0]
    mf = features.shape[0]
    # Word compaction: the mask's empty words contribute nothing to any popcount, so when
    # enough of them are empty the candidate features' words are copied on the non-empty
    # words only and every kernel runs on the compact copy (rows = positions in features).
    nA = 0
    for w in range(W):
        if kw[w] != 0:
            nA += 1
    if nA * 10 <= W * 7:
        active = np.empty(nA, dtype=np.int64)
        q = 0
        for w in range(W):
            if kw[w] != 0:
                active[q] = w
                q += 1
        Fc = np.empty((mf, nA), dtype=np.uint64)
        for t in range(mf):
            fr = features[t]
            for q in range(nA):
                Fc[t, q] = F[fr, active[q]]
        M = np.empty((R, nA), dtype=np.uint64)
        for r in range(R):
            for q in range(nA):
                M[r, q] = kw[active[q]] & mask_matrix[r, active[q]]
        rows_in = np.arange(mf)
        W = nA
    else:
        Fc = F
        M = np.empty((R, W), dtype=np.uint64)
        for r in range(R):
            for w in range(W):
                M[r, w] = kw[w] & mask_matrix[r, w]
        rows_in = features
    nv = node_stats(Fc, features, rows_in, group_of, M, weights, costs, diff, lam, io[0], L, fo[0], fo[1], bo[0],
                    fo[2], fo[3], bo[1], fo[4], dist, io[4], io[5], io[6])
    if nv == 0:
        return 0, 0, 0, 0.0, False, 0, 0.0, 0.0, M, Fc
    feats = io[0, :nv]
    rows = io[6, :nv]
    gidx = np.empty(nv, dtype=np.int64)
    for t in range(nv):
        gidx[t] = group_of[feats[t]]
    n_cand, best_i, min_rejected = prep_candidates(gidx, fo[0, :nv], fo[1, :nv], fo[2, :nv], fo[3, :nv], bound,
                                                   do_exchange, fo[5, :nv], fo[6, :nv], io[1, :nv])
    # a node whose candidates all belong to one numeric column is solved exactly by the
    # segmentation DP in _solve, so the pairwise stage would be wasted there
    single = gidx[0] >= 0
    if single:
        for t in range(1, nv):
            if gidx[t] != gidx[0]:
                single = False
                break
    cheap = nv * nv * W <= 32768
    if single or not (n_cand >= 2 and nv >= 2 and (cheap or (n_cand * 2 >= nv and n_cand >= 8 and nv >= 8))):
        return nv, n_cand, best_i, min_rejected, False, 0, 0.0, 0.0, M, Fc
    depth2_pairs(Fc, feats, rows, group_of, M[:K], costs, lam, dist[:K], L[:nv], fo[7, :nv], fo[8, :nv], io[2, :nv],
                 io[3, :nv], fo[9, :nv], fo[10, :nv])
    i_d2, best_d2, lb_ge4 = shape_bound(fo[0, :nv], fo[7, :nv], fo[9, :nv], fo[2, :nv], fo[8, :nv], fo[10, :nv], lam,
                                        fo[1, :nv], fo[3, :nv], fo[11, :nv], fo[12, :nv], fo[5, :nv], fo[13, :nv])
    return nv, n_cand, best_i, min_rejected, True, i_d2, best_d2, lb_ge4, M, Fc


@njit(cache=NUMBA_CACHE, nogil=True)
def max_pair(a, b):
    """max over both arrays (the children's largest lower bound)."""
    m = 0.0
    for i in range(a.shape[0]):
        if a[i] > m:
            m = a[i]
        if b[i] > m:
            m = b[i]
    return m


# depth-3 stage gate: triples are enumerated when the node has at most this many candidates
TRIPLE_MAX_NV = int(os.environ.get("TRIPLE_MAX_NV", "0"))    # 0: depth-3 stage off (v40)
D3_MAX_COUNT_LAM = float(os.environ.get("D3_MAX_COUNT_LAM", "64"))
TRIPLE_MAX_OPS = 8.0e5
D3_MAX_LEAVES = float(os.environ.get("D3_MAX_LEAVES", "8"))
D3_MAX_KW = int(os.environ.get("D3_MAX_KW", "64"))


@njit(cache=NUMBA_CACHE, nogil=True)
def depth3_triples(F, feats, rows, masks, costs, uniform_w, lam, dist, L, out_val, out_arg):
    """Exact 3-leaf and (2,2)-leaf optima of every child from the class counts of all triples.

    For candidate ``i`` and side ``io`` (1 = left child ``C and i``, 0 = right child) writes
    ``out_val[i, io] = (best 3-leaf loss, best (2,2) loss, cheapest 2-leaf cell, c4, c5b, c6)``
    (c4: leaf cell + cheapest sub-cell of the other side, c5b: 2-leaf cell + cheapest
    sub-cell of the other side, c6: cheapest sub-cells of both sides) and
    ``out_arg[i, io] = (s3, side3, t3, s22, tA22, tB22)``: the 3-leaf tree splits the child by
    ``s3``, keeps cell ``side3`` (0: the true side) as a leaf and splits the other cell by
    ``t3``; the (2,2) tree splits by ``s22`` and the cells by ``tA22`` / ``tB22``.  Splits
    with an empty cell are not trees and are skipped; 1e300 means no such tree.

    Triples are enumerated as i < j < k.  For the (child, split) pairs {(i, j), (j, i)} the
    sub-split k is the largest index, so their minima over k > j accumulate in scalars and
    are merged with the array entries (which hold the contributions of sub-splits below j,
    written by earlier iterations); the pairs involving k update the arrays directly.
    """
    nv = feats.shape[0]
    K = dist.shape[0]
    W = F.shape[1]
    P = np.empty((nv, nv, K))
    for i in range(nv):
        fi = rows[i]
        for c in range(K):
            P[i, i, c] = L[i, c]
        for j in range(i + 1, nv):
            fj = rows[j]
            for c in range(K):
                acc = np.uint64(0)
                for w in range(W):
                    acc += _popcount64(F[fi, w] & F[fj, w] & masks[c, w])
                P[i, j, c] = acc
                P[j, i, c] = acc
    # per (child r, side io, split s): best 2-leaf loss of the true cell A / false cell B of s
    # within the child, its sub-split, and the cheapest non-empty sub-cell of A / B
    b2A = np.full((nv, 2, nv), 1e300)
    b2B = np.full((nv, 2, nv), 1e300)
    b2A_arg = np.full((nv, 2, nv), -1, dtype=np.int64)
    b2B_arg = np.full((nv, 2, nv), -1, dtype=np.int64)
    pA = np.full((nv, 2, nv), 1e300)
    pB = np.full((nv, 2, nv), 1e300)
    cnt = np.empty((8, K))
    cost = np.empty(8)
    ok = np.empty(8, dtype=np.bool_)
    ijk = np.empty(K)
    fij = np.empty(W, dtype=np.uint64)
    # cell index = bi * 4 + bj * 2 + bk
    # scalar accumulators for (r, s) = (i, j) and (j, i), per io: [b2A, b2B, pA, pB] and args
    sv = np.empty((2, 2, 4))
    sa = np.empty((2, 2, 2), dtype=np.int64)
    for i in range(nv):
        fi = rows[i]
        for j in range(i + 1, nv):
            fj = rows[j]
            for w in range(W):
                fij[w] = F[fi, w] & F[fj, w]
            for io in range(2):
                sv[0, io, 0] = b2A[i, io, j]; sv[0, io, 1] = b2B[i, io, j]
                sv[0, io, 2] = pA[i, io, j]; sv[0, io, 3] = pB[i, io, j]
                sa[0, io, 0] = b2A_arg[i, io, j]; sa[0, io, 1] = b2B_arg[i, io, j]
                sv[1, io, 0] = b2A[j, io, i]; sv[1, io, 1] = b2B[j, io, i]
                sv[1, io, 2] = pA[j, io, i]; sv[1, io, 3] = pB[j, io, i]
                sa[1, io, 0] = b2A_arg[j, io, i]; sa[1, io, 1] = b2B_arg[j, io, i]
            for k in range(j + 1, nv):
                fk = rows[k]
                for c in range(K):
                    acc = np.uint64(0)
                    for w in range(W):
                        acc += _popcount64(fij[w] & F[fk, w] & masks[c, w])
                    ijk[c] = acc
                for c in range(K):
                    pij = P[i, j, c]
                    pik = P[i, k, c]
                    pjk = P[j, k, c]
                    v = ijk[c]
                    cnt[7, c] = v
                    cnt[6, c] = pij - v
                    cnt[5, c] = pik - v
                    cnt[3, c] = pjk - v
                    cnt[4, c] = L[i, c] - pij - pik + v
                    cnt[2, c] = L[j, c] - pij - pjk + v
                    cnt[1, c] = L[k, c] - pik - pjk + v
                    cnt[0, c] = dist[c] - L[i, c] - L[j, c] - L[k, c] + pij + pik + pjk - v
                for q in range(8):
                    sz = 0.0
                    mx = 0.0
                    for c in range(K):
                        v = cnt[q, c]
                        sz += v
                        if v > mx:
                            mx = v
                    ok[q] = sz > 0.0
                    if uniform_w > 0.0:
                        cost[q] = uniform_w * (sz - mx)
                    else:
                        cost[q] = _leaf_cost(cnt[q], costs, K) if sz > 0.0 else 1e300
                # --- sub-split k for (r, s) = (i, j) [row 0] and (j, i) [row 1]; io = bit of r
                for io in range(2):
                    # (r, s) = (i, j): A = cells (io, 1, *), B = cells (io, 0, *)
                    a1 = io * 4 + 2 + 1; a0 = io * 4 + 2; b1 = io * 4 + 1; b0 = io * 4
                    if ok[a1] and ok[a0]:
                        v = cost[a1] + cost[a0]
                        if v < sv[0, io, 0]:
                            sv[0, io, 0] = v; sa[0, io, 0] = k
                        v = min(cost[a1], cost[a0])
                        if v < sv[0, io, 2]:
                            sv[0, io, 2] = v
                    if ok[b1] and ok[b0]:
                        v = cost[b1] + cost[b0]
                        if v < sv[0, io, 1]:
                            sv[0, io, 1] = v; sa[0, io, 1] = k
                        v = min(cost[b1], cost[b0])
                        if v < sv[0, io, 3]:
                            sv[0, io, 3] = v
                    # (r, s) = (j, i): A = cells (1, io, *), B = cells (0, io, *)
                    a1 = 4 + io * 2 + 1; a0 = 4 + io * 2; b1 = io * 2 + 1; b0 = io * 2
                    if ok[a1] and ok[a0]:
                        v = cost[a1] + cost[a0]
                        if v < sv[1, io, 0]:
                            sv[1, io, 0] = v; sa[1, io, 0] = k
                        v = min(cost[a1], cost[a0])
                        if v < sv[1, io, 2]:
                            sv[1, io, 2] = v
                    if ok[b1] and ok[b0]:
                        v = cost[b1] + cost[b0]
                        if v < sv[1, io, 1]:
                            sv[1, io, 1] = v; sa[1, io, 1] = k
                        v = min(cost[b1], cost[b0])
                        if v < sv[1, io, 3]:
                            sv[1, io, 3] = v
                    # --- pairs involving k: (r, s) = (i, k) sub j; (k, i) sub j; (j, k) sub i; (k, j) sub i
                    # (i, k): A = (io, *, 1) cells, B = (io, *, 0); sub-cells by j
                    a1 = io * 4 + 2 + 1; a0 = io * 4 + 1; b1 = io * 4 + 2; b0 = io * 4
                    if ok[a1] and ok[a0]:
                        v = cost[a1] + cost[a0]
                        if v < b2A[i, io, k]:
                            b2A[i, io, k] = v; b2A_arg[i, io, k] = j
                        v = min(cost[a1], cost[a0])
                        if v < pA[i, io, k]:
                            pA[i, io, k] = v
                    if ok[b1] and ok[b0]:
                        v = cost[b1] + cost[b0]
                        if v < b2B[i, io, k]:
                            b2B[i, io, k] = v; b2B_arg[i, io, k] = j
                        v = min(cost[b1], cost[b0])
                        if v < pB[i, io, k]:
                            pB[i, io, k] = v
                    # (k, i): child bit k = io, A = (1, *, io), B = (0, *, io); sub-cells by j
                    a1 = 4 + 2 + io; a0 = 4 + io; b1 = 2 + io; b0 = io
                    if ok[a1] and ok[a0]:
                        v = cost[a1] + cost[a0]
                        if v < b2A[k, io, i]:
                            b2A[k, io, i] = v; b2A_arg[k, io, i] = j
                        v = min(cost[a1], cost[a0])
                        if v < pA[k, io, i]:
                            pA[k, io, i] = v
                    if ok[b1] and ok[b0]:
                        v = cost[b1] + cost[b0]
                        if v < b2B[k, io, i]:
                            b2B[k, io, i] = v; b2B_arg[k, io, i] = j
                        v = min(cost[b1], cost[b0])
                        if v < pB[k, io, i]:
                            pB[k, io, i] = v
                    # (j, k): child bit j = io, A = (*, io, 1), B = (*, io, 0); sub-cells by i
                    a1 = 4 + io * 2 + 1; a0 = io * 2 + 1; b1 = 4 + io * 2; b0 = io * 2
                    if ok[a1] and ok[a0]:
                        v = cost[a1] + cost[a0]
                        if v < b2A[j, io, k]:
                            b2A[j, io, k] = v; b2A_arg[j, io, k] = i
                        v = min(cost[a1], cost[a0])
                        if v < pA[j, io, k]:
                            pA[j, io, k] = v
                    if ok[b1] and ok[b0]:
                        v = cost[b1] + cost[b0]
                        if v < b2B[j, io, k]:
                            b2B[j, io, k] = v; b2B_arg[j, io, k] = i
                        v = min(cost[b1], cost[b0])
                        if v < pB[j, io, k]:
                            pB[j, io, k] = v
                    # (k, j): child bit k = io, A = (*, 1, io), B = (*, 0, io); sub-cells by i
                    a1 = 4 + 2 + io; a0 = 2 + io; b1 = 4 + io; b0 = io
                    if ok[a1] and ok[a0]:
                        v = cost[a1] + cost[a0]
                        if v < b2A[k, io, j]:
                            b2A[k, io, j] = v; b2A_arg[k, io, j] = i
                        v = min(cost[a1], cost[a0])
                        if v < pA[k, io, j]:
                            pA[k, io, j] = v
                    if ok[b1] and ok[b0]:
                        v = cost[b1] + cost[b0]
                        if v < b2B[k, io, j]:
                            b2B[k, io, j] = v; b2B_arg[k, io, j] = i
                        v = min(cost[b1], cost[b0])
                        if v < pB[k, io, j]:
                            pB[k, io, j] = v
            for io in range(2):
                b2A[i, io, j] = sv[0, io, 0]; b2B[i, io, j] = sv[0, io, 1]
                pA[i, io, j] = sv[0, io, 2]; pB[i, io, j] = sv[0, io, 3]
                b2A_arg[i, io, j] = sa[0, io, 0]; b2B_arg[i, io, j] = sa[0, io, 1]
                b2A[j, io, i] = sv[1, io, 0]; b2B[j, io, i] = sv[1, io, 1]
                pA[j, io, i] = sv[1, io, 2]; pB[j, io, i] = sv[1, io, 3]
                b2A_arg[j, io, i] = sa[1, io, 0]; b2B_arg[j, io, i] = sa[1, io, 1]
    A = np.empty(K)
    B = np.empty(K)
    for r in range(nv):
        for io in range(2):
            ml3 = 1e300
            m22 = 1e300
            c5 = 1e300
            c4 = 1e300
            c5b = 1e300
            c6 = 1e300
            s3 = -1
            side3 = -1
            t3 = -1
            s22 = -1
            tA = -1
            tB = -1
            for sidx in range(nv):
                if sidx == r:
                    continue
                sa_ = 0.0
                sb_ = 0.0
                for c in range(K):
                    if io == 1:
                        A[c] = P[r, sidx, c]
                        B[c] = L[r, c] - P[r, sidx, c]
                    else:
                        A[c] = L[sidx, c] - P[r, sidx, c]
                        B[c] = dist[c] - L[r, c] - L[sidx, c] + P[r, sidx, c]
                    sa_ += A[c]
                    sb_ += B[c]
                if sa_ <= 0.0 or sb_ <= 0.0:
                    continue
                lA = _leaf_cost(A, costs, K)
                lB = _leaf_cost(B, costs, K)
                vA = b2A[r, io, sidx]
                vB = b2B[r, io, sidx]
                qA = pA[r, io, sidx]
                qB = pB[r, io, sidx]
                if qB < 1e300:
                    if lA + qB < c4:
                        c4 = lA + qB
                    if vA < 1e300 and vA + qB < c5b:
                        c5b = vA + qB
                if qA < 1e300:
                    if lB + qA < c4:
                        c4 = lB + qA
                    if vB < 1e300 and vB + qA < c5b:
                        c5b = vB + qA
                    if qB < 1e300 and qA + qB < c6:
                        c6 = qA + qB
                if vB < 1e300:
                    v = lA + vB
                    if v < ml3:
                        ml3 = v
                        s3 = sidx
                        side3 = 0
                        t3 = b2B_arg[r, io, sidx]
                    if vB < c5:
                        c5 = vB
                if vA < 1e300:
                    v = vA + lB
                    if v < ml3:
                        ml3 = v
                        s3 = sidx
                        side3 = 1
                        t3 = b2A_arg[r, io, sidx]
                    if vA < c5:
                        c5 = vA
                    if vB < 1e300:
                        v = vA + vB
                        if v < m22:
                            m22 = v
                            s22 = sidx
                            tA = b2A_arg[r, io, sidx]
                            tB = b2B_arg[r, io, sidx]
            out_val[r, io, 0] = ml3
            out_val[r, io, 1] = m22
            out_val[r, io, 2] = c5
            out_val[r, io, 3] = c4
            out_val[r, io, 4] = c5b
            out_val[r, io, 5] = c6
            out_arg[r, io, 0] = s3
            out_arg[r, io, 1] = side3
            out_arg[r, io, 2] = t3
            out_arg[r, io, 3] = s22
            out_arg[r, io, 4] = tA
            out_arg[r, io, 5] = tB


@njit(cache=NUMBA_CACHE, nogil=True)
def depth3_bounds(l_leaf, ml2_l, c3_l, r_leaf, ml2_r, c3_r, val, lam, l_ub3, r_ub3, l_lb3, r_lb3, kind_l, kind_r):
    """Per-child achievable risk over {leaf, 2, 3, (2,2)} trees and lower bound over all trees.

    kind: 0 leaf, 1 two leaves, 2 three leaves, 3 (2,2).  Returns (argmin i of ub sum,
    best_d3 = min ub sum, lb_rest = min lb sum): any tree with root split i costs at least
    l_lb3[i] + r_lb3[i]; a tree with a >= 4 leaves on a child is (1,3)/(3,1) (loss >= the
    peeled cell >= c3), (2,2) (exact), or has >= 5 leaves: a = 5 peels a leaf or a 2-leaf
    cell (loss >= min(c3, c5)), a >= 6 costs >= 6 lambda.
    """
    mf = l_leaf.shape[0]
    best_i = 0
    best_d3 = 1e300
    lb_rest = 1e300
    for i in range(mf):
        for side in range(2):
            if side == 1:
                g1 = l_leaf[i]; g2 = 2.0 * lam + ml2_l[i]; c3 = c3_l[i]
            else:
                g1 = r_leaf[i]; g2 = 2.0 * lam + ml2_r[i]; c3 = c3_r[i]
            g3 = 3.0 * lam + val[i, side, 0]
            g4 = 4.0 * lam + val[i, side, 1]
            c5 = val[i, side, 2]
            c4 = val[i, side, 3]
            c5b = val[i, side, 4]
            c6 = val[i, side, 5]
            ub = g1; kind = 0
            if g2 < ub:
                ub = g2; kind = 1
            if g3 < ub:
                ub = g3; kind = 2
            if g4 < ub:
                ub = g4; kind = 3
            # 4 leaves: (2,2) is in ub, (1,3)/(3,1) cost >= c4 (>= c3); 5 leaves: (1,4)/(4,1) peel a
            # leaf (>= c3; the 4-leaf side may be (2,2)), (2,3)/(3,2) >= c5b; 6 leaves: (1,5) >= c3,
            # (2,4) >= c5, (3,3) >= c6.  Seven or more leaves get 7 lam and nothing else: a (3,4)
            # or (4,4) shape can put every leaf below depth 2 ((2,2) sides have no leaf that is a
            # cell or a sub-cell), so none of c3, c5, c6 bounds it; the earlier floor of
            # 6 lam + min(c3, c5, c6) was not admissible for those shapes.
            f4 = max(c3, c4)
            f5 = min(c3, c5b)
            f6 = min(min(c3, c5), c6)
            lb = min(min(min(ub, 4.0 * lam + f4), min(5.0 * lam + f5, 6.0 * lam + f6)), 7.0 * lam)
            if side == 1:
                l_ub3[i] = ub; l_lb3[i] = lb; kind_l[i] = kind
            else:
                r_ub3[i] = ub; r_lb3[i] = lb; kind_r[i] = kind
        v = l_ub3[i] + r_ub3[i]
        if v < best_d3:
            best_d3 = v
            best_i = i
        w = l_lb3[i] + r_lb3[i]
        if w < lb_rest:
            lb_rest = w
    return best_i, best_d3, lb_rest


def int_to_words(value: int, W: int) -> np.ndarray:
    return np.frombuffer(value.to_bytes(W * 8, "little"), dtype=np.uint64)


def warm_up():
    """Trigger JIT compilation (cached on disk afterwards)."""
    F = np.zeros((2, 1), dtype=np.uint64)
    masks = np.zeros((1, 1), dtype=np.uint64)
    child_counts(F, masks, np.zeros((2, 1), dtype=np.uint64))
    child_counts_subset(F, np.zeros(1, dtype=np.int64), masks, np.zeros((1, 1), dtype=np.uint64))
    node_stats(F, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.zeros(2, dtype=np.int64), masks,
               np.zeros(0), np.zeros((1, 1)), np.zeros(1), 0.1, np.empty(1, dtype=np.int64), np.empty((1, 1)),
               np.empty(1), np.empty(1), np.empty(1, dtype=np.bool_), np.empty(1), np.empty(1),
               np.empty(1, dtype=np.bool_), np.empty(1), np.empty(2), np.empty(1, dtype=np.int64),
               np.empty(1, dtype=np.int64), np.empty(1, dtype=np.int64))
    depth2_pairs(F, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.zeros(2, dtype=np.int64), masks,
                 np.zeros((1, 1)), 0.1,
                 np.zeros(1), np.zeros((1, 1)), np.empty(1), np.empty(1), np.empty(1, dtype=np.int64),
                 np.empty(1, dtype=np.int64), np.empty(1), np.empty(1))
    e = np.zeros(1)
    refilter_candidates(np.zeros(1, dtype=np.int64), 1, e, e, 1.0)
    depth3_triples(F, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), masks, np.zeros((1, 1)), 0.0, 0.1,
                   np.zeros(1), np.zeros((1, 1)),
                   np.empty((1, 2, 6)), np.empty((1, 2, 6), dtype=np.int64))
    depth3_bounds(e, e, e, e, e, e, np.zeros((1, 2, 6)), 0.1, np.empty(1), np.empty(1), np.empty(1), np.empty(1),
                  np.empty(1, dtype=np.int64), np.empty(1, dtype=np.int64))
    max_pair(e, e)
    expand_kernel(F, np.zeros(1, dtype=np.int64), np.zeros(2, dtype=np.int64), int_to_words(1, 1), masks, np.zeros(0),
                  np.zeros((1, 1)), np.zeros(1), 0.1, 1.0, False, np.empty((7, 1), dtype=np.int64), np.empty((14, 1)),
                  np.empty((2, 1), dtype=np.bool_), np.empty((1, 1)), np.empty(2))
    shape_bound(e, e, e, e, e, e, 0.1, np.zeros(1), np.zeros(1), np.empty(1), np.empty(1), np.empty(1), np.empty(1))
    prep_candidates(np.zeros(2, dtype=np.int64), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), 1.0, True,
                    np.empty(2), np.empty(2), np.empty(2, dtype=np.int64))
    S = np.zeros((2, 2))
    segment_dp(S, np.zeros((2, 2)), 0.0, np.zeros(2), np.zeros(2, dtype=np.int64))
    segment_dp_uniform(S, 1.0, 0.0, np.zeros(2), np.zeros(2, dtype=np.int64))

# ------------------------------------------------------------------ encoder

_MISSING_STRINGS = {"", "NULL", "null", "Null", "NA", "na", "NaN", "nan", "N/A", "n/a"}


def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])


def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    return pd.api.types.is_numeric_dtype(s)


class BinaryEncoder:
    """Fit/transform arbitrary features into a boolean split matrix.

    Attributes after ``fit``:

    rules : list of dict
        One entry per binary feature with keys ``feature`` (source column
        index), ``name`` (source column name), ``relation`` (``">="`` or
        ``"=="``), ``reference`` (threshold or category value) and ``type``
        (``"integral"``, ``"rational"`` or ``"categorical"``).
    groups : list of list of int
        Indices of binary features (in threshold order) that belong to the
        same ordinal source column with more than one threshold.
    """

    def __init__(self, drop_duplicate_columns: bool = True):
        self.drop_duplicate_columns = drop_duplicate_columns
        self.rules: list[dict] = []
        self.groups: list[list[int]] = []
        self.feature_names: list[str] = []
        self.n_source_features = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X) -> "BinaryEncoder":
        X = _to_dataframe(X)
        self.feature_names = [str(c) for c in X.columns]
        self.n_source_features = X.shape[1]
        rules: list[dict] = []
        groups: list[list[int]] = []

        for j, col in enumerate(X.columns):
            s = X[col]
            name = str(col)
            if _is_numeric_series(s):
                values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                has_missing = finite.shape[0] != values.shape[0]
                uniq = np.unique(finite)
                if uniq.shape[0] <= 1:
                    continue
                integral = bool(np.all(np.equal(np.mod(uniq, 1), 0)))
                kind = "integral" if integral else "rational"
                if uniq.shape[0] == 2 and not has_missing:
                    ref = uniq[1]
                    rules.append({
                        "feature": j, "name": name, "relation": "==",
                        "reference": int(ref) if integral else float(ref), "type": kind,
                    })
                    continue
                start = len(rules)
                for a, b in zip(uniq[:-1], uniq[1:]):
                    if integral:
                        ref = int(b)
                    else:
                        ref = float(0.5 * (a + b))
                    rules.append({
                        "feature": j, "name": name, "relation": ">=",
                        "reference": ref, "type": kind,
                    })
                groups.append(list(range(start, len(rules))))
            else:
                raw = s.to_numpy(dtype=object)
                mask = np.array([not _is_missing(v) for v in raw], dtype=bool)
                present = raw[mask]
                has_missing = present.shape[0] != raw.shape[0]
                uniq = sorted(set(present.tolist()), key=lambda v: str(v))
                if len(uniq) <= 1:
                    continue
                if len(uniq) == 2 and not has_missing:
                    uniq = uniq[1:]
                for v in uniq:
                    rules.append({
                        "feature": j, "name": name, "relation": "==",
                        "reference": v, "type": "categorical",
                    })

        self.rules = rules
        self.groups = groups

        if self.drop_duplicate_columns and rules:
            Xb = self._apply_rules(X, rules)
            keep = _unique_partitions(Xb)
            if keep.shape[0] != len(rules):
                remap = {old: new for new, old in enumerate(keep.tolist())}
                self.rules = [rules[i] for i in keep.tolist()]
                self.groups = [
                    g2 for g2 in ([remap[i] for i in g if i in remap] for g in groups)
                    if len(g2) > 1
                ]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X) -> np.ndarray:
        X = _to_dataframe(X)
        if X.shape[1] != self.n_source_features:
            raise ValueError(
                f"expected {self.n_source_features} feature columns, got {X.shape[1]}"
            )
        return self._apply_rules(X, self.rules)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_binary_features(self) -> int:
        return len(self.rules)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _apply_rules(X: pd.DataFrame, rules: list[dict]) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros((n, len(rules)), dtype=bool)
        cache: dict[int, np.ndarray] = {}
        for k, rule in enumerate(rules):
            j = rule["feature"]
            if rule["relation"] == ">=":
                if j not in cache:
                    cache[j] = pd.to_numeric(X.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
                col = cache[j]
                with np.errstate(invalid="ignore"):
                    out[:, k] = col >= rule["reference"]
            else:
                if rule["type"] == "categorical":
                    col = X.iloc[:, j].to_numpy(dtype=object)
                    ref = rule["reference"]
                    out[:, k] = np.array([(not _is_missing(v)) and v == ref for v in col], dtype=bool)
                else:
                    if j not in cache:
                        cache[j] = pd.to_numeric(X.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
                    col = cache[j]
                    with np.errstate(invalid="ignore"):
                        out[:, k] = col == rule["reference"]
        return out


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v in _MISSING_STRINGS:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _unique_partitions(Xb: np.ndarray) -> np.ndarray:
    """Return indices of columns whose induced row partition is new.

    A column and its complement induce the same partition, so columns are
    canonicalised by flipping them when their first row is ``True``.
    """
    if Xb.shape[1] == 0:
        return np.arange(0)
    canon = Xb ^ Xb[0:1, :]
    packed = np.packbits(canon, axis=0)
    seen: dict[bytes, int] = {}
    keep = []
    for k in range(packed.shape[1]):
        key = packed[:, k].tobytes()
        if key in seen:
            continue
        seen[key] = k
        keep.append(k)
    return np.array(keep, dtype=np.int64)


class TargetEncoder:
    """Map arbitrary labels to contiguous integer class indices."""

    def __init__(self):
        self.classes_: np.ndarray | None = None

    def fit(self, y) -> "TargetEncoder":
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        y = np.asarray(y).ravel()
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= self.classes_.shape[0]) or np.any(self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y):
            raise ValueError("labels contain classes not seen during fit")
        return idx.astype(np.int64)

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse(self, idx: int):
        v = self.classes_[idx]
        if isinstance(v, np.generic):
            return v.item()
        return v

# ------------------------------------------------------------------ dataset

def column_to_int(col: np.ndarray) -> int:
    """Pack a boolean vector into an int whose bit ``i`` is ``col[i]``."""
    packed = np.packbits(np.ascontiguousarray(col, dtype=bool), bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def int_to_column(value: int, n: int) -> np.ndarray:
    nbytes = (n + 7) // 8
    raw = np.frombuffer(value.to_bytes(nbytes, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


def cluster_rows(Xb: np.ndarray, y: np.ndarray, K: int, groups) -> np.ndarray:
    """Permutation sorting the rows lexicographically by the source columns (a numeric
    column's thresholds are nested, so its rank is the number of true thresholds), columns
    ordered by the best root-level misclassification gain among their splits."""
    n, m = Xb.shape
    if n == 0 or m == 0:
        return np.arange(n)
    counts = np.zeros((m, K))
    for k in range(K):
        counts[:, k] = Xb[y == k].sum(axis=0)
    dist = np.bincount(y, minlength=K).astype(float)
    left = counts.sum(axis=1)
    err_left = left - counts.max(axis=1)
    err_right = (n - left) - (dist[None, :] - counts).max(axis=1)
    gain = (n - dist.max()) - err_left - err_right
    in_group = np.zeros(m, dtype=bool)
    keys = []
    for g in groups or []:
        g = list(g)
        if len(g) < 2:
            continue
        in_group[g] = True
        keys.append((float(gain[g].max()), Xb[:, g].sum(axis=1)))
    for j in np.flatnonzero(~in_group):
        keys.append((float(gain[j]), Xb[:, j].astype(np.int64)))
    keys.sort(key=lambda t: -t[0])
    # np.lexsort sorts by the last key first
    return np.lexsort([k for _, k in keys[::-1]])


class BitDataset:
    """Binary features, class targets and misclassification costs as bitsets.

    Parameters
    ----------
    Xb : (n, m) bool array of binary split features.
    y : (n,) int array of class indices in ``[0, n_classes)``.
    n_classes : number of classes.
    costs : optional (K, K) matrix; ``costs[i, j]`` is the cost of predicting
        class ``i`` when the true class is ``j``.  Defaults to ``1/n`` off the
        diagonal (unweighted misclassification rate).
    balance : if True and ``costs`` is None, use ``1 / (K * count_j)`` so every
        class carries the same total weight (the reference ``balance`` flag).
    """

    def __init__(self, Xb: np.ndarray, y: np.ndarray, n_classes: int,
                 costs: np.ndarray | None = None, balance: bool = False):
        Xb = np.ascontiguousarray(Xb, dtype=bool)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, m = Xb.shape
        if y.shape[0] != n:
            raise ValueError("X and y have different numbers of rows")
        self.n = n
        self.m = m
        self.K = int(n_classes)
        self.full = (1 << n) - 1
        self.features = [column_to_int(Xb[:, j]) for j in range(m)]
        self.targets = [column_to_int(y == k) for k in range(self.K)]
        self.class_counts = np.array([int(t.bit_count()) for t in self.targets], dtype=np.int64)

        # ---- cost matrix and its aggregations (Dataset::aggregate_cost_matrix)
        K = self.K
        if costs is not None:
            C = np.asarray(costs, dtype=np.float64)
            if C.shape != (K, K):
                raise ValueError(f"costs must have shape {(K, K)}")
            self.uniform = False
        elif balance:
            C = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    if i != j:
                        C[i, j] = 1.0 / (K * max(int(self.class_counts[j]), 1))
            self.uniform = False
        else:
            C = np.full((K, K), 1.0 / n)
            np.fill_diagonal(C, 0.0)
            self.uniform = True
        self.costs = C
        self.match_costs = np.diag(C).copy()
        self.max_costs = C.max(axis=0)
        self.min_costs = C.min(axis=0)
        self.diff_costs = self.max_costs - self.min_costs
        self._diff_list = [float(v) for v in self.diff_costs]
        mismatch = np.full(K, np.inf)
        for j in range(K):
            for i in range(K):
                if i != j:
                    mismatch[j] = min(mismatch[j], C[i, j])
        if K == 1:
            mismatch[:] = 0.0
        self.mismatch_costs = mismatch
        self._w = float(mismatch[0]) if self.uniform else 0.0

        # ---- equivalent points: rows with identical features but different labels
        _, inverse = np.unique(Xb, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        n_groups = int(inverse.max()) + 1 if n else 0
        dist = np.zeros((n_groups, K), dtype=np.float64)
        np.add.at(dist, (inverse, y), 1.0)
        group_cost = dist @ C.T                       # [g, i] = cost of predicting i for group g
        minimizer = np.argmin(group_cost, axis=1)     # first minimal index, like the reference
        majority_rows = minimizer[inverse] == y
        self.majority = column_to_int(majority_rows)
        self.minority = self.full & ~self.majority
        self.majority_by_class = [self.majority & t for t in self.targets]
        self.minority_by_class = [self.minority & t for t in self.targets]

        # Fast paths for the equivalent-points loss.
        self.zero_diagonal = bool(np.all(self.match_costs == 0.0))
        self.equal_mismatch = bool(np.all(self.mismatch_costs == self.mismatch_costs[0]))

        # Packed 64-bit word representation used by the numba kernels.
        self.W = (n + 63) // 64
        self.F_words = pack_columns(Xb) if m else np.zeros((0, self.W), dtype=np.uint64)
        self.target_words = [int_to_words(t, self.W) for t in self.targets]
        self.minority_words = int_to_words(self.minority, self.W)
        self.minority_by_class_words = [int_to_words(v, self.W) for v in self.minority_by_class]
        self.majority_by_class_words = [int_to_words(v, self.W) for v in self.majority_by_class]
        # (K+1, W) matrix of the class masks followed by the equivalent-points mask, and the
        # matching weight vector, for the uniform-cost fast path of the node kernel
        self.mask_matrix = np.ascontiguousarray(np.vstack(self.target_words + [self.minority_words]))
        self.weights1 = np.array([float(self.mismatch_costs[0])])
        self.ones_words = int_to_words(self.full, self.W)

    # ------------------------------------------------------------------
    def leaf_stats(self, capture: int):
        """Return ``(count, dist, max_loss, min_loss, potential, prediction)``.

        ``max_loss`` is the loss of the best single label (the leaf loss),
        ``min_loss`` the equivalent-points lower bound on any tree's loss and
        ``potential`` the maximal loss reduction any split could achieve.
        """
        if self.uniform:
            # uniform costs: predicting class p costs w * (count - dist[p]); the first
            # maximal class wins ties, as np.argmin over the cost vector would
            counts = [(capture & t).bit_count() for t in self.targets]
            count = sum(counts)
            best = 0
            for k in range(1, self.K):
                if counts[k] > counts[best]:
                    best = k
            w = self._w
            return (count, np.array(counts, dtype=np.float64), w * (count - counts[best]),
                    w * (capture & self.minority).bit_count(), w * count, best)
        dist = np.array([int((capture & t).bit_count()) for t in self.targets], dtype=np.float64)
        count = int(dist.sum())
        pred_costs = self.costs @ dist
        prediction = int(np.argmin(pred_costs))
        max_loss = float(pred_costs[prediction])
        potential = float(self.diff_costs @ dist)
        min_loss = self.equivalent_loss(capture)
        return count, dist, max_loss, min_loss, potential, prediction

    def equivalent_loss(self, capture: int) -> float:
        if self.zero_diagonal:
            if self.equal_mismatch:
                return float(self.mismatch_costs[0]) * (capture & self.minority).bit_count()
            return float(sum(
                float(self.mismatch_costs[k]) * (capture & self.minority_by_class[k]).bit_count()
                for k in range(self.K)
            ))
        total = 0.0
        for k in range(self.K):
            total += float(self.match_costs[k]) * (capture & self.majority_by_class[k]).bit_count()
            total += float(self.mismatch_costs[k]) * (capture & self.minority_by_class[k]).bit_count()
        return total

    def distance(self, capture: int, i: int, j: int, needed: float = np.inf) -> float:
        """Similar-support distance between features ``i`` and ``j`` on ``capture``.

        Returns ``min(cost of rows where i != j, cost of rows where i == j)``.
        If the first term already exceeds ``needed`` the caller cannot prune, so
        the second term is skipped and the first is returned.
        """
        differ = capture & (self.features[i] ^ self.features[j])
        pos = 0.0
        for k in range(self.K):
            d = self._diff_list[k]
            if d != 0.0:
                pos += d * (differ & self.targets[k]).bit_count()
        if pos >= needed:
            return pos
        agree = capture & ~differ
        neg = 0.0
        for k in range(self.K):
            d = self._diff_list[k]
            if d != 0.0:
                neg += d * (agree & self.targets[k]).bit_count()
        return min(pos, neg)

# ------------------------------------------------------------------ model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _leaf_matches(value, node) -> bool:
    rel = node["relation"]
    ref = node["reference"]
    if rel == ">=":
        try:
            return bool(value >= ref)
        except TypeError:
            return False
    if rel == "<=":
        try:
            return bool(value <= ref)
        except TypeError:
            return False
    # equality (categorical or binary numeric)
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    try:
        return bool(value == ref)
    except TypeError:
        return False


class TreeClassifier:
    """Interactive wrapper around a JSON tree (mirrors ``python/model/tree_classifier.py``)."""

    def __init__(self, source: dict):
        self.source = source

    # ---------------------------------------------------------- prediction
    def _find_leaf(self, sample):
        node = self.source
        while "prediction" not in node:
            value = sample[node["feature"]]
            node = node["true"] if _leaf_matches(value, node) else node["false"]
        return node

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.to_numpy(dtype=object)
        else:
            rows = np.asarray(X, dtype=object)
            if rows.ndim == 1:
                rows = rows.reshape(1, -1)
        return np.array([self._find_leaf(row)["prediction"] for row in rows], dtype=object)

    def predict_fast(self, X) -> np.ndarray:
        """Vectorised prediction for numeric-only / categorical feature matrices."""
        if isinstance(X, pd.DataFrame):
            frame = X
        else:
            arr = np.asarray(X)
            frame = pd.DataFrame(arr)
        n = frame.shape[0]
        out = np.empty(n, dtype=object)
        idx = np.arange(n)
        self._predict_rec(self.source, frame, idx, out)
        return out

    def _predict_rec(self, node, frame, idx, out):
        if "prediction" in node:
            out[idx] = node["prediction"]
            return
        col = frame.iloc[idx, node["feature"]]
        rel = node["relation"]
        ref = node["reference"]
        if rel == ">=":
            vals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                mask = vals >= ref
        elif rel == "<=":
            vals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                mask = vals <= ref
        else:
            if node.get("type") == "categorical":
                mask = np.array([(v == ref) if not (isinstance(v, float) and np.isnan(v)) else False
                                 for v in col.to_numpy(dtype=object)], dtype=bool)
            else:
                vals = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
                with np.errstate(invalid="ignore"):
                    mask = vals == ref
        self._predict_rec(node["true"], frame, idx[mask], out)
        self._predict_rec(node["false"], frame, idx[~mask], out)

    def error(self, X, y, weight=None):
        y = np.asarray(y).ravel()
        pred = self.predict_fast(X)
        miss = (pred != y).astype(np.float64)
        if weight is None:
            return float(miss.mean())
        weight = np.asarray(weight, dtype=np.float64).ravel()
        return float((miss * weight).sum() / weight.sum())

    def score(self, X, y, weight=None):
        return 1.0 - self.error(X, y, weight=weight)

    # ------------------------------------------------------------ structure
    def _all_leaves(self, node=None):
        node = self.source if node is None else node
        if "prediction" in node:
            return [node]
        return self._all_leaves(node["true"]) + self._all_leaves(node["false"])

    def leaves(self) -> int:
        return len(self._all_leaves())

    def nodes(self) -> int:
        def rec(node):
            if "prediction" in node:
                return 1
            return 1 + rec(node["true"]) + rec(node["false"])
        return rec(self.source)

    def maximum_depth(self) -> int:
        def rec(node):
            if "prediction" in node:
                return 1
            return 1 + max(rec(node["true"]), rec(node["false"]))
        return rec(self.source)

    def loss(self) -> float:
        return float(sum(leaf["loss"] for leaf in self._all_leaves()))

    def complexity(self) -> float:
        return float(sum(leaf["complexity"] for leaf in self._all_leaves()))

    def risk(self) -> float:
        return self.loss() + self.complexity()

    def __len__(self):
        return self.leaves()

    def json(self, indent: int | None = 2) -> str:
        return json.dumps(self.source, indent=indent, cls=NumpyEncoder)

    def features(self) -> list:
        feats = []

        def rec(node):
            if "prediction" in node:
                return
            feats.append(node["feature"])
            rec(node["true"])
            rec(node["false"])

        rec(self.source)
        return sorted(set(feats))

    def __str__(self):
        lines = []

        def rec(node, depth):
            pad = "    " * depth
            if "prediction" in node:
                lines.append(f"{pad}{node['name']} = {node['prediction']!r}  (loss={node['loss']:.6g})")
                return
            lines.append(f"{pad}if {node['name']} {node['relation']} {node['reference']!r} then:")
            rec(node["true"], depth + 1)
            lines.append(f"{pad}else:")
            rec(node["false"], depth + 1)

        rec(self.source, 0)
        return "\n".join(lines)

    __repr__ = __str__

# ------------------------------------------------------------------ optimizer



class TimeLimitReached(Exception):
    """Raised inside the search when the time or memory limit is hit."""


def _store_bytes(st) -> int:
    """Bytes held by a node store (keys, index, per-node fields, pending)."""
    cap, W = st[ST_KEYS].shape
    return int(cap) * (8 * int(W) + 16 + 7 * 8 + 32 + 1)


def _rss_bytes() -> int:
    """Current resident set size of this process in bytes (0 if unavailable).

    ``resource.getrusage`` only reports the lifetime peak, which would keep
    tripping the guard after one large search, so the live value is read from
    ``/proc`` on Linux and from ``ps`` elsewhere.
    """
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip() or 0) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


class Node:
    __slots__ = ("key", "count", "leaf_risk", "prediction", "lb", "ub", "split", "solved", "cache", "pending")

    def __init__(self, key: int, count: int, leaf_risk: float, prediction: int,
                 lb: float, solved: bool):
        self.key = key
        self.count = count
        self.leaf_risk = leaf_risk
        self.prediction = prediction
        self.lb = lb
        self.ub = leaf_risk
        self.split = -1
        self.solved = solved
        self.cache = None
        # deferred grandchildren of a depth-3 structure: (kind, s, side/tA, t/tB) as feature ids
        self.pending = None


class Optimizer:
    def __init__(self, data: BitDataset, regularization: float, *,
                 groups: list[list[int]] | None = None,
                 time_limit: float = 0.0,
                 look_ahead: bool = True,
                 similar_support: bool = True,
                 feature_exchange: bool = True,
                 continuous_feature_exchange: bool = True,
                 greedy_init: bool = True,
                 upperbound: float = 0.0,
                 engine: str = "auto",
                 memory_limit: int = 0,
                 verbose: bool = False):
        self.data = data
        self.memory_limit = int(memory_limit)
        self.stop_reason = ""
        if engine == "auto":
            engine = "numba" if HAVE_NUMBA else "python"
        if engine == "numba" and not HAVE_NUMBA:
            raise ImportError("numba is not installed; use engine='python'")
        self.engine = engine
        if engine == "numba":
            warm_up()
        self.lam = float(regularization)
        self.time_limit = float(time_limit)
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.feature_exchange = feature_exchange
        self.continuous_feature_exchange = continuous_feature_exchange
        self.greedy_init = greedy_init
        self.upperbound = float(upperbound)
        self.verbose = verbose
        self.memo: dict[int, Node] = {}
        self.iterations = 0          # number of subproblem expansions
        self.start_time = 0.0
        self.elapsed = 0.0
        self.optimal = False

        # ordinal neighbour map used by the continuous feature exchange bound
        # ``feature_exchange`` is accepted for configuration compatibility only: the
        # reference's pairwise version prunes whole subtrees with parent bounds and
        # is not exact, so it is not applied (see README).
        self.next_in_group = np.full(data.m, -1, dtype=np.int64)
        for g in (groups or []):
            for a, b in zip(g[:-1], g[1:]):
                self.next_in_group[a] = b

        self._has_groups = bool(np.any(self.next_in_group >= 0))
        # group id per binary feature (-1: not a numeric threshold); a group's features are
        # consecutive in feature order, which is threshold order
        self.group_of = np.full(data.m, -1, dtype=np.int64)
        for gi, g in enumerate(groups or []):
            self.group_of[g] = gi
        self._pos_buffer = np.full(data.m, -1, dtype=np.int64)

        self._costs_T = data.costs.T.copy()
        self._diff = data.diff_costs.copy()
        # uniform costs (all mismatches cost w, matches 0): a cell's leaf cost is w * (size - max count)
        self._uniform_w = float(data.mismatch_costs[0]) if (data.zero_diagonal and data.equal_mismatch
                                                            and np.all(data.costs == (data.costs > 0) * data.mismatch_costs[0])) else 0.0
        self._no_groups = np.full(len(data.features), -1, dtype=np.int64)

    # ------------------------------------------------------------------ API
    def run(self) -> Node:
        self.start_time = time.perf_counter()
        self._last_mem_check = self.start_time
        root_key = self.data.full
        root = self._make_node(root_key)
        features = np.arange(self.data.m, dtype=np.int64)
        try:
            if self.greedy_init and not root.solved:
                self._greedy(root, features)
            # a user upperbound only restricts the search (a budget); it is never stored
            # as achievable, so a wrong value cannot produce a false certificate
            budget = root.ub if self.upperbound <= 0.0 else min(root.ub, self.upperbound)
            self._solve(root, budget, features)
            self.optimal = root.solved
            self.stop_reason = "optimal" if root.solved else "upperbound"
        except TimeLimitReached as exc:
            self.optimal = False
            self.stop_reason = str(exc)
        self.elapsed = time.perf_counter() - self.start_time
        return root

    # ------------------------------------------------------------- nodes
    def _make_node(self, key: int) -> Node:
        node = self.memo.get(key)
        if node is not None:
            return node
        count, dist, max_loss, min_loss, potential, prediction = self.data.leaf_stats(key)
        leaf_risk = max_loss + self.lam
        lb, solved = self._initial_bounds(count, max_loss, min_loss, potential, leaf_risk)
        node = Node(key, count, leaf_risk, prediction, lb, solved)
        self.memo[key] = node
        return node

    def _initial_bounds(self, count, max_loss, min_loss, potential, leaf_risk):
        lam = self.lam
        # leaf-only conditions, valid for any nonnegative cost matrix:
        #  * a single point cannot be split;
        #  * max_loss - min_loss < lam: any split costs >= min_loss + 2 lam > max_loss + lam;
        #  * potential < 2 lam: a split would create a child of potential < lam, whose leaves
        #    can all be removed for a strict gain (leaf-support lemma).
        if (count <= 1
                or max_loss - min_loss < lam
                or potential < 2.0 * lam):
            return leaf_risk, True
        return min(leaf_risk, min_loss + 2.0 * lam), False

    # ------------------------------------------------------------ children
    def _child_statistics(self, node: Node, features: np.ndarray):
        """Vectorised statistics of the left/right child of every candidate split.

        Returns ``(feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot, L, dist)``
        restricted to features that actually split the capture set and whose children
        both have potential >= lam (leaf-support exclusion, see node_stats).
        """
        data = self.data
        key = node.key
        K = data.K
        lam = self.lam
        if self.engine == "numba":
            kw = int_to_words(key, data.W)
            if data.zero_diagonal and data.equal_mismatch:
                # class masks and the equivalent-points mask in one bitwise op
                M = np.bitwise_and(kw[None, :], data.mask_matrix)
                self._last_class_masks = M[:K]
                buf = self._buffers(features.shape[0])
                nv = node_stats(data.F_words, features, features, self.group_of, M, data.weights1, data.costs,
                                self._diff, lam, *buf)
                if nv == 0:
                    return None
                feats, L, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot = (x[:nv] for x in buf[:9])
                return feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot, L, buf[9][:K]
            masks = [kw & tw for tw in data.target_words]
            weights = []
            pairs = [(data.minority_by_class_words[k], float(data.mismatch_costs[k])) for k in range(K)]
            if not data.zero_diagonal:
                pairs += [(data.majority_by_class_words[k], float(data.match_costs[k])) for k in range(K)]
            for mask_words, w in pairs:
                if w == 0.0:
                    continue
                masks.append(kw & mask_words)
                weights.append(w)
            M = np.stack(masks)
            self._last_class_masks = M[:K]
            buf = self._buffers(features.shape[0])
            nv = node_stats(data.F_words, features, features, self.group_of, M, np.array(weights), data.costs,
                            self._diff, lam, *buf)
            if nv == 0:
                return None
            feats, L, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot = (x[:nv] for x in buf[:9])
            return feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot, L, buf[9][:K]

        L, lmin, dist, min_total = self._counts_python(key, features)
        total = node.count
        lsum = L.sum(axis=1)
        l_pot_all = L @ self._diff
        total_pot = float(dist @ self._diff)
        valid = (lsum > 0) & (lsum < total) & (l_pot_all >= lam - EPS) & (total_pot - l_pot_all >= lam - EPS)
        if not valid.any():
            return None
        feats = features[valid]
        L = L[valid]
        lsum = lsum[valid]
        lmin = lmin[valid]
        rmin = min_total - lmin
        R = dist[None, :] - L
        rsum = total - lsum
        l_max = (L @ self._costs_T).min(axis=1)
        r_max = (R @ self._costs_T).min(axis=1)
        l_pot = l_pot_all[valid]
        r_pot = total_pot - l_pot
        l_leaf = l_max + lam
        r_leaf = r_max + lam
        l_solved = (lsum <= 1) | (l_max - lmin < lam) | (l_pot < 2 * lam)
        r_solved = (rsum <= 1) | (r_max - rmin < lam) | (r_pot < 2 * lam)
        l_lb = np.where(l_solved, l_leaf, np.minimum(l_leaf, lmin + 2 * lam))
        r_lb = np.where(r_solved, r_leaf, np.minimum(r_leaf, rmin + 2 * lam))
        return feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot, L, dist

    def _buffers(self, mf: int):
        """Fresh output arrays for node_stats (the returned views must survive recursion)."""
        K = self.data.K
        fl = np.empty((5, mf))
        bo = np.empty((2, mf), dtype=np.bool_)
        pr = np.empty((3, mf), dtype=np.int64)
        return (np.empty(mf, dtype=np.int64), np.empty((mf, K)), fl[0], fl[1], bo[0], fl[2], fl[3], bo[1],
                fl[4], np.empty(K + 1), pr[0], pr[1], pr[2])

    def _expand(self, node: Node, features: np.ndarray, bound: float):
        """Fused per-node kernel call (numba engine): see expand_kernel.

        The budget-independent outputs are cached on the node (``node.cache``) while the
        parent deepens the node's budget; a re-expansion with the same allowed features
        then only redoes the cheap candidate preparation for the new bound.
        """
        data = self.data
        K = data.K
        c = node.cache
        if c is not None and c[0] is features:
            io, fo, bo, L, dist, nv, ran, i_d2, best_d2, lb_ge4, pij = c[1]
            gidx = self.group_of[io[0]] if self._has_groups else self._no_groups[:nv]
            n_cand, i, min_rejected = prep_candidates(gidx, fo[0], fo[1], fo[2], fo[3], bound,
                                                      self.continuous_feature_exchange and self._has_groups,
                                                      fo[5], fo[6], io[1])
            return io, fo, bo, L, dist, nv, int(n_cand), int(i), float(min_rejected), ran, i_d2, best_d2, lb_ge4, pij
        kw = int_to_words(node.key, data.W)
        if data.zero_diagonal and data.equal_mismatch:
            mask_matrix = data.mask_matrix
            weights = data.weights1
        else:
            masks = [kw & tw for tw in data.target_words]
            ws = []
            pairs = [(data.minority_by_class_words[k], float(data.mismatch_costs[k])) for k in range(K)]
            if not data.zero_diagonal:
                pairs += [(data.majority_by_class_words[k], float(data.match_costs[k])) for k in range(K)]
            for mask_words, w in pairs:
                if w == 0.0:
                    continue
                masks.append(kw & mask_words)
                ws.append(w)
            # (general costs: the masks are already restricted to the node; the kernel's AND with
            # the full-ones word vector is then the identity)
            mask_matrix = np.stack(masks)
            kw = data.ones_words
            weights = np.array(ws)
        mf = features.shape[0]
        io = np.empty((7, mf), dtype=np.int64)
        fo = np.empty((14, mf))
        bo = np.empty((2, mf), dtype=np.bool_)
        L = np.empty((mf, K))
        dist = np.empty(K + 1)
        nv, n_cand, i, min_rejected, ran, i_d2, best_d2, lb_ge4, M, Fc = expand_kernel(
            data.F_words, features, self.group_of, kw, mask_matrix, weights, data.costs, self._diff, self.lam, bound,
            self.continuous_feature_exchange and self._has_groups, io, fo, bo, L, dist)
        if nv == 0:
            return None
        nv = int(nv)
        if nv < mf:
            io = io[:, :nv]
            fo = fo[:, :nv]
            bo = bo[:, :nv]
            L = L[:nv]
        pij = None
        i_d2 = int(i_d2); best_d2 = float(best_d2); lb_ge4 = float(lb_ge4)
        if ran and nv <= TRIPLE_MAX_NV and K * data.W <= D3_MAX_KW and nv * nv * nv / 6.0 * (K * data.W + 150.0) <= TRIPLE_MAX_OPS:
            pij = (M[:K], float(dist[K]), Fc)     # depth-3 stage available; computed on demand by _depth3
        dist = dist[:K]
        node.cache = (features, (io, fo, bo, L, dist, nv, bool(ran), i_d2, best_d2, lb_ge4, pij))
        return io, fo, bo, L, dist, nv, int(n_cand), int(i), float(min_rejected), bool(ran), i_d2, best_d2, lb_ge4, pij

    def _depth3(self, node: Node, io, fo, L, dist, masks, Fc):
        """Depth-3 stage: exact 3-leaf and (2,2) optima of every child from the triples;
        the result replaces the placeholder in the node's cache."""
        data = self.data
        K = data.K
        nv = io.shape[1]
        val = np.empty((nv, 2, 6))
        arg = np.empty((nv, 2, 6), dtype=np.int64)
        depth3_triples(Fc, io[0], io[6], masks, data.costs, self._uniform_w, self.lam, dist[:K], L, val, arg)
        f3 = np.empty((4, nv))
        k3 = np.empty((2, nv), dtype=np.int64)
        i3, best_d3, lb_rest = depth3_bounds(fo[0], fo[7], fo[9], fo[2], fo[8], fo[10], val, self.lam,
                                             f3[0], f3[1], f3[2], f3[3], k3[0], k3[1])
        d3 = (val, arg, f3[0], f3[1], f3[2], f3[3], k3[0], k3[1], int(i3), float(best_d3), float(lb_rest))
        c = node.cache
        if c is not None:
            node.cache = (c[0], c[1][:10] + (d3,))
        return d3

    def _counts_python(self, key: int, features: np.ndarray):
        """Per-feature left-child class counts and equivalent-points loss (big ints)."""
        data = self.data
        K = data.K
        F = data.features
        CT = [key & t for t in data.targets]
        dist = np.array([int(ct.bit_count()) for ct in CT], dtype=np.float64)
        mf = features.shape[0]
        L = np.empty((mf, K), dtype=np.float64)
        for k in range(K):
            ct = CT[k]
            L[:, k] = [(ct & F[j]).bit_count() for j in features]
        if data.zero_diagonal and data.equal_mismatch:
            w = float(data.mismatch_costs[0])
            CM = key & data.minority
            lmin = np.array([(CM & F[j]).bit_count() for j in features], dtype=np.float64) * w
            min_total = w * CM.bit_count()
        else:
            lmin = np.zeros(mf)
            min_total = 0.0
            pairs = [(data.minority_by_class[k], float(data.mismatch_costs[k])) for k in range(K)]
            if not data.zero_diagonal:
                pairs += [(data.majority_by_class[k], float(data.match_costs[k])) for k in range(K)]
            for mask, w in pairs:
                if w == 0.0:
                    continue
                CMk = key & mask
                lmin += np.array([(CMk & F[j]).bit_count() for j in features], dtype=np.float64) * w
                min_total += w * CMk.bit_count()
        return L, lmin, dist, min_total

    def _counts_numba(self, key: int, features: np.ndarray):
        """Same as ``_counts_python`` using the packed-word numba kernel."""
        data = self.data
        K = data.K
        kw = int_to_words(key, data.W)
        masks = [kw & tw for tw in data.target_words]
        weights = []
        min_total = 0.0
        if data.zero_diagonal and data.equal_mismatch:
            w = float(data.mismatch_costs[0])
            masks.append(kw & data.minority_words)
            weights.append(w)
            min_total = w * (key & data.minority).bit_count()
        else:
            pairs = [(data.minority_by_class[k], data.minority_by_class_words[k], float(data.mismatch_costs[k]))
                     for k in range(K)]
            if not data.zero_diagonal:
                pairs += [(data.majority_by_class[k], data.majority_by_class_words[k], float(data.match_costs[k]))
                          for k in range(K)]
            for mask_int, mask_words, w in pairs:
                if w == 0.0:
                    continue
                masks.append(kw & mask_words)
                weights.append(w)
                min_total += w * (key & mask_int).bit_count()
        M = np.stack(masks)
        out = np.empty((features.shape[0], M.shape[0]), dtype=np.uint64)
        child_counts_subset(data.F_words, features, M, out)
        counts = out.astype(np.float64)
        L = counts[:, :K]
        dist = np.array([int((key & t).bit_count()) for t in data.targets], dtype=np.float64)
        lmin = np.zeros(features.shape[0])
        for r, w in enumerate(weights):
            lmin += counts[:, K + r] * w
        return L, lmin, dist, min_total

    def _child_node(self, key: int, count: int, leaf: float, lb: float, solved: bool,
                    prediction: int = -1) -> Node:
        node = self.memo.get(key)
        if node is not None:
            return node
        if prediction < 0:
            # (greedy path) prediction recomputed from the mask: costs may be non-uniform
            _, dist, max_loss, _, _, prediction = self.data.leaf_stats(key)
        node = Node(key, count, leaf, prediction, lb, solved)
        self.memo[key] = node
        return node

    # --------------------------------------------------------------- greedy
    def _greedy(self, node: Node, features: np.ndarray, depth: int = 0) -> float:
        """Greedy dive that seeds ``ub``/``split`` along its path."""
        if node.solved or depth > 30:
            return node.ub
        stats = self._child_statistics(node, features)
        if stats is None:
            node.solved = True
            node.lb = node.ub = node.leaf_risk
            return node.ub
        feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, _, _, _ = stats
        immediate = l_leaf + r_leaf
        i = int(np.argmin(immediate))
        if immediate[i] >= node.leaf_risk - EPS:
            return node.ub
        j = int(feats[i])
        lkey = node.key & self.data.features[j]
        rkey = node.key ^ lkey
        left = self._child_node(lkey, 0, float(l_leaf[i]), float(l_lb[i]), bool(l_solved[i]))
        right = self._child_node(rkey, 0, float(r_leaf[i]), float(r_lb[i]), bool(r_solved[i]))
        left.count = lkey.bit_count()
        right.count = rkey.bit_count()
        value = self._greedy(left, feats, depth + 1) + self._greedy(right, feats, depth + 1)
        if value < node.ub:
            node.ub = value
            node.split = j
        return node.ub

    # ---------------------------------------------------------------- solve
    def _solve(self, node: Node, budget: float, features: np.ndarray) -> None:
        """Establish ``node.lb == node.ub`` if the optimum is within ``budget``,
        otherwise prove ``node.lb > budget``."""
        if node.solved or node.lb > budget + EPS:
            return
        self.iterations += 1
        if (self.iterations & 63) == 0:
            now = time.perf_counter()
            if self.time_limit > 0.0 and now - self.start_time > self.time_limit:
                raise TimeLimitReached("time")
            if self.memory_limit > 0 and now - self._last_mem_check > 0.5:
                # (reading the resident size costs ~4 ms: at most twice a second)
                self._last_mem_check = now
                if _rss_bytes() > self.memory_limit:
                    raise TimeLimitReached("memory")

        data = self.data
        F = data.features
        key = node.key
        memo = self.memo
        d2 = None
        if self.engine == "numba":
            res = self._expand(node, features, min(budget, node.ub))
            if res is None:
                node.lb = node.ub = node.leaf_risk
                node.split = -1
                node.solved = True
                return
            io, fo, bo, L, dist, mf_, n_cand, i, min_rejected, ran, i_d2, best_d2, lb_ge4, pij = res
            d3 = pij if (pij is not None and len(pij) == 11) else None
            d3_avail = pij is not None and len(pij) == 3
            if ran and min(node.leaf_risk, best_d2, lb_ge4) > budget + EPS:
                # probe: no tree fits the budget (see the shape relaxation below); nothing
                # else of this expansion is needed, and the cache serves a re-expansion
                node.lb = max(node.lb, min(node.leaf_risk, best_d2, lb_ge4))
                return
            if d3 is not None and min(node.leaf_risk, d3[10]) > budget + EPS:
                # probe by the depth-3 bounds: any tree with root split i costs at least
                # l_lb3[i] + r_lb3[i] (see depth3_bounds)
                node.lb = max(node.lb, min(node.leaf_risk, d3[10]))
                return
            feats = io[0]; order_buf = io[1]; l_pred = io[4]; r_pred = io[5]
            l_leaf = fo[0]; l_lb = fo[1]; r_leaf = fo[2]; r_lb = fo[3]; l_pot = fo[4]
            split_lb = fo[5]; split_ub = fo[6]
            l_solved = bo[0]; r_solved = bo[1]
            d2 = (io[2], io[3], fo[11], fo[12], fo[13], i_d2, best_d2, lb_ge4) if ran else None
            gidx = self.group_of[feats] if self._has_groups else self._no_groups[:mf_]
        else:
            stats = self._child_statistics(node, features)
            if stats is None:
                node.lb = node.ub = node.leaf_risk
                node.split = -1
                node.solved = True
                return
            feats, l_leaf, l_lb, l_solved, r_leaf, r_lb, r_solved, l_pot, L, dist = stats
            l_pred = r_pred = None
            d3 = None
            d3_avail = False
            mf_ = feats.shape[0]
            gidx = self.group_of[feats] if self._has_groups else self._no_groups[:mf_]
            ws = np.empty((2, mf_))
            split_lb = ws[0]
            split_ub = ws[1]
            order_buf = np.empty(mf_, dtype=np.int64)
            n_cand, i, min_rejected = prep_candidates(gidx, l_leaf, l_lb, r_leaf, r_lb, min(budget, node.ub),
                                                      self.continuous_feature_exchange and self._has_groups,
                                                      split_lb, split_ub, order_buf)

        best = node.ub
        best_split = node.split
        # immediate upper bound: both children as leaves
        if split_ub[i] < best - EPS:
            best = float(split_ub[i])
            best_split = int(feats[i])

        # Single-column segmentation DP.  Every tree built from the thresholds of one
        # numeric column induces a segmentation of the captured points sorted by that
        # column, with the same leaves and loss, and every segmentation into k segments
        # is realised by a chain of k-1 threshold splits; so the best tree over one
        # column is the best segmentation, a quadratic DP over the nested class counts.
        # If the column's thresholds are the only features that split this node, that
        # value is the node's exact optimum (optimal substructure also solves every
        # segment); otherwise it is a valid incumbent tree.
        g_span = None
        if self._has_groups:
            # feats is ordered by column (features are), so a column's thresholds are a
            # contiguous span; boundaries are where the group index changes
            cut = np.flatnonzero(gidx[1:] != gidx[:-1]) + 1
            starts = np.concatenate(([0], cut))
            ends = np.concatenate((cut, [mf_]))
            g_span = {int(gidx[a]): (int(a), int(b)) for a, b in zip(starts, ends) if gidx[a] >= 0}
            single = len(g_span) == 1 and starts.shape[0] == 1
            # the incumbent DP on multi-column nodes is only run at the root: below it the
            # depth-2 stage supplies incumbents, and 23 DPs per node cost more than they save
            large = node.count == data.n
            for g, (a, b) in g_span.items():
                if not single and (not large or b - a < 2):
                    continue
                val, bounds_idx = self._column_dp(L[a:b], dist)
                if val < best - EPS or single:
                    if val < best - EPS:
                        best = float(val)
                        best_split = self._materialize_chain(node, key, feats[a:b], bounds_idx, L[a:b], dist, single)
                    if single:
                        node.ub = min(node.ub, best)
                        if best_split >= 0 or node.ub <= node.leaf_risk + EPS:
                            node.split = best_split if best < node.leaf_risk - EPS else -1
                        node.lb = node.ub
                        node.solved = True
                        return
        bound = min(budget, best)
        # the kernel filtered against min(budget, node.ub); `best` may have dropped since
        # (column DP), which only removes candidates the loop rejects anyway
        cand = order_buf[:n_cand]
        min_pruned = float(min_rejected)

        # Depth-2 pairwise stage.  Worth its O(m^2) cost only when the recursion it
        # replaces is larger: many surviving candidates whose children would each be
        # expanded (O(m) each).  Exact consequences:
        #  * lam2 + best2(child) is achievable, so it is an upper bound per child;
        #  * every tree on a child is a leaf, a single split (>= 2 lam + best2) or has
        #    >= 3 leaves (>= 3 lam): lb(child) = min(leaf, 2 lam + best2, 3 lam);
        #  * with budget < 4 lam no 4-leaf tree fits, and every tree with <= 3 leaves
        #    is a depth-2 tree, so the best depth-2 tree is the node's optimum (or
        #    proves it exceeds the budget).
        lam = self.lam
        l_ub2 = r_ub2 = None
        if d2 is not None:
            # (the kernel ran depth2_pairs + shape_bound; l_lb/r_lb/split_lb are already raised)
            j_l, j_r, l_ub2, r_ub2, split_ub2, i, best_d2, lb_ge4 = d2
            # Shape relaxation (see shape_bound): lb_ge4 bounds every tree with >= 4 leaves,
            # best_d2 is the exact optimum over trees with 2 or 3 leaves (and depth-2 ones).
            lb_all = min(node.leaf_risk, best_d2, lb_ge4)
            if lb_all > budget + EPS:
                # no tree fits the budget: leaves, depth-2 trees and all larger trees exceed it
                # (node.ub is left alone: an ub without its tree in the memo would be unusable)
                node.lb = max(node.lb, lb_all)
                return
            if best_d2 < best - EPS:
                best = best_d2
                best_split = int(feats[i])
                self._materialize_depth2(node, key, feats, i, j_l, j_r, l_ub2, r_ub2, l_leaf, r_leaf,
                                             l_lb, r_lb, l_solved, r_solved, l_pred, r_pred)
                bound = min(budget, best)
            if best_d2 <= lb_ge4 and best_d2 <= budget + EPS and best_d2 <= node.leaf_risk + EPS:
                # the exact best depth-2 tree is no worse than any tree with >= 4 leaves and
                # no worse than the leaf: it is the optimum
                if best_d2 < node.ub - EPS or node.split < 0 and best_d2 < node.leaf_risk - EPS:
                    node.ub = best_d2
                    node.split = int(feats[i])
                    node.pending = None
                    self._materialize_depth2(node, key, feats, i, j_l, j_r, l_ub2, r_ub2, l_leaf, r_leaf,
                                             l_lb, r_lb, l_solved, r_solved, l_pred, r_pred)
                node.ub = min(node.ub, best_d2)
                node.lb = node.ub
                node.solved = True
                return
            if budget < 4 * lam - EPS:
                # Exact resolution (kept from v8): with budget < 4 lam every tree with >= 4
                # leaves exceeds the budget and every smaller tree is a depth-2 tree.
                node.ub = best
                if node.split != best_split:
                    node.pending = None
                node.split = best_split
                if best <= budget + EPS:
                    node.lb = best
                    node.solved = True
                else:
                    node.lb = max(node.lb, min(best, 4 * lam))
                return
            if (d3 is None and d3_avail and budget < pij[1] + D3_MAX_LEAVES * lam - EPS
                    and node.count <= D3_MAX_COUNT_LAM * lam * data.n):
                # (a node whose budget affords >= D3_MAX_LEAVES leaves above its equivalent-points
                # loss is rarely resolved by the depth-3 stage; its children then get expanded anyway)
                d3 = self._depth3(node, io, fo, L, dist, pij[0], pij[2])
            if d3 is not None:
                # Depth-3 stage.  l_ub3/r_ub3: exact best over {leaf, 2, 3, (2,2)}-leaf trees
                # on each child (achievable); l_lb3/r_lb3: lower bound over all trees on the
                # child (depth3_bounds).  best_d3 = min_i l_ub3 + r_ub3 is achievable and
                # lb_rest = min_i l_lb3 + r_lb3 bounds every tree with a root split, so if
                # best <= min(leaf, lb_rest) the node is solved; otherwise the children's
                # bounds tighten the candidate loop.
                val3, arg3, l_ub3, r_ub3, l_lb3, r_lb3, kind_l, kind_r, i3, best_d3, lb_rest = d3
                if best_d3 < best - EPS:
                    best = best_d3
                    best_split = int(feats[i3])
                    self._materialize_depth3(node, key, feats, i3, d3, j_l, j_r)
                    bound = min(budget, best)
                lb3 = min(node.leaf_risk, lb_rest)
                if lb3 > budget + EPS:
                    node.ub = best
                    if node.split != best_split:
                        node.pending = None
                    node.split = best_split
                    node.lb = max(node.lb, lb3)
                    return
                if best <= lb3 + EPS:
                    node.ub = best
                    if node.split != best_split:
                        node.pending = None
                    node.split = best_split
                    node.lb = node.ub
                    node.solved = True
                    return
                np.maximum(l_lb, l_lb3, out=l_lb)
                np.maximum(r_lb, r_lb3, out=r_lb)
                split_lb[:] = l_lb + r_lb
                l_ub2 = l_ub3
                r_ub2 = r_ub3
                split_ub2 = l_ub3 + r_ub3
            # candidates dropped here by their tightened bound still bound the node from below
            n_keep, min_dropped = refilter_candidates(order_buf, n_cand, split_lb, split_ub2, bound + EPS)
            cand = order_buf[:n_keep]
            if min_dropped < min_pruned:
                min_pruned = float(min_dropped)
        # Candidates are visited in increasing order of their cheap lower bound, so
        # the loop can stop at the first one exceeding the budget.  Memoised bounds
        # of existing children are consulted lazily, only for visited candidates.
        # (with the depth-2 stage the survivors are already sorted by (split_lb, split_ub2))
        order = cand.tolist()
        look_ahead = self.look_ahead

        # Similar-support propagation within a numeric column.  Two thresholds t < t'
        # of one column differ exactly on the captured points with t <= x < t', whose
        # cost-weighted count is pot[t] - pot[t'] (pot = potential of the left child,
        # monotone along the column).  Any tree for split t' turns into a tree for
        # split t by moving those points, so |R(t) - R(t')| <= |pot[t] - pot[t']|, and a
        # proven lower bound v on one split lowers-bounds every threshold of its column
        # by v - distance.  This is the reference's similar-support bound applied to
        # the whole column at once instead of to the two neighbouring features only.
        sim = self.similar_support and self._has_groups
        if sim:
            lb_arr = split_lb.copy()
            pot = l_pot

            def propagate(i, v):
                if v <= bound + EPS:
                    return
                g = int(gidx[i])
                if g < 0:
                    return
                a, b = g_span[g]
                seg = lb_arr[a:b]
                np.maximum(seg, v - np.abs(pot[a:b] - pot[i]), out=seg)

        child_lb_max = max_pair(l_lb, r_lb)
        ln = rn = None
        for i in order:
            if ln is not None:
                # the previous candidate's children will not be re-expanded from here
                ln.cache = None
            if rn is not None:
                rn.cache = None
            ln = rn = None
            raw = float(split_lb[i])
            if raw > bound + EPS:
                if raw < min_pruned:
                    min_pruned = raw
                break
            if sim:
                cur = float(lb_arr[i])
                if cur > bound + EPS:
                    if cur < min_pruned:
                        min_pruned = cur
                    continue
            else:
                cur = raw
            j = int(feats[i])
            lkey = key & F[j]
            rkey = key ^ lkey
            ln = memo.get(lkey)
            rn = memo.get(rkey)
            if ln is None:
                llb, lub = float(l_lb[i]), float(l_leaf[i])
            else:
                llb, lub = ln.lb, ln.ub
            if rn is None:
                rlb, rub = float(r_lb[i]), float(r_leaf[i])
            else:
                rlb, rub = rn.lb, rn.ub
            sub = lub + rub
            if sub < best - EPS:
                best = sub
                best_split = j
                bound = min(budget, best)
            slb = llb + rlb
            if slb < cur:
                slb = cur
            if slb > bound + EPS:
                if slb < min_pruned:
                    min_pruned = slb
                if sim:
                    propagate(i, slb)
                continue
            if ln is None:
                ln = self._child_node(lkey, lkey.bit_count(), lub, llb, bool(l_solved[i]),
                                      -1 if l_pred is None else int(l_pred[i]))
            if rn is None:
                rn = self._child_node(rkey, rkey.bit_count(), rub, rlb, bool(r_solved[i]),
                                      -1 if r_pred is None else int(r_pred[i]))
            if l_ub2 is not None:
                # depth-2 (or depth-3) knowledge about these children: achievable ub and lb
                for cn, ub2, jj, side in ((ln, float(l_ub2[i]), int(j_l[i]), True), (rn, float(r_ub2[i]), int(j_r[i]), False)):
                    if ub2 < cn.ub - EPS:
                        if d3 is not None:
                            self._materialize_child(cn, feats, i, 1 if side else 0, int((kind_l if side else kind_r)[i]),
                                                    arg3, jj, ub2)
                        else:
                            cn.ub = ub2
                            cn.split = int(feats[jj]) if (jj >= 0 and ub2 < (l_leaf[i] if side else r_leaf[i]) - EPS) else -1
                            cn.pending = None
                    lbi = float((l_lb if side else r_lb)[i])
                    if cn.lb < lbi:
                        cn.lb = lbi
                    if not cn.solved and cn.ub <= cn.lb + EPS:
                        cn.lb = cn.ub
                        cn.solved = True

            # solve the child with the larger lower bound first (more likely to prune);
            # with look-ahead the child only gets the budget its sibling leaves over
            first, second = (ln, rn) if ln.lb >= rn.lb else (rn, ln)
            if l_ub2 is not None:
                # a child whose budget is below 3 lam cannot afford 3 leaves: its optimum is
                # min(leaf, 2 lam + best2) = ub, exact, so no expansion is needed
                # (with the depth-3 stage the ub is also exact over 3-leaf trees, so the
                # same holds below 4 lam)
                exact_below = 4 * lam if d3 is not None else 3 * lam
                for cn, other in ((first, second), (second, first)):
                    if not cn.solved and bound - other.lb < exact_below - EPS and cn.ub <= bound - other.lb + EPS:
                        cn.lb = cn.ub
                        cn.solved = True
            # Alternating budget deepening.  _solve(node, b) always leaves node either
            # solved or with lb > b, whatever b is, so calling it with a sequence of
            # growing budgets is exact.  Starting small makes a losing child fail cheaply,
            # and each child's raised lower bound shrinks the budget of the other.
            step = 2.0 * lam
            pruned = False
            while look_ahead and not first.solved:
                bf = bound - second.lb
                if first.lb > bf + EPS:
                    pruned = True
                    break
                self._solve(first, min(bf, first.lb + step), feats)
                if first.lb > bf + EPS:
                    pruned = True
                    break
                if first.solved:
                    break
                bs = bound - first.lb
                if not second.solved:
                    if second.lb > bs + EPS:
                        pruned = True
                        break
                    self._solve(second, min(bs, second.lb + step), feats)
                    if second.lb > bs + EPS:
                        pruned = True
                        break
                step *= 2.0
            if not pruned:
                self._solve(first, bound - second.lb if look_ahead else bound, feats)
            if first.lb > child_lb_max:
                child_lb_max = first.lb
            if second.lb > child_lb_max:
                child_lb_max = second.lb
            if pruned or first.lb > bound - second.lb + EPS:
                slb = first.lb + second.lb
                if slb < min_pruned:
                    min_pruned = slb
                if sim:
                    propagate(i, slb)
                continue
            self._solve(second, bound - first.ub if look_ahead else bound, feats)
            if second.lb > child_lb_max:
                child_lb_max = second.lb
            if second.lb > bound - first.ub + EPS:
                slb = first.ub + second.lb
                if slb < min_pruned:
                    min_pruned = slb
                if sim:
                    propagate(i, slb)
                continue
            value = first.ub + second.ub
            if value < best - EPS:
                best = value
                best_split = j
                bound = min(budget, best)
            elif sim:
                propagate(i, value)

        if ln is not None:
            ln.cache = None
        if rn is not None:
            rn.cache = None
        node.ub = best
        if node.split != best_split:
            node.pending = None
        node.split = best_split
        if best <= budget + EPS:
            node.lb = best
            node.solved = True
        else:
            # Superset bound: restricting an optimal tree of this node to any subset drops
            # points (loss can only fall) and empties leaves (leaves can only fall), so
            # R(C) >= R(S) >= lb(S) for every child S of every split.
            node.lb = max(node.lb, min(best, min_pruned), child_lb_max)

    def _column_dp(self, Lg, dist):
        """Optimal segmentation value and boundaries for one column's thresholds.

        ``Lg[i]`` are the class counts of ``{x >= t_i}`` for the column's thresholds in
        increasing order; ``dist`` the class counts of the whole node.  Returns the
        optimal risk and the list of boundary positions (1-based threshold positions).
        """
        M = Lg.shape[0]
        S = np.empty((M + 2, Lg.shape[1]))
        S[0] = 0.0
        S[1:M + 1] = dist[None, :] - Lg
        S[M + 1] = dist
        best = np.empty(M + 2)
        back = np.empty(M + 2, dtype=np.int64)
        if self.data.uniform:
            val = segment_dp_uniform(S, float(self.data.mismatch_costs[0]), self.lam, best, back)
        else:
            val = segment_dp(S, self._costs_T.T.copy(), self.lam, best, back)
        bounds_idx = []
        i = M + 1
        while True:
            j = int(back[i])
            if j <= 0:
                break
            bounds_idx.append(j)
            i = j
        bounds_idx.reverse()
        return float(val), bounds_idx

    def _materialize_chain(self, node, key, feats_g, bounds_idx, Lg, dist, exact):
        """Store the segmentation tree in the memo as a chain of splits.

        The split at boundary ``j`` uses threshold ``feats_g[j - 1]``; its right child
        (``x < t``) is the segment below the boundary, a leaf, and its left child holds
        the remaining segments.  With ``exact`` the chain nodes are marked solved.
        Returns the root's split feature, or -1 when the best tree is a single leaf.
        """
        if not bounds_idx:
            return -1
        F = self.data.features
        lam = self.lam
        costs_T = self._costs_T
        M = Lg.shape[0]
        S = np.empty((M + 2, Lg.shape[1]))
        S[0] = 0.0
        S[1:M + 1] = dist[None, :] - Lg
        S[M + 1] = dist
        # value of each segment (leaf risk)
        edges = [0] + list(bounds_idx) + [M + 1]
        seg_val = [float(((S[edges[t + 1]] - S[edges[t]]) @ costs_T).min()) + lam for t in range(len(edges) - 1)]
        suffix = [0.0] * (len(seg_val) + 1)
        for t in range(len(seg_val) - 1, -1, -1):
            suffix[t] = suffix[t + 1] + seg_val[t]
        cur, cur_key = node, key
        root_split = int(feats_g[bounds_idx[0] - 1])
        for t, j in enumerate(bounds_idx):
            f = int(feats_g[j - 1])
            lkey = cur_key & F[f]
            rkey = cur_key ^ lkey
            rn = self._make_node(rkey)
            ln = self._make_node(lkey)
            if suffix[t] < cur.ub - EPS or (exact and cur is not node):
                cur.ub = min(cur.ub, suffix[t])
                cur.split = f
            if exact:
                cur.ub = min(cur.ub, suffix[t])
                cur.lb = cur.ub
                cur.solved = True
                rn.ub = min(rn.ub, seg_val[t])
                rn.lb = rn.ub
                rn.split = -1
                rn.solved = True
            cur, cur_key = ln, lkey
        # the top segment is a leaf
        if exact:
            cur.ub = min(cur.ub, seg_val[-1])
            cur.lb = cur.ub
            cur.split = -1
            cur.solved = True
        return root_split

    def _materialize_child(self, cn, feats, i, io, kind, arg, jj, ub):
        """Record child ``cn``'s best {leaf, 2, 3, (2,2)}-leaf tree (kind 0..3): its split now,
        the grandchildren's splits deferred to ``_apply_pending`` (extraction or expansion)."""
        cn.ub = ub
        cn.pending = None
        if kind == 0:
            cn.split = -1
        elif kind == 1:
            cn.split = int(feats[jj])
        elif kind == 2:
            s3, side3, t3 = int(arg[i, io, 0]), int(arg[i, io, 1]), int(arg[i, io, 2])
            cn.split = int(feats[s3])
            cn.pending = (2, side3, int(feats[t3]), -1)
        else:
            s22, tA, tB = int(arg[i, io, 3]), int(arg[i, io, 4]), int(arg[i, io, 5])
            cn.split = int(feats[s22])
            cn.pending = (3, int(feats[tA]), int(feats[tB]), -1)

    def _apply_pending(self, cn):
        """Create the deferred grandchildren of a depth-3 structure recorded on ``cn``."""
        pend = cn.pending
        cn.pending = None
        if pend is None or cn.split < 0:
            return
        F = self.data.features
        akey = cn.key & F[cn.split]
        bkey = cn.key ^ akey
        if pend[0] == 2:
            cells = ((bkey if pend[1] == 0 else akey, pend[2]),)
        else:
            cells = ((akey, pend[1]), (bkey, pend[2]))
        for cell, t in cells:
            gn = self._make_node(cell)
            g1 = self._make_node(cell & F[t])
            g2 = self._make_node(cell ^ (cell & F[t]))
            v = g1.leaf_risk + g2.leaf_risk
            if v < gn.ub - EPS:
                gn.ub = v
                gn.split = t
                gn.pending = None

    def _materialize_depth3(self, node, key, feats, i, d3, j_l, j_r):
        """Store the best tree of the depth-3 stage (root split i) in the memo."""
        val3, arg3, l_ub3, r_ub3, l_lb3, r_lb3, kind_l, kind_r, _, _, _ = d3
        F = self.data.features
        f = int(feats[i])
        lkey = key & F[f]
        rkey = key ^ lkey
        for ckey, ub, kind, jj, io in ((lkey, float(l_ub3[i]), int(kind_l[i]), int(j_l[i]), 1),
                                       (rkey, float(r_ub3[i]), int(kind_r[i]), int(j_r[i]), 0)):
            cn = self._make_node(ckey)
            if ub < cn.ub - EPS:
                self._materialize_child(cn, feats, i, io, kind, arg3, jj, ub)

    def _materialize_depth2(self, node, key, feats, i, j_l, j_r, l_ub2, r_ub2, l_leaf, r_leaf,
                            l_lb=None, r_lb=None, l_solved=None, r_solved=None, l_pred=None, r_pred=None):
        """Store the best depth-2 tree (root split i, children possibly split once) in the memo."""
        F = self.data.features
        f = int(feats[i])
        lkey = key & F[f]
        rkey = key ^ lkey
        for ckey, ub2, jj, leaf, lb_a, so_a, pr_a in ((lkey, float(l_ub2[i]), int(j_l[i]), float(l_leaf[i]), l_lb, l_solved, l_pred),
                                                      (rkey, float(r_ub2[i]), int(j_r[i]), float(r_leaf[i]), r_lb, r_solved, r_pred)):
            if lb_a is None:
                cn = self._make_node(ckey)
            else:
                # the kernel's statistics of this child (leaf risk, lower bound, prediction)
                cn = self._child_node(ckey, ckey.bit_count(), leaf, float(lb_a[i]), bool(so_a[i]),
                                      -1 if pr_a is None else int(pr_a[i]))
            if ub2 < cn.ub - EPS:
                cn.ub = ub2
                cn.split = int(feats[jj]) if (jj >= 0 and ub2 < leaf - EPS) else -1
                cn.pending = None

    def _continuous_exchange(self, feats, l_lb, l_leaf, r_lb, r_leaf):
        """Return a mask of splits not dominated by the next threshold of their column.

        Binary feature ``j`` is ``x >= t_j``; its *left* child is the rows where
        it holds.  For consecutive thresholds ``t_i < t_k`` of one ordinal
        column, ``left_i ⊇ left_k`` and ``right_i ⊆ right_k``.  The optimal
        risk is monotone under set inclusion (restricting an optimal tree to a
        subset never increases loss or leaves), so ``R(left_i) >= R(left_k)`` and
        ``R(right_i) <= R(right_k)``.  Hence if ``lb(right_i) >= ub(right_k)``
        split ``k`` dominates split ``i``, and if ``lb(left_k) >= ub(left_i)``
        split ``i`` dominates split ``k``.  Domination chains never form cycles
        because the two rules are mutually exclusive on the same pair.
        """
        mf = feats.shape[0]
        active = np.ones(mf, dtype=bool)
        if mf < 2:
            return active
        # consecutive candidates of the same column are consecutive thresholds (duplicates
        # of identical splits having been dropped in the kernel)
        gidx = self.group_of[feats]
        idx = np.flatnonzero((gidx[:-1] >= 0) & (gidx[:-1] == gidx[1:]))
        if idx.shape[0] == 0:
            return active
        kk = idx + 1
        dominated_i = r_lb[idx] >= r_leaf[kk] - EPS
        dominated_k = (~dominated_i) & (l_lb[kk] >= l_leaf[idx] - EPS)
        active[idx[dominated_i]] = False
        active[kk[dominated_k]] = False
        return active

    # ------------------------------------------------------------ extraction
    def extract(self, node: Node, features_hint=None) -> dict:
        """Return the memoised tree below ``node`` as nested dicts of
        ``{"feature": j, "true": ..., "false": ...}`` / ``{"prediction": k, "key": capture}``."""
        if node.pending is not None:
            self._apply_pending(node)
        if node.split < 0:
            return {"prediction": node.prediction, "key": node.key, "count": node.count}
        j = node.split
        lkey = node.key & self.data.features[j]
        rkey = node.key ^ lkey
        left = self._make_node(lkey)
        right = self._make_node(rkey)
        return {"feature": j, "true": self.extract(left), "false": self.extract(right)}

# ------------------------------------------------------------------ gosdt

# ===========================================================================
# Compiled search (v28): the whole branch-and-bound in numba over an array memo
# ===========================================================================
# Node store: dense arrays indexed by node id (nkeys, ncount, nleaf, npred, nlb, nub,
# nsplit, nsolved, npend) and an open-addressing index (hidx: hash slot -> node id).
# meta: [n_nodes, iterations, abort (0 none, 1 iteration budget, 2 store full), max_iter]

ST_KEYS, ST_HIDX, ST_COUNT, ST_LEAF, ST_PRED, ST_LB, ST_UB, ST_SPLIT, ST_SOLVED, ST_PEND, ST_META = range(11)
DT_F, DT_GROUP, DT_MASKS, DT_WEIGHTS, DT_COSTS, DT_COSTS_T, DT_DIFF = range(7)
# parameters (float array): lam, uniform_w, n; flags (int array): K, W, has_groups, look_ahead,
# similar_support, cont_exchange, d3 enabled, is_uniform
PF_LAM, PF_UW, PF_N = range(3)
PI_K, PI_W, PI_GROUPS, PI_LOOKAHEAD, PI_SIM, PI_EXCH, PI_D3, PI_UNIFORM = range(8)


SH_KEYS, SH_COUNTS, SH_LBS, SH_VALS, SH_USED, SH_META = range(6)   # meta: [T, C, min_count]
SH_MIN_DIV = int(os.environ.get("SH_MIN_DIV", "64"))        # share subproblems with >= n / SH_MIN_DIV rows
SH_MEM_MB = float(os.environ.get("SH_MEM_MB", "192"))       # memory budget of the table (all regions)
SPLIT_EXTERN = -3                                          # split <= -3: solved in thread (-3 - split)


def make_shared_table(T, C, W, min_count):
    return (np.zeros((T, C, W), dtype=np.uint64), np.zeros((T, C), dtype=np.int64), np.zeros((T, C)),
            np.full((T, C), np.nan), np.zeros((T, C), dtype=np.uint8), np.array([T, C, min_count], dtype=np.int64))


def shared_table_capacity(T, W):
    """Slots per region: a power of two within the memory budget (at least 4096, at most 2**14)."""
    per_slot = 8 * W + 33
    C = 4096
    while C * 2 * T * per_slot <= SH_MEM_MB * 1e6 and C < (1 << 14):
        C *= 2
    return C


_SH_POOL = {}


def get_shared_table(T, W, min_count):
    """A table for this fit from the pool (one per (T, W)): its used flags and values are
    reset here, before any thread starts, so every thread sees an empty table."""
    sh = _SH_POOL.pop((T, W), None)
    if sh is None:
        sh = make_shared_table(T, shared_table_capacity(T, W), W, min_count)
    else:
        sh[SH_USED][:] = 0
        sh[SH_VALS][:] = np.nan
        sh[SH_META][2] = min_count
    return sh


def release_shared_table(sh, T, W):
    _SH_POOL[(T, W)] = sh


def no_shared_table(W):
    return make_shared_table(0, 1, W, 1 << 62)


@njit(cache=NUMBA_CACHE, nogil=True)
def sh_lookup(sh, kw, count):
    """(largest bound, exact value or NaN, owner thread) published for the capture ``kw``.
    Lock-free: keys are written once (a partially written key can only coincide with a
    strict subset, which has a different row count), bounds only ever increase, and a
    value is a single 64-bit store made only for the slot's own key, so whatever a reader
    sees is either NaN or the optimum of exactly this subproblem."""
    keys = sh[SH_KEYS]; counts = sh[SH_COUNTS]; lbs = sh[SH_LBS]; vals = sh[SH_VALS]; used = sh[SH_USED]
    T = sh[SH_META][0]; C = sh[SH_META][1]
    W = kw.shape[0]
    hmask = np.int64(C - 1)
    h0 = _slot_of(kw, hmask)
    best = -1.0
    val = np.nan
    owner = -1
    for t in range(T):
        slot = h0
        for _ in range(64):            # bounded probe
            if used[t, slot] == 0:
                break
            if counts[t, slot] == count:
                same = True
                for w in range(W):
                    if keys[t, slot, w] != kw[w]:
                        same = False
                        break
                if same:
                    v = lbs[t, slot]
                    if v > best:
                        best = v
                    x = vals[t, slot]
                    if owner < 0 and x == x:
                        val = x
                        owner = t
                    break
            slot = (slot + 1) & hmask
    return best, val, owner


@njit(cache=NUMBA_CACHE, nogil=True)
def sh_publish(sh, tid, kw, count, lb, val):
    """Record a proven bound (and, if ``val`` is not NaN, the exact optimum) in this thread's
    region (the only writer of that region).  Dropped when the probe finds no room."""
    keys = sh[SH_KEYS]; counts = sh[SH_COUNTS]; lbs = sh[SH_LBS]; vals = sh[SH_VALS]; used = sh[SH_USED]
    C = sh[SH_META][1]
    W = kw.shape[0]
    hmask = np.int64(C - 1)
    slot = _slot_of(kw, hmask)
    for _ in range(64):
        if used[tid, slot] == 0:
            if val == val:
                vals[tid, slot] = val
            lbs[tid, slot] = lb
            counts[tid, slot] = count
            for w in range(W):
                keys[tid, slot, w] = kw[w]
            used[tid, slot] = 1
            return
        if counts[tid, slot] == count:
            same = True
            for w in range(W):
                if keys[tid, slot, w] != kw[w]:
                    same = False
                    break
            if same:
                if lb > lbs[tid, slot]:
                    lbs[tid, slot] = lb
                if val == val and vals[tid, slot] != vals[tid, slot]:
                    vals[tid, slot] = val
                return
        slot = (slot + 1) & hmask


@njit(cache=NUMBA_CACHE, nogil=True)
def _tighten_stack(st, ws, d, look_ahead):
    """After the depth-0 bound dropped: re-derive the budgets of the frames below it from
    their parents' bounds exactly as they were derived when pushed (first child: bound minus
    the second's lb; second child: bound minus the first's lb while deepening, minus its ub
    in the final solve).  Budgets only decrease, so every conclusion stays sound."""
    FI = ws[WS_FI]; FF = ws[WS_FF]
    nlb = st[ST_LB]; nub = st[ST_UB]
    for k in range(1, d + 1):
        p = ws[WS_SLOT][k - 1]; c = ws[WS_SLOT][k]
        pb = FF[p, FF_BOUND]; phase = FI[p, FI_PHASE]
        first = FI[p, FI_FIRST]; second = FI[p, FI_SECOND]
        if not look_ahead:
            nb = pb
        elif phase == 3 or phase == 6:
            nb = pb - nlb[second]
        elif phase == 4:
            nb = pb - nlb[first]
        elif phase == 8:
            nb = pb - nub[first]
        else:
            return
        if phase == 3 and nb < FF[p, FF_BF]:
            FF[p, FF_BF] = nb
        if phase == 4 and nb < FF[p, FF_BS]:
            FF[p, FF_BS] = nb
        if nb < FF[c, FF_BUDGET]:
            FF[c, FF_BUDGET] = nb
        if nb < FF[c, FF_BOUND]:
            FF[c, FF_BOUND] = nb


@njit(cache=NUMBA_CACHE, nogil=True)
def _sh_publish_node(sh, tid, st, node, lb0, lam):
    """After a node's frame: publish its optimum when solved, else its bound when it was
    raised by at least lambda/4 since entry."""
    if st[ST_SOLVED][node] == 1:
        if st[ST_SPLIT][node] > SPLIT_EXTERN:      # not adopted from another thread
            sh_publish(sh, tid, st[ST_KEYS][node], st[ST_COUNT][node], st[ST_UB][node], st[ST_UB][node])
    elif st[ST_LB][node] > lb0 + 0.25 * lam:
        sh_publish(sh, tid, st[ST_KEYS][node], st[ST_COUNT][node], st[ST_LB][node], np.nan)


@njit(cache=NUMBA_CACHE, nogil=True)
def _slot_of(kw, hmask):
    h = np.uint64(1469598103934665603)
    for w in range(kw.shape[0]):
        h = (h ^ kw[w]) * np.uint64(1099511628211)
    return np.int64(h & np.uint64(hmask))


@njit(cache=NUMBA_CACHE, nogil=True)
def store_find(st, kw):
    """Node id of the capture ``kw`` or -1."""
    nkeys = st[ST_KEYS]
    hidx = st[ST_HIDX]
    W = kw.shape[0]
    hmask = np.int64(hidx.shape[0] - 1)
    slot = _slot_of(kw, hmask)
    while True:
        nid = hidx[slot]
        if nid < 0:
            return -1
        same = True
        for w in range(W):
            if nkeys[nid, w] != kw[w]:
                same = False
                break
        if same:
            return nid
        slot = (slot + 1) & hmask


@njit(cache=NUMBA_CACHE, nogil=True)
def store_add(st, kw, count, leaf, pred, lb, solved):
    """Append a node (the caller checked it is absent); -1 if the store is full."""
    meta = st[ST_META]
    nid = meta[0]
    nkeys = st[ST_KEYS]
    if nid >= nkeys.shape[0]:
        return -1
    hidx = st[ST_HIDX]
    hmask = np.int64(hidx.shape[0] - 1)
    slot = _slot_of(kw, hmask)
    while hidx[slot] >= 0:
        slot = (slot + 1) & hmask
    hidx[slot] = nid
    for w in range(kw.shape[0]):
        nkeys[nid, w] = kw[w]
    st[ST_COUNT][nid] = count
    st[ST_LEAF][nid] = leaf
    st[ST_PRED][nid] = pred
    st[ST_LB][nid] = lb
    st[ST_UB][nid] = leaf
    st[ST_SPLIT][nid] = -1
    st[ST_SOLVED][nid] = 1 if solved else 0
    st[ST_PEND][nid, 0] = 0
    meta[0] = nid + 1
    return nid


@njit(cache=NUMBA_CACHE, nogil=True)
def leaf_stats_words(kw, masks, weights, costs, diff, K):
    """(count, leaf loss, equivalent-points loss, potential, prediction) of a capture."""
    W = kw.shape[0]
    nw = weights.shape[0]
    dist = np.empty(K)
    count = 0.0
    pot = 0.0
    for k in range(K):
        acc = np.uint64(0)
        for w in range(W):
            acc += _popcount64(kw[w] & masks[k, w])
        dist[k] = acc
        count += dist[k]
        pot += diff[k] * dist[k]
    n_min = 0.0
    for r in range(nw):
        acc = np.uint64(0)
        for w in range(W):
            acc += _popcount64(kw[w] & masks[K + r, w])
        n_min += weights[r] * acc
    best = 1e300
    pred = 0
    for p in range(K):
        acc = 0.0
        for k in range(K):
            acc += costs[p, k] * dist[k]
        if acc < best:
            best = acc
            pred = p
    return count, best, n_min, pot, pred


@njit(cache=NUMBA_CACHE, nogil=True)
def make_node(st, dat, pf, pi, kw):
    """Id of the capture ``kw``, created with the leaf-only bounds if absent (-1: full)."""
    nid = store_find(st, kw)
    if nid >= 0:
        return nid
    K = pi[PI_K]
    lam = pf[PF_LAM]
    count, max_loss, min_loss, potential, pred = leaf_stats_words(kw, dat[DT_MASKS], dat[DT_WEIGHTS], dat[DT_COSTS],
                                                                  dat[DT_DIFF], K)
    leaf_risk = max_loss + lam
    if count <= 1.0 or max_loss - min_loss < lam or potential < 2.0 * lam:
        return store_add(st, kw, int(count), leaf_risk, pred, leaf_risk, True)
    return store_add(st, kw, int(count), leaf_risk, pred, min(leaf_risk, min_loss + 2.0 * lam), False)


@njit(cache=NUMBA_CACHE, nogil=True)
def child_key(st, dat, parent, f, left, out):
    """Words of the child of ``parent`` under feature ``f`` (left: f true)."""
    nkeys = st[ST_KEYS]
    F = dat[DT_F]
    for w in range(out.shape[0]):
        if left:
            out[w] = nkeys[parent, w] & F[f, w]
        else:
            out[w] = nkeys[parent, w] & ~F[f, w]


@njit(cache=NUMBA_CACHE, nogil=True)
def child_node(st, dat, pf, pi, parent, f, left, leaf, lb, solved, pred, out):
    """Child of ``parent`` under ``f`` with the kernel's statistics (existing node kept)."""
    child_key(st, dat, parent, f, left, out)
    nid = store_find(st, out)
    if nid >= 0:
        return nid
    count = 0
    for w in range(out.shape[0]):
        count += int(_popcount64(out[w]))
    return store_add(st, out, count, leaf, pred, lb, solved)


@njit(cache=NUMBA_CACHE, nogil=True)
def set_child_tree(st, cn, feats, i, io, kind, arg, jj, ub):
    """Record a child's best {leaf, 2, 3, (2,2)}-leaf tree (kind 0..3): its split now, the
    grandchildren's splits as pending (materialised at extraction)."""
    st[ST_UB][cn] = ub
    npend = st[ST_PEND]
    npend[cn, 0] = 0
    if kind == 0:
        st[ST_SPLIT][cn] = -1
    elif kind == 1:
        st[ST_SPLIT][cn] = feats[jj]
    elif kind == 2:
        st[ST_SPLIT][cn] = feats[arg[i, io, 0]]
        npend[cn, 0] = 2
        npend[cn, 1] = arg[i, io, 1]
        npend[cn, 2] = feats[arg[i, io, 2]]
    else:
        st[ST_SPLIT][cn] = feats[arg[i, io, 3]]
        npend[cn, 0] = 3
        npend[cn, 1] = feats[arg[i, io, 4]]
        npend[cn, 2] = feats[arg[i, io, 5]]


@njit(cache=NUMBA_CACHE, nogil=True)
def column_dp_chain(st, dat, pf, pi, node, feats_g, Lg, dist, exact, best_in, kw_buf, kw_buf2):
    """Single-column segmentation DP over the thresholds ``feats_g`` (increasing) with left
    counts ``Lg``; materialises the chain when it beats ``best_in`` (or exact).  Returns
    (value, root split or -1)."""
    M = Lg.shape[0]
    K = dist.shape[0]
    lam = pf[PF_LAM]
    S = np.empty((M + 2, K))
    for k in range(K):
        S[0, k] = 0.0
        S[M + 1, k] = dist[k]
    for i in range(M):
        for k in range(K):
            S[i + 1, k] = dist[k] - Lg[i, k]
    best = np.empty(M + 2)
    back = np.empty(M + 2, dtype=np.int64)
    if pi[PI_UNIFORM] == 1:
        val = segment_dp_uniform(S, pf[PF_UW], lam, best, back)
    else:
        val = segment_dp(S, dat[DT_COSTS], lam, best, back)
    if not (val < best_in - EPS or exact):
        return val, -2
    # boundaries (1-based threshold positions), top first
    nb = 0
    i = M + 1
    while True:
        j = back[i]
        if j <= 0:
            break
        nb += 1
        i = j
    bounds = np.empty(nb, dtype=np.int64)
    i = M + 1
    t = nb - 1
    while True:
        j = back[i]
        if j <= 0:
            break
        bounds[t] = j
        t -= 1
        i = j
    if nb == 0:
        return val, -1
    # segment leaf risks and suffix sums
    seg_val = np.empty(nb + 1)
    costs = dat[DT_COSTS]
    for t in range(nb + 1):
        lo = 0 if t == 0 else bounds[t - 1]
        hi = M + 1 if t == nb else bounds[t]
        c = 1e300
        for p in range(K):
            acc = 0.0
            for k in range(K):
                acc += costs[p, k] * (S[hi, k] - S[lo, k])
            if acc < c:
                c = acc
        seg_val[t] = c + lam
    suffix = np.zeros(nb + 2)
    for t in range(nb, -1, -1):
        suffix[t] = suffix[t + 1] + seg_val[t]
    # Every node of the chain is created before any structure is written: a node's value
    # is only consistent once its left child carries the rest of the chain, so a chain
    # cut by a full store (make_node -1) must leave nothing behind.
    rids = np.empty(nb, dtype=np.int64)
    lids = np.empty(nb, dtype=np.int64)
    cur = node
    for t in range(nb):
        f = feats_g[bounds[t] - 1]
        child_key(st, dat, cur, f, True, kw_buf)
        child_key(st, dat, cur, f, False, kw_buf2)
        rn = make_node(st, dat, pf, pi, kw_buf2)
        ln = make_node(st, dat, pf, pi, kw_buf)
        if rn < 0 or ln < 0:
            return val, -3
        rids[t] = rn
        lids[t] = ln
        cur = ln
    cur = node
    root_split = feats_g[bounds[0] - 1]
    nub = st[ST_UB]; nlb = st[ST_LB]; nsplit = st[ST_SPLIT]; nsolved = st[ST_SOLVED]; npend = st[ST_PEND]
    for t in range(nb):
        f = feats_g[bounds[t] - 1]
        rn = rids[t]
        ln = lids[t]
        if suffix[t] < nub[cur] - EPS or (exact and cur != node and suffix[t] <= nub[cur] + EPS):
            if suffix[t] < nub[cur]:
                nub[cur] = suffix[t]
            if nsplit[cur] != f:
                npend[cur, 0] = 0
            nsplit[cur] = f
        if exact:
            if suffix[t] < nub[cur]:
                nub[cur] = suffix[t]
            nlb[cur] = nub[cur]
            nsolved[cur] = 1
            if seg_val[t] <= nub[rn] + EPS:
                # (a node with a strictly better structure keeps it: its bound stays valid)
                if seg_val[t] < nub[rn]:
                    nub[rn] = seg_val[t]
                if nsplit[rn] != -1:
                    npend[rn, 0] = 0
                nsplit[rn] = -1
            nlb[rn] = nub[rn]
            nsolved[rn] = 1
        cur = ln
    if exact:
        if seg_val[nb] <= nub[cur] + EPS:
            if seg_val[nb] < nub[cur]:
                nub[cur] = seg_val[nb]
            if nsplit[cur] != -1:
                npend[cur, 0] = 0
            nsplit[cur] = -1
        nlb[cur] = nub[cur]
        nsolved[cur] = 1
    return val, root_split


@njit(cache=NUMBA_CACHE, nogil=True)
def _count_words(kw):
    c = np.uint64(0)
    for w in range(kw.shape[0]):
        c += _popcount64(kw[w])
    return c


@njit(cache=NUMBA_CACHE, nogil=True)
def _propagate(lb_arr, gidx, pot, i, v, bound):
    """Column-wide similar-support propagation of a proven split bound (see the Python solver)."""
    if v <= bound + EPS:
        return
    g = gidx[i]
    if g < 0:
        return
    a = i
    while a > 0 and gidx[a - 1] == g:
        a -= 1
    b = i + 1
    n = gidx.shape[0]
    while b < n and gidx[b] == g:
        b += 1
    for t in range(a, b):
        cand = v - abs(pot[t] - pot[i])
        if cand > lb_arr[t]:
            lb_arr[t] = cand


# ---------------------------------------------------------------------------
# Iterative form of the compiled search: an explicit stack of frames (one per depth) with
# per-depth workspaces, so no numba recursion is needed.
# ---------------------------------------------------------------------------
MAXD = 48
# integer frame fields
FI_NODE, FI_PHASE, FI_NV, FI_NKEEP, FI_OI, FI_II, FI_LN, FI_RN, FI_FIRST, FI_SECOND, FI_PRUNED, FI_HAVE_D2, FI_HAVE_D3, FI_SIM, FI_BEST_SPLIT, FI_PARENT_NV, FI_VALID, FI_CACHED_NODE = range(18)
# float frame fields
FF_BUDGET, FF_BEST, FF_BOUND, FF_MINPR, FF_LBMAX, FF_STEP, FF_BF, FF_BS, FF_EXACT_BELOW, FF_LEAF, FF_OUT_KIND, FF_OUT_VALUE, FF_BD2, FF_LBGE4, FF_LBREST, FF_LB0 = range(16)
# workspace tuple indices
WS_FI, WS_FF, WS_IO, WS_FO, WS_BO, WS_L, WS_DIST, WS_GIDX, WS_LBARR, WS_ARG3, WS_F3, WS_K3, WS_KW, WS_KW2, WS_ROOTFEATS, WS_SLOT = range(16)
NSLOT = 2 * MAXD      # two frame slots per depth (one per sibling), slot 0 is the root


_WS_POOL = {}
STORE_CAPACITY = 1 << 14      # initial node store; grown by doubling when full


def make_workspace(m, K, W):
    """Per-thread workspace; pooled by shape across fits (the frame cache flags are reset)."""
    key = (m, K, W)
    pool = _WS_POOL.setdefault(key, [])
    if pool:
        ws = pool.pop()
        ws[WS_FI][:, FI_VALID] = 0
        ws[WS_SLOT][:] = 0
        return ws
    m3 = min(m, TRIPLE_MAX_NV)     # the depth-3 stage only runs on narrow nodes
    return (np.zeros((NSLOT, 18), dtype=np.int64), np.zeros((NSLOT, 16)), np.zeros((NSLOT, 7, m), dtype=np.int64),
            np.zeros((NSLOT, 14, m)), np.zeros((NSLOT, 2, m), dtype=np.bool_), np.zeros((NSLOT, m, K)),
            np.zeros((NSLOT, K + 1)), np.zeros((NSLOT, m), dtype=np.int64), np.zeros((NSLOT, m)),
            np.zeros((NSLOT, m3, 2, 6), dtype=np.int64), np.zeros((NSLOT, 4, m)), np.zeros((NSLOT, 2, m), dtype=np.int64),
            np.zeros(W, dtype=np.uint64), np.zeros(W, dtype=np.uint64), np.arange(m, dtype=np.int64),
            np.zeros(MAXD + 1, dtype=np.int64))


def release_workspace(ws, m, K, W):
    _WS_POOL.setdefault((m, K, W), []).append(ws)


@njit(cache=NUMBA_CACHE, nogil=True)
def _frame_feats(ws, d):
    """Allowed features of the frame at depth d: the parent's surviving candidates."""
    if d == 0:
        return ws[WS_ROOTFEATS]
    pp = ws[WS_SLOT][d - 1]
    pn = ws[WS_FI][pp, FI_NV]
    return ws[WS_IO][pp, 0, :pn]


@njit(cache=NUMBA_CACHE, nogil=True)
def _expand_frame(st, dat, pf, pi, ws, d):
    """Phase 0 of a frame: the expansion and kernel stages of ``_solve`` up to the candidate
    loop.  Returns True when the node is resolved (frame done), else the frame is ready
    for its loop (fields filled)."""
    FI = ws[WS_FI]; FF = ws[WS_FF]
    ps = ws[WS_SLOT][d]
    FI[ps, FI_VALID] = 0
    node = FI[ps, FI_NODE]
    budget = FF[ps, FF_BUDGET]
    nlb = st[ST_LB]; nub = st[ST_UB]; nsplit = st[ST_SPLIT]; nsolved = st[ST_SOLVED]
    npend = st[ST_PEND]; nleaf = st[ST_LEAF]; ncount = st[ST_COUNT]; nkeys = st[ST_KEYS]
    meta = st[ST_META]
    lam = pf[PF_LAM]
    K = pi[PI_K]
    W = pi[PI_W]
    F = dat[DT_F]
    group_of = dat[DT_GROUP]
    features = _frame_feats(ws, d)
    mf = features.shape[0]
    io = ws[WS_IO][ps, :, :mf]
    fo = ws[WS_FO][ps, :, :mf]
    bo = ws[WS_BO][ps, :, :mf]
    L = ws[WS_L][ps, :mf]
    dist = ws[WS_DIST][ps]
    kw = nkeys[node]
    nv, n_cand, i0, min_rejected, ran, i_d2, best_d2, lb_ge4, M, Fc = expand_kernel(
        F, features, group_of, kw, dat[DT_MASKS], dat[DT_WEIGHTS], dat[DT_COSTS], dat[DT_DIFF], lam,
        min(budget, nub[node]), pi[PI_EXCH] == 1 and pi[PI_GROUPS] == 1, io, fo, bo, L, dist)
    leaf_risk = nleaf[node]
    FF[ps, FF_LEAF] = leaf_risk
    if nv == 0:
        nlb[node] = leaf_risk
        nub[node] = leaf_risk
        nsplit[node] = -1
        npend[node, 0] = 0
        nsolved[node] = 1
        return True
    if ran and min(leaf_risk, min(best_d2, lb_ge4)) > budget + EPS:
        v = min(leaf_risk, min(best_d2, lb_ge4))
        if v > nlb[node]:
            nlb[node] = v
        return True
    FI[ps, FI_NV] = nv
    feats = io[0, :nv]; order_buf = io[1, :nv]; j_l = io[2, :nv]; j_r = io[3, :nv]
    l_pred = io[4, :nv]; r_pred = io[5, :nv]
    l_leaf = fo[0, :nv]; l_lb = fo[1, :nv]; r_leaf = fo[2, :nv]; r_lb = fo[3, :nv]
    split_lb = fo[5, :nv]; split_ub = fo[6, :nv]
    l_ub2 = fo[11, :nv]; r_ub2 = fo[12, :nv]; split_ub2 = fo[13, :nv]
    l_solved = bo[0, :nv]; r_solved = bo[1, :nv]
    n_min = dist[K]
    distK = dist[:K]
    Lv = L[:nv]
    kw_buf = ws[WS_KW]
    kw_buf2 = ws[WS_KW2]
    best = nub[node]
    best_split = nsplit[node]
    i = i0
    if split_ub[i] < best - EPS:
        best = split_ub[i]
        best_split = feats[i]
    gidx = ws[WS_GIDX][ps, :nv]
    for t in range(nv):
        gidx[t] = group_of[feats[t]]
    if pi[PI_GROUPS] == 1:
        single = gidx[0] >= 0
        if single:
            for t in range(1, nv):
                if gidx[t] != gidx[0]:
                    single = False
                    break
        large = ncount[node] == int(pf[PF_N])
        if single or large:
            a = 0
            while a < nv:
                b = a + 1
                while b < nv and gidx[b] == gidx[a]:
                    b += 1
                if gidx[a] >= 0 and (single or b - a >= 2):
                    val, rs = column_dp_chain(st, dat, pf, pi, node, feats[a:b], Lv[a:b], distK, single, best,
                                              kw_buf, kw_buf2)
                    if rs == -3:
                        meta[2] = 2
                        return True
                    if val < best - EPS and rs != -2:
                        best = val
                        best_split = rs
                    if single:
                        if best < nub[node]:
                            nub[node] = best
                        if best_split >= 0 or nub[node] <= leaf_risk + EPS:
                            new_split = best_split if best < leaf_risk - EPS else -1
                            if new_split != nsplit[node]:
                                npend[node, 0] = 0
                            nsplit[node] = new_split
                        nlb[node] = nub[node]
                        nsolved[node] = 1
                        return True
                a = b
    bound = min(budget, best)
    n_keep = n_cand
    min_pruned = min_rejected
    have_d3 = False
    f3 = ws[WS_F3][ps, :, :nv]
    k3 = ws[WS_K3][ps, :, :nv]
    arg3 = ws[WS_ARG3][ps, :min(nv, ws[WS_ARG3].shape[1])]
    FF[ps, FF_BD2] = best_d2
    FF[ps, FF_LBGE4] = lb_ge4
    FF[ps, FF_LBREST] = 1e300
    FI[ps, FI_HAVE_D2] = 1 if ran else 0
    FI[ps, FI_HAVE_D3] = 0
    FI[ps, FI_CACHED_NODE] = node
    if ran:
        if best_d2 < best - EPS:
            best = best_d2
            best_split = feats[i_d2]
            for side in range(2):
                ii = i_d2
                if side == 0:
                    cn = child_node(st, dat, pf, pi, node, feats[ii], True, l_leaf[ii], l_lb[ii], l_solved[ii], l_pred[ii], kw_buf)
                    ub2 = l_ub2[ii]; jj = j_l[ii]; lf = l_leaf[ii]
                else:
                    cn = child_node(st, dat, pf, pi, node, feats[ii], False, r_leaf[ii], r_lb[ii], r_solved[ii], r_pred[ii], kw_buf)
                    ub2 = r_ub2[ii]; jj = j_r[ii]; lf = r_leaf[ii]
                if cn < 0:
                    meta[2] = 2
                    return True
                if ub2 < nub[cn] - EPS:
                    nub[cn] = ub2
                    nsplit[cn] = feats[jj] if (jj >= 0 and ub2 < lf - EPS) else -1
                    npend[cn, 0] = 0
            bound = min(budget, best)
        if best_d2 <= lb_ge4 and best_d2 <= budget + EPS and best_d2 <= leaf_risk + EPS:
            if best_d2 < nub[node]:
                nub[node] = best_d2
            if best_split != nsplit[node]:
                npend[node, 0] = 0
            nsplit[node] = best_split
            nlb[node] = nub[node]
            nsolved[node] = 1
            return True
        if budget < 4.0 * lam - EPS:
            nub[node] = best
            if best_split != nsplit[node]:
                npend[node, 0] = 0
            nsplit[node] = best_split
            if best <= budget + EPS:
                nlb[node] = best
                nsolved[node] = 1
            else:
                v = min(best, 4.0 * lam)
                if v > nlb[node]:
                    nlb[node] = v
            return True
        if (pi[PI_D3] == 1 and nv <= TRIPLE_MAX_NV and K * W <= D3_MAX_KW
                and nv * nv * nv / 6.0 * (K * W + 150.0) <= TRIPLE_MAX_OPS
                and budget < n_min + D3_MAX_LEAVES * lam - EPS
                and ncount[node] <= D3_MAX_COUNT_LAM * lam * pf[PF_N]):
            val3 = np.empty((nv, 2, 6))
            depth3_triples(Fc, feats, io[6, :nv], M[:K], dat[DT_COSTS], pf[PF_UW], lam, distK, Lv, val3, arg3)
            i3, best_d3, lb_rest = depth3_bounds(l_leaf, fo[7, :nv], fo[9, :nv], r_leaf, fo[8, :nv], fo[10, :nv], val3, lam,
                                                 f3[0], f3[1], f3[2], f3[3], k3[0], k3[1])
            have_d3 = True
            FI[ps, FI_HAVE_D3] = 1
            FF[ps, FF_LBREST] = lb_rest
            if best_d3 < best - EPS:
                best = best_d3
                best_split = feats[i3]
                for side in range(2):
                    if side == 0:
                        cn = child_node(st, dat, pf, pi, node, feats[i3], True, l_leaf[i3], l_lb[i3], l_solved[i3], l_pred[i3], kw_buf)
                        ub3 = f3[0, i3]; kind = k3[0, i3]; jj = j_l[i3]
                    else:
                        cn = child_node(st, dat, pf, pi, node, feats[i3], False, r_leaf[i3], r_lb[i3], r_solved[i3], r_pred[i3], kw_buf)
                        ub3 = f3[1, i3]; kind = k3[1, i3]; jj = j_r[i3]
                    if cn < 0:
                        meta[2] = 2
                        return True
                    if ub3 < nub[cn] - EPS:
                        set_child_tree(st, cn, feats, i3, 1 - side, kind, arg3, jj, ub3)
                bound = min(budget, best)
            lb3 = min(leaf_risk, lb_rest)
            if lb3 > budget + EPS:
                nub[node] = best
                if best_split != nsplit[node]:
                    npend[node, 0] = 0
                nsplit[node] = best_split
                if lb3 > nlb[node]:
                    nlb[node] = lb3
                return True
            if best <= lb3 + EPS:
                nub[node] = best
                if best_split != nsplit[node]:
                    npend[node, 0] = 0
                nsplit[node] = best_split
                nlb[node] = nub[node]
                nsolved[node] = 1
                return True
            for t in range(nv):
                if f3[2, t] > l_lb[t]:
                    l_lb[t] = f3[2, t]
                if f3[3, t] > r_lb[t]:
                    r_lb[t] = f3[3, t]
                split_lb[t] = l_lb[t] + r_lb[t]
                split_ub2[t] = f3[0, t] + f3[1, t]
        else:
            for t in range(nv):
                f3[0, t] = l_ub2[t]
                f3[1, t] = r_ub2[t]
        FI[ps, FI_VALID] = 1
        n_keep, min_dropped = refilter_candidates(order_buf, n_cand, split_lb, split_ub2, bound + EPS)
        if min_dropped < min_pruned:
            min_pruned = min_dropped
    sim = pi[PI_SIM] == 1 and pi[PI_GROUPS] == 1
    lb_arr = ws[WS_LBARR][ps, :nv]
    for t in range(nv):
        lb_arr[t] = split_lb[t]
    FI[ps, FI_NKEEP] = n_keep
    FI[ps, FI_OI] = 0
    FI[ps, FI_HAVE_D2] = 1 if ran else 0
    FI[ps, FI_HAVE_D3] = 1 if have_d3 else 0
    FI[ps, FI_SIM] = 1 if sim else 0
    FI[ps, FI_BEST_SPLIT] = best_split
    FI[ps, FI_LN] = -1
    FI[ps, FI_RN] = -1
    FF[ps, FF_BEST] = best
    FF[ps, FF_BOUND] = bound
    FF[ps, FF_MINPR] = min_pruned
    FF[ps, FF_LBMAX] = max_pair(l_lb, r_lb)
    FF[ps, FF_EXACT_BELOW] = 4.0 * lam if have_d3 else 3.0 * lam
    return False


@njit(cache=NUMBA_CACHE, nogil=True)
def _rearm_frame(st, dat, pf, pi, ws, d):
    """Re-entry of a cached frame with a new budget: the budget-dependent steps of
    _expand_frame only (probe checks, candidate filtering and ordering, exact-resolution
    tests).  Returns True when the node is resolved."""
    FI = ws[WS_FI]; FF = ws[WS_FF]
    ps = ws[WS_SLOT][d]
    node = FI[ps, FI_NODE]
    budget = FF[ps, FF_BUDGET]
    nlb = st[ST_LB]; nub = st[ST_UB]; nsplit = st[ST_SPLIT]; nsolved = st[ST_SOLVED]; npend = st[ST_PEND]
    lam = pf[PF_LAM]
    leaf_risk = FF[ps, FF_LEAF]
    ran = FI[ps, FI_HAVE_D2] == 1
    have_d3 = FI[ps, FI_HAVE_D3] == 1
    best_d2 = FF[ps, FF_BD2]
    lb_ge4 = FF[ps, FF_LBGE4]
    lb_rest = FF[ps, FF_LBREST]
    if ran and min(leaf_risk, min(best_d2, lb_ge4)) > budget + EPS:
        v = min(leaf_risk, min(best_d2, lb_ge4))
        if v > nlb[node]:
            nlb[node] = v
        return True
    if have_d3 and min(leaf_risk, lb_rest) > budget + EPS:
        v = min(leaf_risk, lb_rest)
        if v > nlb[node]:
            nlb[node] = v
        return True
    nv = FI[ps, FI_NV]
    io = ws[WS_IO][ps]; fo = ws[WS_FO][ps]
    feats = io[0, :nv]; order_buf = io[1, :nv]
    l_leaf = fo[0, :nv]; l_lb = fo[1, :nv]; r_leaf = fo[2, :nv]; r_lb = fo[3, :nv]
    split_lb = fo[5, :nv]; split_ub = fo[6, :nv]; split_ub2 = fo[13, :nv]
    gidx = ws[WS_GIDX][ps, :nv]
    best = nub[node]
    best_split = nsplit[node]
    n_cand, i0, min_rejected = prep_candidates(gidx, l_leaf, l_lb, r_leaf, r_lb, min(budget, best),
                                               pi[PI_EXCH] == 1 and pi[PI_GROUPS] == 1, split_lb, split_ub, order_buf)
    if n_cand > 0 and split_ub[i0] < best - EPS:
        best = split_ub[i0]
        best_split = feats[i0]
    bound = min(budget, best)
    n_keep = n_cand
    min_pruned = min_rejected
    if ran:
        if best_d2 <= lb_ge4 and best_d2 <= budget + EPS and best_d2 <= leaf_risk + EPS:
            if best_d2 < nub[node]:
                nub[node] = best_d2
            if best_split != nsplit[node]:
                npend[node, 0] = 0
            nsplit[node] = best_split
            nlb[node] = nub[node]
            nsolved[node] = 1
            return True
        if budget < 4.0 * lam - EPS:
            nub[node] = best
            if best_split != nsplit[node]:
                npend[node, 0] = 0
            nsplit[node] = best_split
            if best <= budget + EPS:
                nlb[node] = best
                nsolved[node] = 1
            else:
                v = min(best, 4.0 * lam)
                if v > nlb[node]:
                    nlb[node] = v
            return True
        if have_d3:
            lb3 = min(leaf_risk, lb_rest)
            if best <= lb3 + EPS:
                nub[node] = best
                if best_split != nsplit[node]:
                    npend[node, 0] = 0
                nsplit[node] = best_split
                nlb[node] = nub[node]
                nsolved[node] = 1
                return True
        n_keep, min_dropped = refilter_candidates(order_buf, n_cand, split_lb, split_ub2, bound + EPS)
        if min_dropped < min_pruned:
            min_pruned = min_dropped
    lb_arr = ws[WS_LBARR][ps, :nv]
    for t in range(nv):
        if split_lb[t] > lb_arr[t]:
            lb_arr[t] = split_lb[t]
    FI[ps, FI_NKEEP] = n_keep
    FI[ps, FI_OI] = 0
    FI[ps, FI_SIM] = 1 if (pi[PI_SIM] == 1 and pi[PI_GROUPS] == 1) else 0
    FI[ps, FI_BEST_SPLIT] = best_split
    FI[ps, FI_LN] = -1
    FI[ps, FI_RN] = -1
    FF[ps, FF_BEST] = best
    FF[ps, FF_BOUND] = bound
    FF[ps, FF_MINPR] = min_pruned
    FF[ps, FF_LBMAX] = max_pair(l_lb, r_lb)
    FF[ps, FF_EXACT_BELOW] = 4.0 * lam if have_d3 else 3.0 * lam
    return False


@njit(cache=NUMBA_CACHE, nogil=True)
def _finish_frame(st, ws, d):
    """Epilogue of a frame's candidate loop."""
    FI = ws[WS_FI]; FF = ws[WS_FF]
    ps = ws[WS_SLOT][d]
    node = FI[ps, FI_NODE]
    best = FF[ps, FF_BEST]
    best_split = FI[ps, FI_BEST_SPLIT]
    if best_split != st[ST_SPLIT][node]:
        st[ST_PEND][node, 0] = 0
    st[ST_UB][node] = best
    st[ST_SPLIT][node] = best_split
    if best <= FF[ps, FF_BUDGET] + EPS:
        st[ST_LB][node] = best
        st[ST_SOLVED][node] = 1
    else:
        v = max(min(best, FF[ps, FF_MINPR]), FF[ps, FF_LBMAX])
        if v > st[ST_LB][node]:
            st[ST_LB][node] = v


@njit(cache=NUMBA_CACHE, nogil=True)
def _flush_frames(st, ws, d):
    """On an interruption, record each open frame's incumbent (achievable, its tree is in
    the store) on its node so the tree returned at a time limit is the best one found."""
    FI = ws[WS_FI]; FF = ws[WS_FF]
    for k in range(d + 1):
        pk = ws[WS_SLOT][k]
        if FI[pk, FI_PHASE] == 0:
            continue
        node = FI[pk, FI_NODE]
        best = FF[pk, FF_BEST]
        if best < st[ST_UB][node] - EPS:
            st[ST_UB][node] = best
            if FI[pk, FI_BEST_SPLIT] != st[ST_SPLIT][node]:
                st[ST_PEND][node, 0] = 0
            st[ST_SPLIT][node] = FI[pk, FI_BEST_SPLIT]


@njit(cache=NUMBA_CACHE, nogil=True)
def solve_iter(st, dat, pf, pi, ws, root, budget, only, shared, sh, tid):
    """Iterative ``_solve``: frames on an explicit stack, the candidate loop as phases.

    ``only >= 0`` (parallel workers): the depth-0 frame is already expanded; only the
    candidate at position ``only`` of its order is processed, the depth-0 bound is capped
    by ``shared[0]`` (the incumbent shared between workers), and the outcome is written to
    ``FF[0, FF_OUT_KIND]`` (1 solved / 0 pruned) and ``FF[0, FF_OUT_VALUE]``."""
    FI = ws[WS_FI]; FF = ws[WS_FF]
    nlb = st[ST_LB]; nub = st[ST_UB]; nsplit = st[ST_SPLIT]; nsolved = st[ST_SOLVED]; npend = st[ST_PEND]
    meta = st[ST_META]
    lam = pf[PF_LAM]
    look_ahead = pi[PI_LOOKAHEAD] == 1
    kw_buf = ws[WS_KW]; kw_buf2 = ws[WS_KW2]
    d = 0
    if only >= 0:
        FI[0, FI_OI] = only
        FI[0, FI_NKEEP] = only + 1
        FI[0, FI_PHASE] = 1
        FF[0, FF_OUT_KIND] = -1.0
    else:
        FI[0, FI_NODE] = root
        FF[0, FF_BUDGET] = budget
        FI[0, FI_PHASE] = 0
    ws[WS_SLOT][0] = 0
    while d >= 0:
        ps = ws[WS_SLOT][d]
        node = FI[ps, FI_NODE]
        phase = FI[ps, FI_PHASE]
        if only >= 0 and shared[0] < FF[0, FF_BOUND]:
            # another thread improved the incumbent: tighten the whole stack
            FF[0, FF_BOUND] = shared[0]
            if shared[0] < FF[0, FF_BEST]:
                FF[0, FF_BEST] = shared[0]
            _tighten_stack(st, ws, d, look_ahead)
        if phase == 0:
            # entry of _solve
            if nsolved[node] == 1 or nlb[node] > FF[ps, FF_BUDGET] + EPS:
                d -= 1
                continue
            sh_big = sh[SH_META][0] > 0 and st[ST_COUNT][node] >= sh[SH_META][2]
            if sh_big:
                # a bound, or the optimum, another thread proved for this subproblem
                v, x, owner = sh_lookup(sh, st[ST_KEYS][node], st[ST_COUNT][node])
                if owner >= 0:
                    # adopt the optimum: the tree lives in the owner's store (see extract)
                    if x < nub[node]:
                        nub[node] = x
                        nsplit[node] = SPLIT_EXTERN - owner
                        npend[node, 0] = 0
                    nlb[node] = nub[node]
                    nsolved[node] = 1
                    meta[6] += 1
                    d -= 1
                    continue
                if v > nlb[node]:
                    nlb[node] = v
                    if v > FF[ps, FF_BUDGET] + EPS:
                        d -= 1
                        continue
            FF[ps, FF_LB0] = nlb[node]
            if meta[1] >= meta[3]:
                meta[2] = 1
                _flush_frames(st, ws, d)
                return
            meta[1] += 1
            if FI[ps, FI_VALID] == 1 and FI[ps, FI_CACHED_NODE] == node:
                # the slot still holds this node's expansion: only the budget-dependent part
                if _rearm_frame(st, dat, pf, pi, ws, d):
                    if sh_big:
                        _sh_publish_node(sh, tid, st, node, FF[ps, FF_LB0], lam)
                    d -= 1
                    continue
            elif _expand_frame(st, dat, pf, pi, ws, d):
                if meta[2] != 0:
                    _flush_frames(st, ws, d)
                    return
                if sh_big:
                    _sh_publish_node(sh, tid, st, node, FF[ps, FF_LB0], lam)
                d -= 1
                continue
            FI[ps, FI_PHASE] = 1
            continue
        nv = FI[ps, FI_NV]
        io = ws[WS_IO][ps]; fo = ws[WS_FO][ps]; bo = ws[WS_BO][ps]
        feats = io[0, :nv]; order_buf = io[1, :nv]; j_l = io[2, :nv]; j_r = io[3, :nv]
        l_pred = io[4, :nv]; r_pred = io[5, :nv]
        l_leaf = fo[0, :nv]; l_lb = fo[1, :nv]; r_leaf = fo[2, :nv]; r_lb = fo[3, :nv]; l_pot = fo[4, :nv]
        split_lb = fo[5, :nv]
        l_solved = bo[0, :nv]; r_solved = bo[1, :nv]
        f3 = ws[WS_F3][ps]; k3 = ws[WS_K3][ps]; arg3 = ws[WS_ARG3][ps]
        gidx = ws[WS_GIDX][ps, :nv]; lb_arr = ws[WS_LBARR][ps, :nv]
        have_d2 = FI[ps, FI_HAVE_D2] == 1
        have_d3 = FI[ps, FI_HAVE_D3] == 1
        sim = FI[ps, FI_SIM] == 1
        bound = FF[ps, FF_BOUND]
        budget_f = FF[ps, FF_BUDGET]
        first = FI[ps, FI_FIRST]; second = FI[ps, FI_SECOND]
        if phase == 1:
            # next candidate
            oi = FI[ps, FI_OI]
            if oi >= FI[ps, FI_NKEEP]:
                if d == 0 and only >= 0:
                    return
                _finish_frame(st, ws, d)
                if sh[SH_META][0] > 0 and st[ST_COUNT][node] >= sh[SH_META][2]:
                    _sh_publish_node(sh, tid, st, node, FF[ps, FF_LB0], lam)
                d -= 1
                continue
            FI[ps, FI_OI] = oi + 1
            ii = order_buf[oi]
            raw = split_lb[ii]
            if raw > bound + EPS:
                if raw < FF[ps, FF_MINPR]:
                    FF[ps, FF_MINPR] = raw
                if d == 0 and only >= 0:
                    FF[0, FF_OUT_KIND] = 0.0
                    FF[0, FF_OUT_VALUE] = raw
                    return
                _finish_frame(st, ws, d)
                if sh[SH_META][0] > 0 and st[ST_COUNT][node] >= sh[SH_META][2]:
                    _sh_publish_node(sh, tid, st, node, FF[ps, FF_LB0], lam)
                d -= 1
                continue
            cur = lb_arr[ii]
            if sim and cur > bound + EPS:
                if cur < FF[ps, FF_MINPR]:
                    FF[ps, FF_MINPR] = cur
                if d == 0 and only >= 0:
                    FF[0, FF_OUT_KIND] = 0.0
                    FF[0, FF_OUT_VALUE] = cur
                continue
            f = feats[ii]
            child_key(st, dat, node, f, True, kw_buf)
            ln = store_find(st, kw_buf)
            child_key(st, dat, node, f, False, kw_buf2)
            rn = store_find(st, kw_buf2)
            if ln < 0:
                llb = l_lb[ii]; lub = l_leaf[ii]
            else:
                llb = nlb[ln]; lub = nub[ln]
            if rn < 0:
                rlb = r_lb[ii]; rub = r_leaf[ii]
            else:
                rlb = nlb[rn]; rub = nub[rn]
            sub = lub + rub
            if sub < FF[ps, FF_BEST] - EPS:
                FF[ps, FF_BEST] = sub
                FI[ps, FI_BEST_SPLIT] = f
                bound = min(budget_f, sub)
                FF[ps, FF_BOUND] = bound
            slb = llb + rlb
            if slb < cur:
                slb = cur
            if slb > bound + EPS:
                if slb < FF[ps, FF_MINPR]:
                    FF[ps, FF_MINPR] = slb
                if sim:
                    _propagate(lb_arr, gidx, l_pot, ii, slb, bound)
                if d == 0 and only >= 0:
                    FF[0, FF_OUT_KIND] = 0.0
                    FF[0, FF_OUT_VALUE] = slb
                continue
            if ln < 0:
                ln = store_add(st, kw_buf, int(_count_words(kw_buf)), lub, l_pred[ii], llb, l_solved[ii])
            if rn < 0:
                rn = store_add(st, kw_buf2, int(_count_words(kw_buf2)), rub, r_pred[ii], rlb, r_solved[ii])
            if ln < 0 or rn < 0:
                meta[2] = 2
                _flush_frames(st, ws, d)
                return
            if have_d2:
                for side in range(2):
                    cn = ln if side == 0 else rn
                    ub2 = f3[0, ii] if side == 0 else f3[1, ii]
                    jj = j_l[ii] if side == 0 else j_r[ii]
                    lf = l_leaf[ii] if side == 0 else r_leaf[ii]
                    if ub2 < nub[cn] - EPS:
                        if have_d3:
                            kind = k3[0, ii] if side == 0 else k3[1, ii]
                            set_child_tree(st, cn, feats, ii, 1 - side, kind, arg3, jj, ub2)
                        else:
                            nub[cn] = ub2
                            nsplit[cn] = feats[jj] if (jj >= 0 and ub2 < lf - EPS) else -1
                            npend[cn, 0] = 0
                    lbi = l_lb[ii] if side == 0 else r_lb[ii]
                    if nlb[cn] < lbi:
                        nlb[cn] = lbi
                    if nsolved[cn] == 0 and nub[cn] <= nlb[cn] + EPS:
                        nlb[cn] = nub[cn]
                        nsolved[cn] = 1
            if nlb[ln] >= nlb[rn]:
                first = ln; second = rn
            else:
                first = rn; second = ln
            if have_d2:
                eb = FF[ps, FF_EXACT_BELOW]
                for side in range(2):
                    cn = first if side == 0 else second
                    other = second if side == 0 else first
                    if nsolved[cn] == 0 and bound - nlb[other] < eb - EPS and nub[cn] <= bound - nlb[other] + EPS:
                        nlb[cn] = nub[cn]
                        nsolved[cn] = 1
            FI[ps, FI_II] = ii; FI[ps, FI_LN] = ln; FI[ps, FI_RN] = rn
            FI[ps, FI_FIRST] = first; FI[ps, FI_SECOND] = second
            FI[ps, FI_PRUNED] = 0
            FF[ps, FF_STEP] = 2.0 * lam
            FI[ps, FI_PHASE] = 2
            continue
        ii = FI[ps, FI_II]
        if phase == 2:
            # deepening loop head
            if look_ahead and nsolved[first] == 0:
                bf = bound - nlb[second]
                FF[ps, FF_BF] = bf
                if nlb[first] > bf + EPS:
                    FI[ps, FI_PRUNED] = 1
                    FI[ps, FI_PHASE] = 7
                    continue
                FI[ps, FI_PHASE] = 3
                if d + 1 >= MAXD:
                    meta[2] = 3
                    _flush_frames(st, ws, d)
                    return
                cs = 2 * (d + 1) - 1
                ws[WS_SLOT][d + 1] = cs
                FI[cs, FI_NODE] = first; FF[cs, FF_BUDGET] = min(bf, nlb[first] + FF[ps, FF_STEP]); FI[cs, FI_PHASE] = 0
                d += 1
                continue
            FI[ps, FI_PHASE] = 5
            continue
        if phase == 3:
            bf = FF[ps, FF_BF]
            if nlb[first] > bf + EPS:
                FI[ps, FI_PRUNED] = 1
                FI[ps, FI_PHASE] = 7
                continue
            if nsolved[first] == 1:
                FI[ps, FI_PHASE] = 5
                continue
            bs = bound - nlb[first]
            FF[ps, FF_BS] = bs
            if nsolved[second] == 0:
                if nlb[second] > bs + EPS:
                    FI[ps, FI_PRUNED] = 1
                    FI[ps, FI_PHASE] = 7
                    continue
                FI[ps, FI_PHASE] = 4
                if d + 1 >= MAXD:
                    meta[2] = 3
                    _flush_frames(st, ws, d)
                    return
                cs = 2 * (d + 1)
                ws[WS_SLOT][d + 1] = cs
                FI[cs, FI_NODE] = second; FF[cs, FF_BUDGET] = min(bs, nlb[second] + FF[ps, FF_STEP]); FI[cs, FI_PHASE] = 0
                d += 1
                continue
            FF[ps, FF_STEP] = FF[ps, FF_STEP] * 2.0
            FI[ps, FI_PHASE] = 2
            continue
        if phase == 4:
            bs = FF[ps, FF_BS]
            if nlb[second] > bs + EPS:
                FI[ps, FI_PRUNED] = 1
                FI[ps, FI_PHASE] = 7
                continue
            FF[ps, FF_STEP] = FF[ps, FF_STEP] * 2.0
            FI[ps, FI_PHASE] = 2
            continue
        if phase == 5:
            # final solve of first (not pruned)
            FI[ps, FI_PHASE] = 6
            if d + 1 >= MAXD:
                meta[2] = 3
                _flush_frames(st, ws, d)
                return
            cs = 2 * (d + 1) - 1
            ws[WS_SLOT][d + 1] = cs
            FI[cs, FI_NODE] = first
            FF[cs, FF_BUDGET] = bound - nlb[second] if look_ahead else bound
            FI[cs, FI_PHASE] = 0
            d += 1
            continue
        if phase == 6 or phase == 7:
            if nlb[first] > FF[ps, FF_LBMAX]:
                FF[ps, FF_LBMAX] = nlb[first]
            if nlb[second] > FF[ps, FF_LBMAX]:
                FF[ps, FF_LBMAX] = nlb[second]
            if phase == 7 or nlb[first] > bound - nlb[second] + EPS:
                slb = nlb[first] + nlb[second]
                if slb < FF[ps, FF_MINPR]:
                    FF[ps, FF_MINPR] = slb
                if sim:
                    _propagate(lb_arr, gidx, l_pot, ii, slb, bound)
                if d == 0 and only >= 0:
                    FF[0, FF_OUT_KIND] = 0.0
                    FF[0, FF_OUT_VALUE] = slb
                FI[ps, FI_PHASE] = 1
                continue
            FI[ps, FI_PHASE] = 8
            if d + 1 >= MAXD:
                meta[2] = 3
                _flush_frames(st, ws, d)
                return
            cs = 2 * (d + 1)
            ws[WS_SLOT][d + 1] = cs
            FI[cs, FI_NODE] = second
            FF[cs, FF_BUDGET] = bound - nub[first] if look_ahead else bound
            FI[cs, FI_PHASE] = 0
            d += 1
            continue
        if phase == 8:
            if nlb[second] > FF[ps, FF_LBMAX]:
                FF[ps, FF_LBMAX] = nlb[second]
            if nlb[second] > bound - nub[first] + EPS:
                slb = nub[first] + nlb[second]
                if slb < FF[ps, FF_MINPR]:
                    FF[ps, FF_MINPR] = slb
                if sim:
                    _propagate(lb_arr, gidx, l_pot, ii, slb, bound)
                if d == 0 and only >= 0:
                    FF[0, FF_OUT_KIND] = 0.0
                    FF[0, FF_OUT_VALUE] = slb
                FI[ps, FI_PHASE] = 1
                continue
            value = nub[first] + nub[second]
            if value < FF[ps, FF_BEST] - EPS:
                FF[ps, FF_BEST] = value
                FI[ps, FI_BEST_SPLIT] = feats[ii]
                FF[ps, FF_BOUND] = min(budget_f, value)
            elif sim:
                _propagate(lb_arr, gidx, l_pot, ii, value, bound)
            if d == 0 and only >= 0:
                FF[0, FF_OUT_KIND] = 1.0
                FF[0, FF_OUT_VALUE] = value
            FI[ps, FI_PHASE] = 1
            continue


class CompiledOptimizer:
    """Driver of the compiled search: owns the array memo, re-enters the search in
    iteration chunks to honour the time and memory limits, grows the store on demand."""

    def __init__(self, data: BitDataset, regularization: float, *, groups=None, time_limit=0.0,
                 look_ahead=True, similar_support=True, feature_exchange=True, continuous_feature_exchange=True,
                 greedy_init=True, upperbound=0.0, engine="numba", memory_limit=0, verbose=False,
                 n_jobs=1, parallel_after=0.01, force_parallel=False, store_capacity=None):
        self.data = data
        self.lam = float(regularization)
        self.time_limit = float(time_limit)
        self.memory_limit = int(memory_limit)
        self.upperbound = float(upperbound)
        self.verbose = verbose
        self.iterations = 0
        self.optimal = False
        self.stop_reason = ""
        self.elapsed = 0.0
        warm_up()
        group_of = np.full(data.m, -1, dtype=np.int64)
        for gi, g in enumerate(groups or []):
            group_of[g] = gi
        has_groups = any(len(g) >= 2 for g in (groups or []))
        self.group_of = group_of
        # uniform-cost matrix
        w = float(data.mismatch_costs[0])
        uniform = data.zero_diagonal and data.equal_mismatch and bool(np.all(data.costs == (data.costs > 0) * w))
        uniform_w = w if uniform else 0.0
        # masks and weights: class masks, then the equivalent-points masks (see node_stats)
        masks = [data.target_words[k] for k in range(data.K)]
        weights = []
        pairs = [(data.minority_by_class_words[k], float(data.mismatch_costs[k])) for k in range(data.K)]
        if not data.zero_diagonal:
            pairs += [(data.majority_by_class_words[k], float(data.match_costs[k])) for k in range(data.K)]
        if data.zero_diagonal and data.equal_mismatch:
            masks.append(data.minority_words)
            weights.append(w)
        else:
            for mw, ww in pairs:
                if ww == 0.0:
                    continue
                masks.append(mw)
                weights.append(ww)
        self.dat = (data.F_words, group_of, np.ascontiguousarray(np.vstack(masks)), np.array(weights, dtype=np.float64),
                    data.costs, data.costs.T.copy(), data.diff_costs.copy())
        self.pf = np.array([self.lam, uniform_w, float(data.n)])
        self.pi = np.array([data.K, data.W, 1 if has_groups else 0, 1 if look_ahead else 0, 1 if similar_support else 0,
                            1 if continuous_feature_exchange else 0, 1, 1 if uniform else 0], dtype=np.int64)
        self._alloc(int(STORE_CAPACITY if store_capacity is None else store_capacity))
        self.ws = make_workspace(data.m, data.K, data.W)
        self._no_shared = np.array([1e300])
        self._no_table = no_shared_table(data.W)
        self.n_jobs = int(n_jobs)
        self.parallel_after = float(parallel_after)
        self.force_parallel = bool(force_parallel)
        self.parallel_tree = None
        self.stores = None
        self._thread_bytes = np.zeros(max(1, self.n_jobs))
        self._table_bytes = 0
        _compile_search()

    def _alloc(self, cap, old=None):
        self.st = self._new_store(cap, old)

    def _new_store(self, cap, old=None):
        W = self.data.W
        st = (np.empty((cap, W), dtype=np.uint64), np.full(2 * cap, -1, dtype=np.int64), np.zeros(cap, dtype=np.int64),
              np.zeros(cap), np.zeros(cap, dtype=np.int64), np.zeros(cap), np.zeros(cap), np.full(cap, -1, dtype=np.int64),
              np.zeros(cap, dtype=np.uint8), np.zeros((cap, 4), dtype=np.int64), np.zeros(8, dtype=np.int64))
        if old is not None:
            n = int(old[ST_META][0])
            for a in (ST_KEYS, ST_COUNT, ST_LEAF, ST_PRED, ST_LB, ST_UB, ST_SPLIT, ST_SOLVED, ST_PEND):
                st[a][:n] = old[a][:n]
            st[ST_META][:] = old[ST_META]
            _rebuild_index(st[ST_KEYS], st[ST_HIDX], n)
        return st

    def run(self):
        self.start_time = time.perf_counter()
        last_mem = self.start_time
        data = self.data
        kw = int_to_words(data.full, data.W).copy()
        root = make_node(self.st, self.dat, self.pf, self.pi, kw)
        features = np.arange(data.m, dtype=np.int64)
        st = self.st
        meta = st[ST_META]
        chunk = 500
        budget = None
        try:
            while True:
                meta[2] = 0
                meta[3] = meta[1] + chunk
                if budget is None:
                    budget = st[ST_UB][root] if self.upperbound <= 0.0 else min(st[ST_UB][root], self.upperbound)
                t0 = time.perf_counter()
                if self.force_parallel and self.n_jobs > 1:
                    meta[2] = 1
                else:
                    solve_iter(st, self.dat, self.pf, self.pi, self.ws, np.int64(root), float(budget), np.int64(-1), self._no_shared,
                               self._no_table, np.int64(0))
                dt = time.perf_counter() - t0
                if meta[2] == 0:
                    break
                if meta[2] == 1 and self.n_jobs > 1 and (self.force_parallel or (time.perf_counter() - self.start_time >= self.parallel_after
                                                                               and self._worth_parallel())):
                    self._run_parallel(root, budget)
                    break
                if meta[2] == 2:
                    self._alloc(st[ST_KEYS].shape[0] * 2, st)
                    st = self.st
                    meta = st[ST_META]
                    continue
                if meta[2] == 3:
                    raise TimeLimitReached("depth")
                # iteration budget hit: check the limits, re-enter with a chunk of ~50 ms
                now = time.perf_counter()
                if self.time_limit > 0.0 and now - self.start_time > self.time_limit:
                    raise TimeLimitReached("time")
                if self.memory_limit > 0 and now - last_mem > 0.5:
                    last_mem = now
                    if self._mem_bytes() > self.memory_limit:
                        raise TimeLimitReached("memory")
                if dt > 0.0:
                    # before the hand-off the chunk ends near ``parallel_after`` so the
                    # threads start on time; afterwards (or sequentially) ~50 ms chunks
                    target = 0.05
                    if self.n_jobs > 1:
                        target = max(0.002, self.parallel_after - (now - self.start_time))
                    chunk = int(min(max(chunk * target / dt, 100), 200000))
            self.optimal = st[ST_SOLVED][root] == 1
            self.stop_reason = "optimal" if self.optimal else "upperbound"
        except TimeLimitReached as exc:
            self.optimal = False
            self.stop_reason = str(exc)
        self.iterations = int(meta[1])
        self.elapsed = time.perf_counter() - self.start_time
        self.root = root
        return root

    def _mem_bytes(self) -> int:
        """Live bytes of this search: main store, the threads' stores, the shared table."""
        return _store_bytes(self.st) + int(self._thread_bytes.sum()) + self._table_bytes

    def release(self):
        """Return the workspace to the pool (after extraction)."""
        if self.ws is not None:
            release_workspace(self.ws, self.data.m, self.data.K, self.data.W)
            self.ws = None

    # ------------------------------------------------------------ parallel
    def _worth_parallel(self):
        """Hand-off gate: the root frame's candidate position extrapolates the remaining
        sequential work; the threads are worth their set-up (~2 ms) only if it is larger."""
        FI = self.ws[WS_FI]
        if FI[0, FI_PHASE] == 0:
            return True                     # root not expanded yet: unknown, go parallel
        oi = int(FI[0, FI_OI]); nk = int(FI[0, FI_NKEEP])
        remaining = nk - oi + 1             # the candidate in progress counts as remaining
        done = max(oi - 1, 1)
        if remaining < 2:
            return False
        elapsed = time.perf_counter() - self.start_time
        return elapsed * remaining / done >= 0.004

    def _run_parallel(self, root, budget):
        """Root-parallel phase with threads on private memo copies (see DESCRIPTION)."""
        st = self.st; ws = self.ws; meta = st[ST_META]
        FI = ws[WS_FI]; FF = ws[WS_FF]
        FI[0, FI_NODE] = root; FF[0, FF_BUDGET] = float(budget); FI[0, FI_PHASE] = 0
        meta[2] = 0; meta[3] = meta[1] + 10 ** 9
        t_x = time.perf_counter()
        if FI[0, FI_VALID] == 1 and FI[0, FI_CACHED_NODE] == root and ws[WS_SLOT][0] == 0:
            resolved = _rearm_frame(st, self.dat, self.pf, self.pi, ws, 0)   # cached expansion
        else:
            resolved = _expand_frame(st, self.dat, self.pf, self.pi, ws, 0)
        self.handoff_expand_time = time.perf_counter() - t_x
        if resolved:
            if meta[2] == 2:
                self._alloc(st[ST_KEYS].shape[0] * 2, st)
                return self._run_parallel(root, budget)
            return                      # resolved by the kernel stages
        FI[0, FI_PHASE] = 1
        n_keep = int(FI[0, FI_NKEEP])
        best = float(FF[0, FF_BEST]); best_split = int(FI[0, FI_BEST_SPLIT])
        if best < st[ST_UB][root] - EPS or best_split != st[ST_SPLIT][root]:
            st[ST_UB][root] = min(st[ST_UB][root], best)
            if best_split != st[ST_SPLIT][root]:
                st[ST_PEND][root, 0] = 0
            st[ST_SPLIT][root] = best_split
        min_pruned = float(FF[0, FF_MINPR]); child_lb_max = float(FF[0, FF_LBMAX])
        shared = np.array([best])
        lock = threading.Lock()
        tasks = queue.Queue()
        n_threads = min(self.n_jobs, n_keep)
        sh = get_shared_table(n_threads, self.data.W, max(2, self.data.n // SH_MIN_DIV))
        self._table_bytes = int(sh[SH_KEYS].nbytes + sh[SH_COUNTS].nbytes + sh[SH_LBS].nbytes + sh[SH_VALS].nbytes + sh[SH_USED].nbytes)
        thread_bytes = self._thread_bytes; thread_bytes[:] = 0
        stores = {}
        for pos in range(n_keep):
            tasks.put(pos)
        deadline = self.start_time + self.time_limit if self.time_limit > 0.0 else float("inf")
        results = []
        state = {"failure": "", "iters": 0}
        m = self.data.m; K = self.data.K; W = self.data.W

        def worker(tid):
            # private copies of the store (sized by its contents, grown on demand) and of the root frame
            cap_k = 1 << max(12, int(2 * int(st[ST_META][0]) - 1).bit_length())
            st_k = self._new_store(min(cap_k, st[ST_KEYS].shape[0]), st)
            thread_bytes[tid] = _store_bytes(st_k)
            ws_k = make_workspace(m, K, W)
            for a in (WS_FI, WS_FF, WS_IO, WS_FO, WS_BO, WS_L, WS_DIST, WS_GIDX, WS_LBARR, WS_ARG3, WS_F3, WS_K3):
                ws_k[a][0] = ws[a][0]
            frame0_fi = ws_k[WS_FI][0].copy(); frame0_ff = ws_k[WS_FF][0].copy()
            meta_k = st_k[ST_META]; iters0 = int(meta_k[1]); last_mem = time.perf_counter()
            try:
                while True:
                    try:
                        pos = tasks.get_nowait()
                    except queue.Empty:
                        break
                    chunk = 2000
                    while True:
                        ws_k[WS_FI][0] = frame0_fi; ws_k[WS_FF][0] = frame0_ff
                        meta_k[2] = 0; meta_k[3] = meta_k[1] + chunk
                        t0 = time.perf_counter()
                        solve_iter(st_k, self.dat, self.pf, self.pi, ws_k, np.int64(root), float(budget), np.int64(pos), shared,
                                   sh, np.int64(tid))
                        dt = time.perf_counter() - t0
                        if meta_k[2] == 0:
                            break
                        if meta_k[2] == 2:
                            st_k = self._new_store(st_k[ST_KEYS].shape[0] * 2, st_k); meta_k = st_k[ST_META]
                            thread_bytes[tid] = _store_bytes(st_k)
                            continue
                        if meta_k[2] == 3:
                            raise TimeLimitReached("depth")
                        now = time.perf_counter()
                        if now > deadline or state["failure"]:
                            raise TimeLimitReached("time")
                        if self.memory_limit > 0 and now - last_mem > 0.5:
                            last_mem = now
                            if self._mem_bytes() > self.memory_limit:
                                raise TimeLimitReached("memory")
                        if dt > 0.0:
                            chunk = int(min(max(chunk * 0.05 / dt, 500), 200000))
                    kind = int(ws_k[WS_FF][0, FF_OUT_KIND]); value = float(ws_k[WS_FF][0, FF_OUT_VALUE])
                    ln = int(ws_k[WS_FI][0, FI_LN]); rn = int(ws_k[WS_FI][0, FI_RN])
                    lbmax = 0.0
                    if ln >= 0:
                        lbmax = max(lbmax, float(st_k[ST_LB][ln]))
                    if rn >= 0:
                        lbmax = max(lbmax, float(st_k[ST_LB][rn]))
                    f = -1
                    if kind == 1:
                        with lock:
                            if value < shared[0]:
                                shared[0] = value
                        f = int(ws_k[WS_IO][0, 0, ws_k[WS_IO][0, 1, pos]])
                    with lock:
                        results.append((pos, kind, value, lbmax, (tid, f, ln, rn)))
            except TimeLimitReached as exc:
                with lock:
                    state["failure"] = state["failure"] or str(exc)
            except Exception as exc:
                with lock:
                    state["failure"] = state["failure"] or f"worker error: {exc!r}"
            with lock:
                state["iters"] += int(meta_k[1]) - iters0
                state["adopted"] = state.get("adopted", 0) + int(meta_k[6])
                stores[tid] = st_k          # final store: trees adopted by other threads live here
            release_workspace(ws_k, m, K, W)

        threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        meta[1] += state["iters"]
        self.peak_thread_bytes = int(thread_bytes.sum())
        release_shared_table(sh, n_threads, self.data.W)
        self.n_adopted = state.get("adopted", 0)
        best_tree = None; best_ref = None
        for pos, kind, value, lbmax, ref in results:
            if kind == 1:
                if ref[1] >= 0 and value < best - EPS:
                    best = value; best_split = ref[1]; best_ref = ref
            else:
                if value < min_pruned:
                    min_pruned = value
            if lbmax > child_lb_max:
                child_lb_max = lbmax
        if best < st[ST_UB][root]:
            st[ST_UB][root] = best
        if best_ref is not None:
            # extraction after the join: every store is final, so subtrees solved by other
            # threads are followed into their owner's store
            tid, f, ln, rn = best_ref
            self.stores = stores
            ext = _StoreView(self, stores[tid], stores)
            best_tree = {"feature": f, "true": ext.extract(ln), "false": ext.extract(rn)}
        if best_tree is not None:
            self.parallel_tree = best_tree
            st[ST_SPLIT][root] = best_split
            st[ST_PEND][root, 0] = 0
        if state["failure"] or len(results) < n_keep:
            raise TimeLimitReached(state["failure"] or "time")
        if best <= budget + EPS:
            st[ST_LB][root] = best
            st[ST_SOLVED][root] = 1
        else:
            v = max(min(best, min_pruned), child_lb_max)
            if v > st[ST_LB][root]:
                st[ST_LB][root] = v

    # ------------------------------------------------------------ extraction
    def _node_id(self, kw):
        nid = store_find(self.st, kw)
        if nid < 0:
            nid = make_node(self.st, self.dat, self.pf, self.pi, kw)
            if nid < 0:
                self._alloc(self.st[ST_KEYS].shape[0] * 2, self.st)
                nid = make_node(self.st, self.dat, self.pf, self.pi, kw)
        return nid

    def _apply_pending(self, nid):
        st = self.st
        kind = int(st[ST_PEND][nid, 0])
        st[ST_PEND][nid, 0] = 0
        f = int(st[ST_SPLIT][nid])
        if kind == 0 or f < 0:
            return
        W = self.data.W
        akey = np.empty(W, dtype=np.uint64); bkey = np.empty(W, dtype=np.uint64)
        child_key(st, self.dat, nid, f, True, akey)
        child_key(st, self.dat, nid, f, False, bkey)
        if kind == 2:
            cells = [(bkey if int(st[ST_PEND][nid, 1]) == 0 else akey, int(st[ST_PEND][nid, 2]))]
        else:
            cells = [(akey, int(st[ST_PEND][nid, 1])), (bkey, int(st[ST_PEND][nid, 2]))]
        for cell, t in cells:
            # _node_id may grow the store and replace self.st: re-read it after every call
            gn = self._node_id(cell)
            st = self.st
            c1 = np.empty(W, dtype=np.uint64); c2 = np.empty(W, dtype=np.uint64)
            child_key(st, self.dat, gn, t, True, c1)
            child_key(st, self.dat, gn, t, False, c2)
            g1 = self._node_id(c1); g2 = self._node_id(c2)
            st = self.st
            v = float(st[ST_LEAF][g1] + st[ST_LEAF][g2])
            if v < st[ST_UB][gn] - EPS:
                st[ST_UB][gn] = v
                st[ST_SPLIT][gn] = t
                st[ST_PEND][gn, 0] = 0

    def extract(self, nid):
        st = self.st
        f = int(st[ST_SPLIT][nid])
        if f <= SPLIT_EXTERN:
            # solved by another thread: its tree lives in that thread's (final) store
            owner = SPLIT_EXTERN - f
            view = _StoreView(getattr(self, "opt", self), self.stores[owner], self.stores)
            nid2 = store_find(view.st, st[ST_KEYS][nid])
            if nid2 < 0 or view.st[ST_SOLVED][nid2] != 1:
                raise RuntimeError("shared solution missing from its owner's store")
            return view.extract(nid2)
        if st[ST_PEND][nid, 0] != 0:
            self._apply_pending(nid)
            f = int(st[ST_SPLIT][nid])
        if f < 0:
            return {"prediction": int(st[ST_PRED][nid]), "key": words_to_int(st[ST_KEYS][nid]),
                    "count": int(st[ST_COUNT][nid])}
        W = self.data.W
        a = np.empty(W, dtype=np.uint64); b = np.empty(W, dtype=np.uint64)
        child_key(st, self.dat, nid, f, True, a)
        child_key(st, self.dat, nid, f, False, b)
        return {"feature": f, "true": self.extract(self._node_id(a)), "false": self.extract(self._node_id(b))}

    @property
    def memo(self):
        return range(int(self.st[ST_META][0]))


@njit(cache=NUMBA_CACHE, nogil=True)
def _rebuild_index(nkeys, hidx, n):
    hmask = np.int64(hidx.shape[0] - 1)
    for nid in range(n):
        slot = _slot_of(nkeys[nid], hmask)
        while hidx[slot] >= 0:
            slot = (slot + 1) & hmask
        hidx[slot] = nid


def words_to_int(kw) -> int:
    return int.from_bytes(np.ascontiguousarray(kw).tobytes(), "little")


class _StoreView:
    """Extraction from a given store (a thread's private copy); ``stores`` maps a thread id to
    its final store for subtrees solved by other threads."""

    def __init__(self, opt, st, stores=None):
        self.opt = opt
        self.st = st
        self.stores = stores
        self.data = opt.data
        self.dat = opt.dat
        self.pf = opt.pf
        self.pi = opt.pi

    def _alloc(self, cap, old):
        self.st = self.opt._new_store(cap, old)

    _node_id = CompiledOptimizer._node_id
    _apply_pending = CompiledOptimizer._apply_pending
    extract = CompiledOptimizer.extract


def _compiled_worker(opt, shared, lock, tasks, results, memory_limit, deadline):
    """Worker: solve one root split at a time with the compiled search (single-candidate
    mode), report (position, kind, value, children's max lb, tree)."""
    st = opt.st; ws = opt.ws; meta = st[ST_META]
    FI = ws[WS_FI]; FF = ws[WS_FF]
    root = int(FI[0, FI_NODE]); budget = float(FF[0, FF_BUDGET])
    frame0_fi = FI[0].copy(); frame0_ff = FF[0].copy()
    iters0 = int(meta[1]); failure = ""; last_mem = time.perf_counter()
    try:
        while True:
            pos = tasks.get()
            if pos < 0:
                break
            chunk = 2000
            while True:
                FI[0] = frame0_fi; FF[0] = frame0_ff       # restart the candidate from scratch (memo kept)
                meta[2] = 0; meta[3] = meta[1] + chunk
                t0 = time.perf_counter()
                solve_iter(st, opt.dat, opt.pf, opt.pi, ws, np.int64(root), budget, np.int64(pos), shared, opt._no_table, np.int64(0))
                dt = time.perf_counter() - t0
                if meta[2] == 0:
                    break
                if meta[2] == 2:
                    opt._alloc(st[ST_KEYS].shape[0] * 2, st); st = opt.st; meta = st[ST_META]
                    continue
                if meta[2] == 3:
                    raise TimeLimitReached("depth")
                now = time.perf_counter()
                if now > deadline:
                    raise TimeLimitReached("time")
                if memory_limit > 0 and now - last_mem > 0.5:
                    last_mem = now
                    if _rss_bytes() > memory_limit:
                        raise TimeLimitReached("memory")
                if dt > 0.0:
                    chunk = int(min(max(chunk * 0.05 / dt, 500), 200000))
            kind = int(FF[0, FF_OUT_KIND]); value = float(FF[0, FF_OUT_VALUE])
            ln = int(FI[0, FI_LN]); rn = int(FI[0, FI_RN])
            lbmax = 0.0
            if ln >= 0:
                lbmax = max(lbmax, float(st[ST_LB][ln]))
            if rn >= 0:
                lbmax = max(lbmax, float(st[ST_LB][rn]))
            tree = None
            if kind == 1:
                with lock:
                    if value < shared[0]:
                        shared[0] = value
                f = int(ws[WS_IO][0, 0, ws[WS_IO][0, 1, pos]])
                tree = {"feature": f, "true": opt.extract(ln), "false": opt.extract(rn)}
            results.put(("split", pos, kind, value, lbmax, tree))
    except TimeLimitReached as exc:
        failure = str(exc)
    except Exception as exc:
        failure = f"worker error: {exc!r}"
    results.put(("done", int(meta[1]) - iters0, failure))
    results.close()
    results.join_thread()


_COMPILED = [False]


def _compile_search():
    """Run the compiled search once on a tiny problem so its compilation (or cache load)
    happens before any timed fit."""
    if _COMPILED[0]:
        return
    _COMPILED[0] = True
    Xb = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=bool)
    y = np.array([0, 1, 1, 0])
    data = BitDataset(Xb, y, 2)
    opt = CompiledOptimizer(data, 0.1)
    st, dat, pf, pi = opt.st, opt.dat, opt.pf, opt.pi
    # the callees must be compiled by direct calls before the recursive search is
    # compiled, otherwise numba fails to link them into it ("unresolved symbol")
    kw = int_to_words(data.full, data.W).copy()
    root = make_node(st, dat, pf, pi, kw)
    W = data.W; K = data.K; mf = data.m
    buf = np.empty(W, dtype=np.uint64); buf2 = np.empty(W, dtype=np.uint64)
    child_key(st, dat, root, 0, True, buf)
    feats = np.arange(mf, dtype=np.int64)
    io = np.empty((7, mf), dtype=np.int64); fo = np.empty((14, mf)); bo = np.empty((2, mf), dtype=np.bool_)
    L = np.empty((mf, K)); dist = np.empty(K + 1)
    nv, n_cand, i0, mr, ran, i_d2, bd2, lbg, M, Fc = expand_kernel(dat[DT_F], feats, dat[DT_GROUP], st[ST_KEYS][root],
                                                                 dat[DT_MASKS], dat[DT_WEIGHTS], dat[DT_COSTS], dat[DT_DIFF],
                                                                 0.1, 1.0, False, io, fo, bo, L, dist)
    leaf_stats_words(kw, dat[DT_MASKS], dat[DT_WEIGHTS], dat[DT_COSTS], dat[DT_DIFF], K)
    child_node(st, dat, pf, pi, root, 0, True, 0.5, 0.1, False, 0, buf)
    _count_words(buf)
    _propagate(np.zeros(2), np.zeros(2, dtype=np.int64), np.zeros(2), 0, 1.0, 0.5)
    column_dp_chain(st, dat, pf, pi, root, feats[:1], L[:1], dist[:K], False, 1e300, buf, buf2)
    val3 = np.empty((nv, 2, 6)); arg3 = np.empty((nv, 2, 6), dtype=np.int64)
    if nv >= 1:
        depth3_triples(Fc, np.ascontiguousarray(io[0, :nv]), io[6, :nv], M[:K], dat[DT_COSTS], pf[PF_UW], 0.1, dist[:K], L[:nv], val3, arg3)
        f3 = np.empty((4, nv)); k3 = np.empty((2, nv), dtype=np.int64)
        depth3_bounds(fo[0, :nv], fo[7, :nv], fo[9, :nv], fo[2, :nv], fo[8, :nv], fo[10, :nv], val3, 0.1,
                      f3[0], f3[1], f3[2], f3[3], k3[0], k3[1])
        set_child_tree(st, root, np.ascontiguousarray(io[0, :nv]), 0, 1, 0, arg3, 0, st[ST_UB][root])
        refilter_candidates(io[1, :nv], nv, fo[5, :nv], fo[13, :nv], 1.0)
    max_pair(fo[1, :nv], fo[3, :nv])
    st[ST_SPLIT][root] = -1
    st[ST_PEND][root, 0] = 0
    ws = opt.ws
    ws[WS_FI][0, FI_NODE] = root; ws[WS_FF][0, FF_BUDGET] = float(st[ST_UB][root]); ws[WS_FI][0, FI_PHASE] = 0
    _expand_frame(st, dat, pf, pi, ws, 0)
    _finish_frame(st, ws, 0)
    st[ST_SPLIT][root] = -1
    st[ST_PEND][root, 0] = 0
    st[ST_SOLVED][root] = 0
    opt.run()
    opt.extract(opt.root)
    _flush_frames(st, ws, 0)
    # store growth (workers may grow their store: compile the rebuild here, in the parent)
    opt._alloc(st[ST_KEYS].shape[0] * 2, st)
    st = opt.st
    # the single-candidate mode (workers) with a shared bound array
    st[ST_SOLVED][root] = 0
    ws[WS_FI][0, FI_NODE] = root; ws[WS_FF][0, FF_BUDGET] = float(st[ST_UB][root]); ws[WS_FI][0, FI_PHASE] = 0
    if not _expand_frame(st, dat, pf, pi, ws, 0):
        solve_iter(st, dat, pf, pi, ws, np.int64(root), float(st[ST_UB][root]), np.int64(0), np.array([1e300]),
                   make_shared_table(1, 8, data.W, 1), np.int64(0))
    sh = make_shared_table(1, 8, data.W, 1)
    sh_publish(sh, np.int64(0), st[ST_KEYS][root], int(st[ST_COUNT][root]), 0.0, np.nan)
    sh_lookup(sh, st[ST_KEYS][root], int(st[ST_COUNT][root]))
    _sh_publish_node(sh, np.int64(0), st, root, 0.0, 0.1)
    st[ST_SOLVED][root] = 0
