import numpy as np
from scipy.stats import rankdata


def tauAP_b(x, y, decreasing=True):
    if decreasing:
        return tauAP_b(-x, -y, decreasing=False)
    else:
        return (_tauAP_b_ties(x, y) + _tauAP_b_ties(y, x)) / 2


def _tauAP_b_ties(x, y):
    n = len(x)
    rx = rankdata(x)
    ry = rankdata(y, method="ordinal")
    p = rankdata(y, method="min")
    c_all = 0
    not_top = np.argwhere(p != 1)
    for i in not_top:
        c_above = 0
        for j in np.argwhere(p < p[i]):
            sx = np.sign(rx[i] - rx[j])
            sy = np.sign(ry[i] - ry[j])
            if sx == sy:
                c_above = c_above + 1
        c_all = c_all + c_above/(p[i] - 1)
    return 2 / len(not_top) * c_all - 1


def rbo(s, t, p, k=None, side="top", uneven_lengths=True):
    assert side in ["top", "bottom"]
    if k is None:
        k = int(np.floor(max(len(s), len(t)) / 2))
    if side == "top":
        ids = {"s": _select_ids(s, "top"),
               "t": _select_ids(t, "top")}
    elif side == "bottom":
        ids = {"s": _select_ids(s, "bottom"),
               "t": _select_ids(t, "bottom")}
    return min(1, _rbo_ext(ids["s"], ids["t"], p, k, uneven_lengths=uneven_lengths))


def _select_ids(x, side="top"):
    assert side in ["top", "bottom"]
    if side == "top":
        return np.argsort(-x)
    elif side == "bottom":
        return np.argsort(x)


def _rbo_ext(x, y, p, k, uneven_lengths=True):
    if len(x) <= len(y):
        S = x
        L = y
    else:
        S = y
        L = x
    l = min(k, len(L))
    s = min(k, len(S))
    if uneven_lengths:
        Xd = [len(np.intersect1d(S[:(i+1)], L[:(i+1)])) for i in range(l)]
        if l > s:
            sl_range = np.arange(s+1, l+1)
        else:
            sl_range = np.arange(l, s+2)
        result = ((1 - p) / p) * \
                 ((sum(Xd[:l] / np.arange(1, l+1) * p**np.arange(1, l+1))) +
                  (sum(Xd[s-1] * (sl_range - s) / (s * sl_range) * p**sl_range))) + \
                 ((Xd[l-1] - Xd[s-1]) / l + (Xd[s-1] / s)) * p**l
    else:
        k = min(s, k)
        Xd = [len(np.intersect1d(x[:(i+1)], y[:(i+1)])) for i in range(k)]
        Xk = Xd[k-1]
        result = (Xk / k) * p**k + (((1 - p) / p) * sum((Xd / np.arange(1, k+1)) * p**np.arange(1, k+1)))
    return result