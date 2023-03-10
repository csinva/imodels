import numpy as np
from scipy.stats import rankdata


def tauAP_b(x, y, decreasing=True):
    """
    Weighted kendall tau correlation metric, which handles ties.
    Proposed in "The Treatment of Ties in AP Correlation" by
    Urbano and Marrero (2017). This is the python implementation
    of ircor::tauAP_b from R.

    Parameters
    ----------
    x: array-like of shape (n,)
        Numeric vector.
    y: array-like of shape (n,)
        Numeric vector of same length as x.
    decreasing: bool
        Should the sort order be increasing or decreasing (default)?

    Returns
    -------
    Scalar value between -1 and 1, quantifying how much the
    rankings of x and y agree with each other. A higher
    values indicates greater similarity.

    """
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
    """
    Rank-based overlap (RBO) metric.
    Proposed in "A Similarity Measure for Indefinite Rankings" by
    Webber et al. (2010). This is the python implementation
    of gespeR::rbo from R.

    Parameters
    ----------
    s: array-like of shape (n,)
        Numeric vector.
    t: array-like of shape (n,)
        Numeric vector of same length as s.
    p: float between 0 and 1
        Weighting parameter in [0, 1]. High p implies strong emphasis
        on the top-ranked elements (i.e, the larger elements).
    k: None or int
        Evaluation depth for extrapolation
    side: string in {"top", "bottom"}
        Evaluate similarity between the top or the bottom of the
        ranked lists.
    uneven_lengths: bool
        Indicator if lists have uneven lengths.

    Returns
    -------
    Scalar value between 0 and 1, quantifying how much the
    rankings of x and y agree with each other. A higher
    values indicates greater similarity.

    """
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