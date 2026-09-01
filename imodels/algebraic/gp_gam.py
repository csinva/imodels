"""Additive Gaussian-process GAM fit from binned sufficient statistics.

A generalized additive model with pairwise interactions (a GA2M) in which every
component is a Gaussian process over the quantile bins of its feature(s).

The useful consequence of binning is that the *exact* GP marginal likelihood
stops depending on the sample size. Writing ``Z`` for the indicator matrix that
records which bin each row falls in, the likelihood touches the data only
through the bin co-occurrence counts ``C = Z.T @ Z``, the bin sums
``b = Z.T @ y`` and ``y.T @ y``. One pass over the data builds those; every
optimizer step afterwards costs ``O(P^3)`` in the total number of bins ``P``,
regardless of how many rows there were.

All kernel amplitudes and the noise level are chosen by maximizing that
likelihood, which makes the model free of the usual tuning knobs: smoothness is
inferred per feature from a two-kernel mixture, irrelevant features are pruned
because their amplitudes go to zero (automatic relevance determination), and the
resolution of each interaction grid is picked by comparing marginal likelihoods.
Nothing is chosen by cross-validation, so the fit is deterministic -- no splits,
no seeds, no bagging.

Reference implementation: https://github.com/csinva/imodels
"""

from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from imodels.util.arguments import check_predict_X, set_feature_names_in


class GPGamRegressor(RegressorMixin, BaseEstimator):
    """Additive Gaussian-process GAM with pairwise interactions.

    The fitted model is ``y = sum_j f_j(x_j) + sum_(a,b) f_ab(x_a, x_b)``, where
    every term is a lookup table over quantile bins, so the model can be read off
    directly (see :meth:`shape_function`).

    Parameters
    ----------
    schedule : bool, default=True
        Scale model capacity with the sample size. Small problems get a lean
        64-bin model with few interactions; larger ones get 256 bins and more.
        Set ``False`` to control capacity yourself with the parameters below.
    n_bins : int, default=64
        Maximum quantile bins per feature.
    p_budget : int or None, default=None
        Total-bin budget. The per-feature bin count is the budget divided by the
        number of features (capped by ``n_bins``), which keeps the fit tractable
        on wide data.
    scales : tuple, default=(0.05,)
        Lengthscales for the Matern-1/2 kernels on each feature's bin grid, in
        units of the grid width. These produce rough, locally adaptive shapes.
    rbf_scales : tuple, default=(0.25,)
        Lengthscales for the squared-exponential kernels, which produce smooth
        shapes. The marginal likelihood decides the mixture per feature.
    n_pairs : int, default=6
        Maximum number of pairwise interaction terms.
    pair_bins : int, default=12
        Bins per axis for interaction grids.
    pair_res : tuple or None, default=None
        Candidate interaction-grid resolutions. Each block of interactions is fit
        at every candidate and the marginal likelihood keeps the best.
    pair_scales : tuple, default=(0.05, 0.3)
        Lengthscales for the product kernels used by interaction terms.
    screen_bins : int, default=8
        Grid resolution used when screening candidate interactions.
    pair_shrink : float, default=8.0
        Shrinkage applied to sparsely populated cells during screening.
    n_steps : int, default=200
        Gradient steps on the marginal likelihood. The step count is a real part
        of the model: stopping here regularizes the fit, and running the
        likelihood to convergence overfits.
    lr : float, default=0.05
        Adam step size for the log-amplitudes and log-noise.
    log_target : {'auto', True, False}, default='auto'
        Fit on ``log(y)`` when the target is positive and taking logs reduces its
        skew substantially. Predictions are returned on the original scale.
    n_features_in_ : int
        Set after fitting.

    Examples
    --------
    >>> from imodels import GPGamRegressor
    >>> from sklearn.datasets import make_friedman1
    >>> X, y = make_friedman1(n_samples=500, random_state=0)
    >>> model = GPGamRegressor().fit(X, y)
    >>> preds = model.predict(X)
    >>> grid, values = model.shape_function(0)   # feature 0's fitted curve
    """

    def __init__(
        self,
        schedule=True,
        n_bins=64,
        p_budget=None,
        scales=(0.05,),
        rbf_scales=(0.25,),
        n_pairs=6,
        pair_bins=12,
        pair_res=None,
        pair_scales=(0.05, 0.3),
        screen_bins=8,
        pair_shrink=8.0,
        n_steps=200,
        lr=0.05,
        noise_init=0.3,
        noise_floor=1e-4,
        jitter=1e-6,
        log_target="auto",
        cat_max_levels=32,
    ):
        self.schedule = schedule
        self.n_bins = n_bins
        self.p_budget = p_budget
        self.scales = scales
        self.rbf_scales = rbf_scales
        self.n_pairs = n_pairs
        self.pair_bins = pair_bins
        self.pair_res = pair_res
        self.pair_scales = pair_scales
        self.screen_bins = screen_bins
        self.pair_shrink = pair_shrink
        self.n_steps = n_steps
        self.lr = lr
        self.noise_init = noise_init
        self.noise_floor = noise_floor
        self.jitter = jitter
        self.log_target = log_target
        self.cat_max_levels = cat_max_levels

    # ------------------------------------------------------------------
    # capacity
    # ------------------------------------------------------------------
    def _p(self, name):
        """Effective capacity parameter; the size-derived schedule wins if on."""
        return self._sched.get(name, getattr(self, name))

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    def _feature_kernels(self, n_bins):
        """Covariance kernels for one feature's bin grid."""
        if n_bins <= 3:
            # a delta kernel already spans every function on so few points
            return [np.eye(n_bins)]
        grid = np.linspace(0.0, 1.0, n_bins)
        dist = np.abs(grid[:, None] - grid[None, :])
        mats = [np.exp(-dist / s) for s in self.scales]
        mats += [np.exp(-((dist / s) ** 2)) for s in self.rbf_scales]
        return mats

    def _pair_kernels(self, na, nb):
        """Product kernels over a 2-D interaction grid."""
        ga, gb = np.linspace(0, 1, na), np.linspace(0, 1, nb)
        da = np.abs(ga[:, None] - ga[None, :])
        db = np.abs(gb[:, None] - gb[None, :])
        out = [np.kron(np.exp(-da / s), np.exp(-db / s)) for s in self.pair_scales]
        out.append(np.kron(np.exp(-((da / 0.3) ** 2)), np.exp(-((db / 0.3) ** 2))))
        return out

    # ------------------------------------------------------------------
    # marginal likelihood
    # ------------------------------------------------------------------
    def _nll_and_grad(self, blocks, offsets, C, b, yy, n, log_amps, log_noise):
        """Negative log marginal likelihood and its gradient.

        Everything is expressed in the sufficient statistics, so the cost is
        independent of ``n``. Returns ``(nll, grad_log_amps, grad_log_noise,
        state)`` where ``state`` carries the factorization needed downstream, or
        ``None`` if the parameters produced a non-positive-definite matrix.
        """
        P = int(offsets[-1])
        sig2 = float(np.exp(log_noise))
        binv, logdet_a = [], 0.0
        for u, kernels in enumerate(blocks):
            amps = np.exp(log_amps[u])
            A = sum(a * K for a, K in zip(amps, kernels))
            A = A + self.jitter * np.eye(A.shape[0])
            try:
                cf = cho_factor(A, lower=True)
            except np.linalg.LinAlgError:
                return None
            logdet_a += 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
            binv.append(cho_solve(cf, np.eye(A.shape[0])))

        G = C / sig2
        for u, Bu in enumerate(binv):
            i0, i1 = offsets[u], offsets[u + 1]
            G[i0:i1, i0:i1] += Bu
        G[np.diag_indices_from(G)] += 1e-6
        try:
            cg = cho_factor(G, lower=True)
        except np.linalg.LinAlgError:
            return None
        logdet_g = 2.0 * float(np.sum(np.log(np.diag(cg[0]))))
        with np.errstate(over="ignore", invalid="ignore"):
            ginv = cho_solve(cg, np.eye(P))
            mu = ginv @ (b / sig2)
        if not (np.all(np.isfinite(ginv)) and np.all(np.isfinite(mu))):
            return None

        quad = (yy - float(b @ mu)) / sig2
        nll = 0.5 * (quad + n * float(log_noise) + logdet_a + logdet_g)
        if not np.isfinite(nll):
            return None

        # gradients: M = Z' Sigma^-1 Z and r = Z' Sigma^-1 y, both built from
        # the sufficient statistics, so this stays independent of the sample size
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            T = ginv @ C                       # P x P
            r = (b - C @ mu) / sig2
            grad_a = []
            for u, kernels in enumerate(blocks):
                i0, i1 = offsets[u], offsets[u + 1]
                Muu = (C[i0:i1, i0:i1] - C[i0:i1, :] @ T[:, i0:i1] / sig2) / sig2
                ru = r[i0:i1]
                amps = np.exp(log_amps[u])
                g = np.empty(len(kernels))
                for s_, K in enumerate(kernels):
                    g[s_] = 0.5 * (float(np.sum(K * Muu)) - float(ru @ K @ ru))
                grad_a.append(g * amps)        # chain rule for the log-amplitudes

            tr_sinv = (n - float(np.trace(T)) / sig2) / sig2
            alpha_sq = (yy - 2.0 * float(b @ mu) + float(mu @ C @ mu)) / sig2 ** 2
            grad_noise = 0.5 * (tr_sinv - alpha_sq) * sig2

        if not (np.isfinite(grad_noise) and all(np.all(np.isfinite(g)) for g in grad_a)):
            return None                        # ill-conditioned: back off on noise
        return nll, grad_a, float(grad_noise), (binv, ginv)

    def _fit_ml(self, blocks, offsets, C, b, yy, n):
        """Maximize the marginal likelihood with Adam on the log-parameters."""
        n_kernels = sum(len(k) for k in blocks)
        init = float(np.log(0.5 / max(n_kernels, 1)))
        log_amps = [np.full(len(k), init) for k in blocks]
        log_noise = float(np.log(self.noise_init))
        m_a = [np.zeros_like(v) for v in log_amps]
        v_a = [np.zeros_like(v) for v in log_amps]
        m_n = v_n = 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = (np.inf, None, None)
        t = 0
        for _ in range(self.n_steps):
            out = self._nll_and_grad(blocks, offsets, C, b, yy, n, log_amps, log_noise)
            if out is None:                # non-PD: back off toward more noise
                log_noise += 0.25
                continue
            nll, grad_a, grad_n, _ = out
            if nll < best[0]:
                best = (nll, [v.copy() for v in log_amps], log_noise)
            t += 1
            for u in range(len(log_amps)):
                m_a[u] = b1 * m_a[u] + (1 - b1) * grad_a[u]
                v_a[u] = b2 * v_a[u] + (1 - b2) * grad_a[u] ** 2
                mh = m_a[u] / (1 - b1 ** t)
                vh = v_a[u] / (1 - b2 ** t)
                log_amps[u] = np.clip(log_amps[u] - self.lr * mh / (np.sqrt(vh) + eps), -25, 12)
            m_n = b1 * m_n + (1 - b1) * grad_n
            v_n = b2 * v_n + (1 - b2) * grad_n ** 2
            log_noise = float(np.clip(
                log_noise - self.lr * (m_n / (1 - b1 ** t)) / (np.sqrt(v_n / (1 - b2 ** t)) + eps),
                -25, 12))

        nll_best, amps_best, noise_best = best
        if amps_best is None:
            amps_best, noise_best = log_amps, log_noise
        amps = [np.exp(v) for v in amps_best]
        sig2 = max(float(np.exp(noise_best)), self.noise_floor)

        # posterior mean of the bin values, solved once in full precision
        P = int(offsets[-1])
        G = C / sig2
        for u, kernels in enumerate(blocks):
            A = sum(a * K for a, K in zip(amps[u], kernels))
            A = A + self.jitter * np.eye(A.shape[0])
            try:
                Ai = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                Ai = np.linalg.pinv(A)
            i0, i1 = offsets[u], offsets[u + 1]
            G[i0:i1, i0:i1] += Ai
        for ridge in (0.0, 1e-4, 1e-3, 1e-2):
            try:
                cf = cho_factor(G + ridge * np.eye(P), lower=True)
                fhat = cho_solve(cf, b / sig2)
                if np.isfinite(fhat).all() and np.abs(fhat).max() < 1e6:
                    break
            except np.linalg.LinAlgError:
                continue
        else:
            fhat = np.linalg.lstsq(G + np.eye(P), b / sig2, rcond=None)[0]
        return fhat, amps, (nll_best if np.isfinite(nll_best) else np.inf)

    # ------------------------------------------------------------------
    @staticmethod
    def _suffstats(cols, sizes, target):
        """Bin co-occurrence counts and bin sums for a set of terms."""
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        P = int(offsets[-1])
        C = np.zeros((P, P))
        b = np.zeros(P)
        for u, cu in enumerate(cols):
            b[offsets[u]:offsets[u + 1]] = np.bincount(cu, weights=target, minlength=sizes[u])
            for v in range(u, len(cols)):
                m = np.bincount(cu * sizes[v] + cols[v],
                                minlength=sizes[u] * sizes[v]
                                ).reshape(sizes[u], sizes[v]).astype(float)
                C[offsets[u]:offsets[u + 1], offsets[v]:offsets[v + 1]] = m
                if v != u:
                    C[offsets[v]:offsets[v + 1], offsets[u]:offsets[u + 1]] = m.T
        return C, b, offsets

    def _bin_edges(self, x, n_bins):
        uniq = np.unique(x[np.isfinite(x)])
        if len(uniq) <= 1:
            return None
        if len(uniq) <= n_bins:
            return (uniq[:-1] + uniq[1:]) / 2.0
        return np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))

    # ------------------------------------------------------------------
    def fit(self, X, y, feature_names=None):
        """Fit the additive GP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        """
        X_original = X
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        else:
            set_feature_names_in(self, X_original)
        n, d = X.shape

        # capacity grows with the sample size
        if self.schedule:
            if n <= 1000:
                self._sched = dict(n_bins=64, p_budget=1500, pair_bins=12,
                                   n_pairs=min(2 * d, 12), pair_res=(12,))
            else:
                self._sched = dict(n_bins=256, p_budget=4200, pair_bins=28,
                                   n_pairs=min(3 * d, 48), pair_res=(28, 24, 16))
        else:
            self._sched = {}

        # 1. condition the target
        self.log_target_ = False
        if self.log_target in ("auto", True) and np.min(y) > 0:
            from scipy.stats import skew
            if self.log_target is True or abs(skew(np.log(y))) < abs(skew(y)) - 1.0:
                self.log_target_ = True
                y = np.log(y)
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        iqr = q3 - q1
        if iqr > 0:                            # winsorize only genuine outliers
            lo, hi = med - 8.0 * iqr, med + 8.0 * iqr
            if 0.0 < np.mean((y < lo) | (y > hi)) <= 0.01:
                y = np.clip(y, lo, hi)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) + 1e-12
        yn = (y - self.y_mean_) / self.y_std_
        yy = float(np.sum(yn ** 2))

        # 2. bin every feature under the budget
        budget = self._p("p_budget")
        n_bins = self._p("n_bins")
        if budget:
            n_bins = int(np.clip(budget // max(d, 1), 2, n_bins))
        self.edges_, self.grids_, self.cats_ = {}, {}, np.zeros(d, dtype=bool)
        bidx = np.zeros((n, d), dtype=np.int64)
        units, sizes = [], []
        for j in range(d):
            e = self._bin_edges(X[:, j], n_bins)
            if e is None:
                continue
            self.edges_[j] = e
            nb = len(e) + 1
            bidx[:, j] = np.searchsorted(e, X[:, j], side="right")
            self.grids_[j] = (np.concatenate([[e[0]], (e[:-1] + e[1:]) / 2.0, [e[-1]]])
                              if len(e) > 1 else np.array([e[0] - 0.5, e[0] + 0.5]))[:nb]
            u = np.unique(X[:, j])
            if len(u) <= self.cat_max_levels and np.allclose(u, np.round(u)):
                self.cats_[j] = True
            units.append(j)
            sizes.append(nb)
        if not units:
            raise ValueError("every feature is constant; nothing to fit")
        self.units_ = units

        # 3. sufficient statistics, then 4. fit the main effects
        cols = [bidx[:, j] for j in units]
        C, b, offsets = self._suffstats(cols, sizes, yn)
        blocks = [self._feature_kernels(s) for s in sizes]
        fhat, amps, _ = self._fit_ml(blocks, offsets, C, b, yy, n)
        self.main_offsets_ = offsets
        self.main_values_ = fhat
        self.pairs_ = []
        self.pair_values_ = []

        # 5. screen interactions, 6. fit them blockwise-jointly
        n_pairs = self._p("n_pairs")
        if n_pairs > 0 and len(units) >= 2:
            resid = yn.copy()
            for u, j in enumerate(units):
                resid -= fhat[offsets[u]:offsets[u + 1]][bidx[:, j]]
            selected = self._screen_pairs(X, units, resid, n_pairs, amps)
            if selected:
                fhat, self.pairs_, self.pair_values_ = self._fit_pairs(
                    X, bidx, units, sizes, blocks, C, b, yy, n, offsets,
                    fhat, selected, yn)
                self.main_values_ = fhat

        rng = float(np.max(y) - np.min(y))
        self.clip_ = (float(np.min(y)) - 0.05 * rng, float(np.max(y)) + 0.05 * rng)
        self.bias_ = 0.0
        pred = self.predict(X)
        pred_t = np.log(np.maximum(pred, 1e-300)) if self.log_target_ else pred
        self.bias_ = float(np.mean(y) - np.mean(pred_t))
        return self

    def _screen_pairs(self, X, units, resid, n_pairs, amps):
        """Rank candidate interactions by shrunken residual cell means (FAST)."""
        feats = units
        if len(units) * (len(units) - 1) // 2 > 5000:      # keep wide data tractable
            strength = {j: float(np.sum(amps[u])) for u, j in enumerate(units)}
            feats = sorted(sorted(units, key=lambda j: -strength[j])[:100])
        binned = {}
        for j in feats:
            e = self._bin_edges(X[:, j], self.screen_bins)
            if e is not None and len(e) >= 1:
                binned[j] = (np.searchsorted(e, X[:, j], side="right"), len(e) + 1)
        gains = []
        for a, b_ in combinations(sorted(binned), 2):
            ia, na = binned[a]
            ib, nb = binned[b_]
            cell = ia * nb + ib
            cnt = np.bincount(cell, minlength=na * nb).astype(float)
            tot = np.bincount(cell, weights=resid, minlength=na * nb)
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
            mean *= cnt / (cnt + self.pair_shrink)
            gains.append((float(np.sum(cnt * mean ** 2)), a, b_))
        gains.sort(reverse=True)
        return [(a, b_) for _, a, b_ in gains[:n_pairs]]

    def _fit_pairs(self, X, bidx, units, sizes, blocks, C, b, yy, n, offsets,
                   main_vals, selected, yn):
        """Fit interaction terms in blocks, alternating with the main effects.

        Interactions are fit in chunks so that terms in a chunk share shrinkage,
        and each chunk is fit at every candidate grid resolution with the
        marginal likelihood keeping the winner.
        """
        resolutions = sorted(set(self._p("pair_res") or (self._p("pair_bins"),)), reverse=True)
        chunk = max(1, 3600 // (max(resolutions) ** 2))
        chunks = [selected[i:i + chunk] for i in range(0, len(selected), chunk)]
        defs = {p: None for p in selected}
        cols = {p: None for p in selected}
        vals = {p: None for p in selected}

        def build(pairs, R):
            cc, ss, dd = [], [], []
            for (a, b_) in pairs:
                ea = self._bin_edges(X[:, a], R)
                eb = self._bin_edges(X[:, b_], R)
                if ea is None or eb is None:
                    continue
                na, nb = len(ea) + 1, len(eb) + 1
                cc.append(np.searchsorted(ea, X[:, a], side="right") * nb
                          + np.searchsorted(eb, X[:, b_], side="right"))
                ss.append(na * nb)
                dd.append(dict(i=a, j=b_, ei=ea, ej=eb, na=na, nb=nb))
            return cc, ss, dd

        for _ in range(2):
            for ch in chunks:
                # what this chunk must explain: the target minus the main
                # effects and minus every interaction outside the chunk
                target = yn.copy()
                for u, j in enumerate(units):
                    target -= main_vals[offsets[u]:offsets[u + 1]][bidx[:, j]]
                for p in selected:
                    if p not in ch and defs[p] is not None:
                        target -= vals[p][cols[p]]
                best = None
                for R in resolutions:
                    cc, ss, dd = build(ch, R)
                    if not cc:
                        continue
                    Cc, bc, offc = self._suffstats(cc, ss, target)
                    kern = [self._pair_kernels(t["na"], t["nb"]) for t in dd]
                    fc, _, nll = self._fit_ml(kern, offc, Cc, bc,
                                              float(np.sum(target ** 2)), n)
                    if best is None or nll < best[0]:
                        best = (nll, fc, offc, cc, dd)
                if best is None:
                    continue
                _, fc, offc, cc, dd = best
                for i, t in enumerate(dd):
                    p = (t["i"], t["j"])
                    vals[p] = fc[offc[i]:offc[i + 1]]
                    cols[p] = cc[i]
                    defs[p] = t
            # refit the main effects against the fitted interactions
            adj = yn.copy()
            for p in selected:
                if defs[p] is not None:
                    adj -= vals[p][cols[p]]
            bm = np.concatenate([np.bincount(bidx[:, j], weights=adj, minlength=sizes[u])
                                 for u, j in enumerate(units)])
            main_vals, _, _ = self._fit_ml(blocks, offsets, C, bm,
                                           float(np.sum(adj ** 2)), n)
        keep = [p for p in selected if defs[p] is not None]
        return main_vals, [defs[p] for p in keep], [vals[p] for p in keep]

    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict on new data."""
        check_is_fitted(self, "main_values_")
        X = check_predict_X(self, X)
        X = check_array(X, accept_sparse=False)
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        offs = self.main_offsets_
        for u, j in enumerate(self.units_):
            f = self.main_values_[offs[u]:offs[u + 1]]
            if self.cats_[j] or len(f) < 3:
                out += f[np.searchsorted(self.edges_[j], X[:, j], side="right")]
            else:
                out += np.interp(X[:, j], self.grids_[j], f)
        for t, v in zip(self.pairs_, self.pair_values_):
            ia = np.searchsorted(t["ei"], X[:, t["i"]], side="right")
            ib = np.searchsorted(t["ej"], X[:, t["j"]], side="right")
            out += v[ia * t["nb"] + ib]
        out = self.y_mean_ + self.y_std_ * out + self.bias_
        out = np.clip(out, self.clip_[0], self.clip_[1])
        return np.exp(out) if self.log_target_ else out

    # ------------------------------------------------------------------
    def shape_function(self, feature):
        """Return ``(grid, values)``: the fitted curve for one feature.

        The values are on the (standardized, possibly log) fitting scale, which
        is the scale on which the model is additive.
        """
        check_is_fitted(self, "main_values_")
        j = int(feature)
        if j not in self.edges_:
            raise ValueError(f"feature {j} was constant and carries no shape function")
        u = self.units_.index(j)
        offs = self.main_offsets_
        return self.grids_[j].copy(), self.main_values_[offs[u]:offs[u + 1]].copy() * self.y_std_

    def interaction_terms(self):
        """List the fitted pairwise interactions as ``(feature_a, feature_b)``."""
        check_is_fitted(self, "main_values_")
        return [(t["i"], t["j"]) for t in self.pairs_]
