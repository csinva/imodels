"""A GAM whose shape functions are Gaussian processes over binned features.

The model is a GA2M. It sums one function of each feature plus functions of a
few feature pairs, and gives every one of those functions a Gaussian process
prior over the quantile bins of its feature.

Binning is what makes the exact marginal likelihood cheap to compute. Let ``Z``
be the indicator matrix recording which bin each row falls into. The likelihood
depends on the data only through three quantities: the bin co-occurrence counts
``C = Z.T @ Z``, the bin sums ``b = Z.T @ y``, and ``y.T @ y``. One pass over
the data computes all three. Every optimizer step after that costs ``O(P^3)``,
where ``P`` is the total number of bins, no matter how many rows the data has.

Maximizing that likelihood sets every kernel amplitude and the noise level. It
also settles the choices a GAM usually asks the user to make. How smooth each
shape function should be follows from the mixture of two kernels. Features that
explain nothing get amplitudes near zero and drop out of the model. The grid
resolution for each interaction is chosen by comparing likelihoods. None of this
uses cross-validation, so fitting is deterministic: no splits, no seeds, no
bagging, and two fits on the same data give the same model.

Three choices that a GAM usually leaves to its user are also made by that
likelihood here. One Matern and one squared-exponential lengthscale are learned
and shared by every feature, with a weak prior around their defaults. Each
kernel's log-amplitudes are shrunk toward their centre across features, a
hierarchical prior that keeps a feature from being pruned on one split and kept
on another. And above 1000 rows, interactions beyond the first 48 are backfit on
the joint model's residual, up to five per feature, each as an exact 2-D GP with
the grid resolution chosen by marginal likelihood; the first 48 are fit jointly.

Reference implementation: https://github.com/csinva/imodels
"""

import inspect
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from imodels.util.arguments import check_predict_X, set_feature_names_in
from imodels.util.progress import progress_iter


class GPGamRegressor(RegressorMixin, BaseEstimator):
    """A GAM with pairwise interactions, fit as a Gaussian process.

    The fitted model is ``y = sum_j f_j(x_j) + sum_(a,b) f_ab(x_a, x_b)``. Every
    term is a lookup table over quantile bins, so you can read the model itself
    rather than explain it after the fact (see :meth:`shape_function`).

    Parameters
    ----------
    schedule : bool, default=True
        Set model capacity from the sample size. Data with at most 1000 rows gets
        96 bins per feature and up to 16 interactions, scaled by n. Larger data
        gets 256 bins and up to five interactions per feature, the first 48 fit
        jointly and the rest backfit. Pass ``False`` to set capacity yourself
        with the parameters below.
    n_bins : int, default=64
        The most quantile bins to give one feature.
    p_budget : int or None, default=None
        Budget for the total number of bins. Each feature gets the budget divided
        by the number of features, capped at ``n_bins``. This keeps the fit
        tractable when the data has many columns.
    scales : tuple, default=(0.05,)
        Lengthscales for the Matern 1/2 kernels on each feature's bin grid, as a
        fraction of the grid width. These kernels produce rough shapes that can
        turn sharply.
    rbf_scales : tuple, default=(0.25,)
        Lengthscales for the squared exponential kernels, which produce smooth
        shapes. The marginal likelihood decides how much of each kernel to use,
        one feature at a time.
    n_pairs : int, default=6
        The most interaction terms to include.
    pair_bins : int, default=12
        Bins along each axis of an interaction grid.
    pair_res : tuple or None, default=None
        Candidate resolutions for interaction grids. Each block of interactions is
        fit at every candidate, and the marginal likelihood keeps the best one.
    pair_scales : tuple, default=(0.05, 0.3)
        Lengthscales for the product kernels that interaction terms use.
    screen_bins : int, default=8
        Grid resolution used to screen candidate interactions.
    pair_shrink : float, default=8.0
        Shrinkage applied to cells holding few points while screening.
    n_steps : int, default=200
        Gradient steps taken on the marginal likelihood. This count is part of the
        model, not just a budget: stopping here regularizes the fit, and running
        the likelihood to convergence overfits.
    lr : float, default=0.05
        Adam step size for the log amplitudes and the log noise level.
    log_target : {'auto', True, False}, default='auto'
        Fit on ``log(y)`` when ``y`` is positive and taking logs makes it much
        less skewed. Predictions come back on the original scale.
    tau : float, default=1.0
        Width of the hierarchical prior that shrinks each kernel's log-amplitudes
        toward their centre across features. ``0`` turns it off.
    learn_scales : bool, default=True
        Learn one Matern and one squared-exponential lengthscale, shared by all
        features, by marginal likelihood. ``scales`` and ``rbf_scales`` then
        give the starting values and the centre of the prior.
    scale_prior : float, default=0.5
        Standard deviation, in log space, of the prior on the learned
        lengthscales around their starting values. ``0`` turns it off.
    sweeps : int, default=3
        Backfitting sweeps for interactions beyond the first 48 (above 1000
        rows), each surface refit on the residual of all the others.
    n_features_in_ : int
        Set after fitting.

    Examples
    --------
    >>> from imodels import GPGamRegressor
    >>> from sklearn.datasets import make_friedman1
    >>> X, y = make_friedman1(n_samples=500, random_state=0)
    >>> model = GPGamRegressor().fit(X, y)
    >>> preds = model.predict(X)
    >>> grid, values = model.shape_function(0)   # the curve fit for feature 0
    """

    def __init__(
        self,
        schedule=True,
        n_bins=64,
        p_budget=None,
        scales=(0.05,
        ),
        rbf_scales=(0.25,
        ),
        n_pairs=6,
        pair_bins=12,
        pair_res=None,
        pair_scales=(0.05,
        0.3),
        screen_bins=8,
        pair_shrink=8.0,
        n_steps=200,
        lr=0.05,
        noise_init=0.3,
        noise_floor=1e-4,
        jitter=1e-6,
        log_target="auto",
        cat_max_levels=32,
        tau=1.0,
        learn_scales=True,
        scale_prior=0.5,
        sweeps=3,
        verbose=0,
    ):
        self.schedule = schedule
        self.n_bins = n_bins
        self.p_budget = p_budget
        self.scales = scales
        self.rbf_scales = rbf_scales
        self.tau = tau
        self.learn_scales = learn_scales
        self.scale_prior = scale_prior
        self.sweeps = sweeps
        self.verbose = verbose
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
        """The capacity value to use. The schedule wins when it is turned on."""
        return self._sched.get(name, getattr(self, name))

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    @staticmethod
    def _feature_dist(n_bins):
        """Pairwise distances of a feature's bin grid, or None if it has <= 3 bins."""
        if n_bins <= 3:
            return None
        grid = np.linspace(0.0, 1.0, n_bins)
        return np.abs(grid[:, None] - grid[None, :])

    def _feature_kernels(self, n_bins, scales=None, rbf_scales=None):
        """Covariance kernels for one feature's bin grid."""
        if n_bins <= 3:
            # on so few points a delta kernel already spans every function
            return [np.eye(n_bins)]
        dist = self._feature_dist(n_bins)
        scales = self.scales if scales is None else scales
        rbf_scales = self.rbf_scales if rbf_scales is None else rbf_scales
        mats = [np.exp(-dist / s) for s in scales]
        mats += [np.exp(-((dist / s) ** 2)) for s in rbf_scales]
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
    def _nll_and_grad(self, blocks, offsets, C, b, yy, n, log_amps, log_noise,
                      dists=None, log_ell=None):
        """Negative log marginal likelihood and its gradient.

        Everything here is written in terms of the sufficient statistics, so the
        cost does not depend on ``n``. Returns ``(nll, grad_log_amps,
        grad_log_noise, state)``, where ``state`` holds the factorization used
        later, or ``None`` if the parameters gave a matrix that is not positive
        definite.
        """
        P = int(offsets[-1])
        sig2 = float(np.exp(log_noise))
        learn = log_ell is not None and dists is not None
        if learn:
            # kernels rebuilt from the shared lengthscales; their derivatives with
            # respect to log-lengthscale are K * D / ell (Matern) and K * 2 (D / ell)^2 (RBF)
            ell = np.exp(log_ell).clip(0.005, 2.0)
            blocks = [ks if dists[u] is None else
                      [np.exp(-dists[u] / ell[0]), np.exp(-((dists[u] / ell[1]) ** 2))]
                      for u, ks in enumerate(blocks)]
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

        # gradients: M = Z' Sigma^-1 Z and r = Z' Sigma^-1 y. Both come from the
        # sufficient statistics, so this cost does not grow with the sample size.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            T = ginv @ C                       # P x P
            r = (b - C @ mu) / sig2
            grad_a = []
            grad_ell = np.zeros(2) if learn else None
            for u, kernels in enumerate(blocks):
                i0, i1 = offsets[u], offsets[u + 1]
                Muu = (C[i0:i1, i0:i1] - C[i0:i1, :] @ T[:, i0:i1] / sig2) / sig2
                ru = r[i0:i1]
                amps = np.exp(log_amps[u])
                g = np.empty(len(kernels))
                for s_, K in enumerate(kernels):
                    g[s_] = 0.5 * (float(np.sum(K * Muu)) - float(ru @ K @ ru))
                grad_a.append(g * amps)        # chain rule, since we fit log amplitudes
                if learn and dists[u] is not None:
                    D = dists[u]
                    dK = [kernels[0] * (D / ell[0]), kernels[1] * (2.0 * (D / ell[1]) ** 2)]
                    for s_ in range(2):
                        dA = amps[s_] * dK[s_]
                        grad_ell[s_] += 0.5 * (float(np.sum(dA * Muu)) - float(ru @ dA @ ru))

            tr_sinv = (n - float(np.trace(T)) / sig2) / sig2
            alpha_sq = (yy - 2.0 * float(b @ mu) + float(mu @ C @ mu)) / sig2 ** 2
            grad_noise = 0.5 * (tr_sinv - alpha_sq) * sig2

        if not (np.isfinite(grad_noise) and all(np.all(np.isfinite(g)) for g in grad_a)):
            return None                        # ill conditioned, so raise the noise
        if learn and not np.all(np.isfinite(grad_ell)):
            return None
        return nll, grad_a, float(grad_noise), grad_ell, (binv, ginv)

    def _fit_ml(self, blocks, offsets, C, b, yy, n, dists=None, fixed_sig2=None, tau=None):
        """Maximize the marginal likelihood with Adam on the log parameters.

        ``dists`` enables the shared learned lengthscales (main effects only);
        ``fixed_sig2`` freezes the noise level (used when a surface is fit on a
        residual whose noise is already known); ``tau`` is the hierarchical prior
        width on log-amplitudes (defaults to ``self.tau``).
        """
        n_kernels = sum(len(k) for k in blocks)
        init = float(np.log(0.5 / max(n_kernels, 1)))
        log_amps = [np.full(len(k), init) for k in blocks]
        log_noise = float(np.log(self.noise_init if fixed_sig2 is None
                                 else max(fixed_sig2, self.noise_floor)))
        tau = self.tau if tau is None else tau
        learn = bool(self.learn_scales) and dists is not None and any(D is not None for D in dists)
        centre = np.array([np.log(self.scales[0]), np.log(self.rbf_scales[0])])
        log_ell = centre.copy() if learn else None
        # units sharing a kernel count share a prior centre per kernel slot
        groups = {}
        for u, ks in enumerate(blocks):
            groups.setdefault(len(ks), []).append(u)
        m_a = [np.zeros_like(v) for v in log_amps]
        v_a = [np.zeros_like(v) for v in log_amps]
        m_n = v_n = 0.0
        m_e = np.zeros(2); v_e = np.zeros(2)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best = (np.inf, None, None, None)
        t = 0
        for _ in range(self.n_steps):
            out = self._nll_and_grad(blocks, offsets, C, b, yy, n, log_amps, log_noise,
                                     dists=dists if learn else None, log_ell=log_ell)
            if out is None:                # not positive definite, so raise the noise
                if fixed_sig2 is None:
                    log_noise += 0.25
                else:
                    break
                continue
            nll, grad_a, grad_n, grad_ell, _ = out
            if tau and tau > 0:
                # hierarchical prior: each kernel slot's log-amplitudes shrink toward
                # their centre across features (the centre is profiled out)
                for cnt, us in groups.items():
                    if len(us) < 2:
                        continue
                    M = np.stack([log_amps[u] for u in us])
                    dev = M - M.mean(axis=0, keepdims=True)
                    nll += float(np.sum(dev ** 2)) / (2.0 * tau ** 2)
                    for k_, u in enumerate(us):
                        grad_a[u] = grad_a[u] + dev[k_] / tau ** 2
            if learn and self.scale_prior and self.scale_prior > 0:
                nll += float(np.sum((log_ell - centre) ** 2)) / (2.0 * self.scale_prior ** 2)
                grad_ell = grad_ell + (log_ell - centre) / self.scale_prior ** 2
            if nll < best[0]:
                best = (nll, [v.copy() for v in log_amps], log_noise,
                        (log_ell.copy() if learn else None))
            t += 1
            for u in range(len(log_amps)):
                m_a[u] = b1 * m_a[u] + (1 - b1) * grad_a[u]
                v_a[u] = b2 * v_a[u] + (1 - b2) * grad_a[u] ** 2
                mh = m_a[u] / (1 - b1 ** t)
                vh = v_a[u] / (1 - b2 ** t)
                log_amps[u] = np.clip(log_amps[u] - self.lr * mh / (np.sqrt(vh) + eps), -25, 12)
            if fixed_sig2 is None:
                m_n = b1 * m_n + (1 - b1) * grad_n
                v_n = b2 * v_n + (1 - b2) * grad_n ** 2
                log_noise = float(np.clip(
                    log_noise - self.lr * (m_n / (1 - b1 ** t)) / (np.sqrt(v_n / (1 - b2 ** t)) + eps),
                    -25, 12))
            if learn:
                m_e = b1 * m_e + (1 - b1) * grad_ell
                v_e = b2 * v_e + (1 - b2) * grad_ell ** 2
                log_ell = log_ell - self.lr * (m_e / (1 - b1 ** t)) / (np.sqrt(v_e / (1 - b2 ** t)) + eps)

        nll_best, amps_best, noise_best, ell_best = best
        if amps_best is None:
            amps_best, noise_best, ell_best = log_amps, log_noise, log_ell
        amps = [np.exp(v) for v in amps_best]
        sig2 = max(float(np.exp(noise_best)), self.noise_floor)
        if learn:
            ell = np.exp(ell_best).clip(0.005, 2.0)
            self.scales_learned_ = (float(ell[0]), float(ell[1]))
            blocks = [ks if dists[u] is None else
                      [np.exp(-dists[u] / ell[0]), np.exp(-((dists[u] / ell[1]) ** 2))]
                      for u, ks in enumerate(blocks)]
        self._fitted_blocks = blocks

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
        post_var = None
        for ridge in (0.0, 1e-4, 1e-3, 1e-2):
            try:
                cf = cho_factor(G + ridge * np.eye(P), lower=True)
                fhat = cho_solve(cf, b / sig2)
                if np.isfinite(fhat).all() and np.abs(fhat).max() < 1e6:
                    # The posterior covariance of the bin values is the inverse
                    # of this same matrix. A shape function is only identified up
                    # to a constant, though, because any level shift can be
                    # absorbed by the intercept, so report the variance of the
                    # curve about its own mean rather than the raw diagonal.
                    cov = cho_solve(cf, np.eye(P))
                    post_var = np.empty(P)
                    for u in range(len(blocks)):
                        i0, i1 = int(offsets[u]), int(offsets[u + 1])
                        S = cov[i0:i1, i0:i1]
                        row_mean = S.mean(axis=1)
                        post_var[i0:i1] = np.diag(S) - 2 * row_mean + S.mean()
                    post_var = np.clip(post_var, 0.0, None)
                    break
            except np.linalg.LinAlgError:
                continue
        else:
            fhat = np.linalg.lstsq(G + np.eye(P), b / sig2, rcond=None)[0]
        if post_var is None:
            post_var = np.full(P, np.nan)
        return fhat, amps, (nll_best if np.isfinite(nll_best) else np.inf), post_var

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

        # Capacity grows with the sample size, but only for the knobs the caller
        # left alone: anything passed to the constructor takes precedence, so
        # asking for n_pairs=2 gets two pairs rather than the schedule's count.
        self._sched = {}
        if self.schedule:
            if len(y) <= 1000:
                # the pair count grows linearly with n at fixed grid resolution, so a
                # dataset never carries more than about four pair cells per row
                sched = dict(n_bins=96, p_budget=2500, pair_bins=16,
                             n_pairs=int(min(2 * d, 16, max(1, round(16 * len(y) / 1000)))),
                             pair_res=(16, 12))
            else:
                # up to five interactions per feature; the first 48 are fit jointly,
                # the rest are backfit on that model's residual
                sched = dict(n_bins=256, p_budget=4200, pair_bins=28,
                             n_pairs=int(min(5 * d, 250, round(16 * len(y) / 1000))),
                             pair_res=(28, 24, 16))
            defaults = {k: v.default for k, v in
                        inspect.signature(type(self).__init__).parameters.items()}
            self._sched = {k: v for k, v in sched.items()
                           if getattr(self, k) == defaults.get(k)}

        # 1. condition the target
        self.log_target_ = False
        if self.log_target in ("auto", True) and np.min(y) > 0:
            from scipy.stats import skew
            if self.log_target is True or abs(skew(np.log(y))) < abs(skew(y)) - 1.0:
                self.log_target_ = True
                y = np.log(y)
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        iqr = q3 - q1
        if iqr > 0:                            # clip only true outliers
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
        fhat, amps, _, post_var = self._fit_ml(blocks, offsets, C, b, yy, n,
                                               dists=[self._feature_dist(s) for s in sizes])
        blocks = self._fitted_blocks
        self.main_offsets_ = offsets
        self.main_values_ = fhat
        self.main_var_ = post_var
        self.main_amps_ = amps
        self.pairs_ = []
        self.pair_values_ = []

        # 5. screen interactions, 6. fit them in blocks
        n_pairs = self._p("n_pairs")
        if n_pairs > 0 and len(units) >= 2:
            resid = yn.copy()
            for u, j in enumerate(units):
                resid -= fhat[offsets[u]:offsets[u + 1]][bidx[:, j]]
            selected = self._screen_pairs(X, units, resid, n_pairs, amps)
            extra = selected[48:]
            selected = selected[:48]
            if selected:
                fhat, self.pairs_, self.pair_values_, post_var, amps = self._fit_pairs(
                    X, bidx, units, sizes, blocks, C, b, yy, n, offsets,
                    fhat, selected, yn)
                self.main_values_ = fhat
                self.main_var_ = post_var
                self.main_amps_ = amps
            if extra:
                self._backfit_extra(X, yn, n, extra)

        rng = float(np.max(y) - np.min(y))
        self.clip_ = (float(np.min(y)) - 0.05 * rng, float(np.max(y)) + 0.05 * rng)
        self.bias_ = 0.0
        pred = self.predict(X)
        pred_t = np.log(np.maximum(pred, 1e-300)) if self.log_target_ else pred
        self.bias_ = float(np.mean(y) - np.mean(pred_t))
        return self

    def _screen_pairs(self, X, units, resid, n_pairs, amps):
        """Rank candidate interactions by their shrunken residual cell means."""
        feats = units
        if len(units) * (len(units) - 1) // 2 > 5000:      # too many pairs to score
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

        Terms are fit in chunks so that the terms within a chunk share shrinkage.
        Each chunk is fit at every candidate grid resolution, and the marginal
        likelihood picks which resolution to keep.
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
                # what this chunk has to explain: the target, minus the main
                # effects, minus every interaction outside the chunk
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
                    fc, _, nll, _ = self._fit_ml(kern, offc, Cc, bc,
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
            main_vals, main_amps, _, main_var = self._fit_ml(blocks, offsets, C, bm,
                                                             float(np.sum(adj ** 2)), n)
        keep = [p for p in selected if defs[p] is not None]
        return (main_vals, [defs[p] for p in keep], [vals[p] for p in keep],
                main_var, main_amps)

    def _backfit_extra(self, X, yn, n, extra):
        """Interactions beyond the first 48: each surface is an exact 2-D GP fit to
        the residual of the joint model, swept ``sweeps`` times with a shared noise
        level; its grid resolution is chosen by marginal likelihood in the first
        sweep. Linear in the number of pairs, which is what lets a wide dataset
        carry five interactions per feature."""
        self.clip_, self.bias_ = (-np.inf, np.inf), 0.0
        pred = self.predict(X)
        pred_t = np.log(np.maximum(pred, 1e-300)) if self.log_target_ else pred
        resid = yn - (pred_t - self.y_mean_) / self.y_std_
        menu = sorted(set(self._p("pair_res") or (self._p("pair_bins"),)), reverse=True)[:2]
        grids = []
        for (a, b_) in extra:
            cands = []
            for R in menu:
                ea, eb = self._bin_edges(X[:, a], R), self._bin_edges(X[:, b_], R)
                if ea is None or eb is None:
                    continue
                na, nb = len(ea) + 1, len(eb) + 1
                cols = (np.searchsorted(ea, X[:, a], side="right") * nb
                        + np.searchsorted(eb, X[:, b_], side="right"))
                cands.append((cols, na * nb, dict(i=a, j=b_, ei=ea, ej=eb, na=na, nb=nb),
                              self._pair_kernels(na, nb)))
            if cands:
                grids.append(cands)
        K = len(grids)
        chosen = [0] * K
        vals = [np.zeros(g[0][1]) for g in grids]
        cols_k = [g[0][0] for g in grids]
        T = np.zeros(n)
        for sw in progress_iter(range(int(self.sweeps)), verbose=self.verbose,
                                desc='backfitting sweeps'):
            s2 = float(np.var(resid - T))
            for k in range(K):
                r_k = resid - T + vals[k][cols_k[k]]
                rr = float(np.sum(r_k ** 2))
                best = None
                for ci in (range(len(grids[k])) if sw == 0 else [chosen[k]]):
                    cols, size, _, kern = grids[k][ci]
                    cnt = np.bincount(cols, minlength=size).astype(float)
                    b_k = np.bincount(cols, weights=r_k, minlength=size)
                    f_c, _, nll_c, _ = self._fit_ml([kern], np.array([0, size]), np.diag(cnt),
                                                    b_k, rr, n, fixed_sig2=s2, tau=0.0)
                    if best is None or nll_c < best[0]:
                        best = (nll_c, ci, f_c)
                _, ci, f_k = best
                if ci != chosen[k]:
                    chosen[k] = ci
                    cols_k[k] = grids[k][ci][0]
                    vals[k] = np.zeros(grids[k][ci][1])
                T += f_k[cols_k[k]] - vals[k][cols_k[k]]
                vals[k] = f_k
        self.pairs_ = list(self.pairs_) + [grids[k][chosen[k]][2] for k in range(K)]
        self.pair_values_ = list(self.pair_values_) + vals

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
    def shape_function(self, feature, return_std=False):
        """Return ``(grid, values)``, the fitted curve for one feature.

        Pass ``return_std=True`` to also get ``std``, the posterior standard
        deviation of the curve at each bin. The Gaussian process supplies this
        from the same fit, at no extra cost beyond one solve.

        The standard deviation is for the curve measured about its own mean. A
        shape function is only identified up to a constant, since any level shift
        can be absorbed by the intercept, so the raw per-bin variance is mostly a
        shared offset and would overstate how uncertain the shape is.

        The values are on the scale the model was fit on, which is standardized
        and may be logged. That is the scale on which the model is additive.
        """
        check_is_fitted(self, "main_values_")
        j = int(feature)
        if j not in self.edges_:
            raise ValueError(f"feature {j} was constant and carries no shape function")
        u = self.units_.index(j)
        i0, i1 = self.main_offsets_[u], self.main_offsets_[u + 1]
        grid = self.grids_[j].copy()
        values = self.main_values_[i0:i1].copy() * self.y_std_
        if not return_std:
            return grid, values
        std = np.sqrt(self.main_var_[i0:i1]) * self.y_std_
        return grid, values, std

    def kernel_weights(self, feature):
        """Return ``{kernel: amplitude}`` for one feature's prior.

        The amplitudes are what the marginal likelihood chose, and their balance
        is how the model expresses smoothness: weight on the short Matern kernel
        buys a curve that can turn sharply, weight on the squared exponential
        buys a gentle one. A feature that explains nothing ends up with every
        amplitude near zero, which is how irrelevant features drop out.
        """
        check_is_fitted(self, "main_amps_")
        j = int(feature)
        if j not in self.edges_:
            raise ValueError(f"feature {j} was constant and carries no shape function")
        u = self.units_.index(j)
        n_bins = len(self.grids_[j])
        if n_bins <= 3:
            names = ["delta"]
        else:
            sc = getattr(self, "scales_learned_", None)
            mat = [sc[0]] if sc else list(self.scales)
            rbf = [sc[1]] if sc else list(self.rbf_scales)
            names = ["matern-%.3g" % v for v in mat] + ["rbf-%.3g" % v for v in rbf]
        amps = np.asarray(self.main_amps_[u], dtype=float)
        return {nm: float(a) for nm, a in zip(names, amps[:len(names)])}

    def interaction_terms(self):
        """List the fitted pairwise interactions as ``(feature_a, feature_b)``."""
        check_is_fitted(self, "main_values_")
        return [(t["i"], t["j"]) for t in self.pairs_]
