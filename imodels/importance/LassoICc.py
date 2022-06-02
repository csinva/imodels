import copy
import warnings
import numpy as np
import numbers

from math import log
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.linear_model import LassoLars,lars_path
from sklearn.linear_model import LinearRegression


class LassoLarsICc(LassoLars):
    
    """Lasso model fit with Lars using IC for model selection.
    The optimization objective for Lasso is::
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    AIC is the Akaike information criterion [2]_ and BIC is the Bayes
    Information criterion [3]_. Such criteria are useful to select the value
    of the regularization parameter by making a trade-off between the
    goodness of fit and the complexity of the model. A good model should
    explain well the data while being simple.
    Read more in the :ref:`User Guide <lasso_lars_ic>`.
    Parameters
    ----------
    criterion : {'aic', 'bic','aic_c'}, default='aic_c'
        The type of criterion to use.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    verbose : bool or int, default=False
        Sets the verbosity amount.
    normalize : bool, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
            ``normalize`` was deprecated in version 1.0. It will default
            to False in 1.2 and be removed in 1.4.
    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.
    max_iter : int, default=500
        Maximum number of iterations to perform. Can be used for
        early stopping.
    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsIC only makes sense for problems where
        a sparse solution is expected and/or reached.
    noise_variance : float, default=None
        The estimated noise variance of the data. If `None`, an unbiased
        estimate is computed by an OLS model. However, it is only possible
        in the case where `n_samples > n_features + fit_intercept`.
        .. versionadded:: 1.1
    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)
    intercept_ : float
        independent term in decision function.
    alpha_ : float
        the alpha parameter chosen by the information criterion
    alphas_ : array-like of shape (n_alphas + 1,) or list of such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller. If a list, it will be of length `n_targets`.
    n_iter_ : int
        number of iterations run by lars_path to find the grid of
        alphas.
    criterion_ : array-like of shape (n_alphas,)
        The value of the information criteria ('aic', 'bic') across all
        alphas. The alpha which has the smallest information criterion is
        chosen, as specified in [1]_.
    noise_variance_ : float
        The estimated noise variance from the data used to compute the
        criterion.
        .. versionadded:: 1.1
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoLarsCV: Cross-validated Lasso, using the LARS algorithm.
    sklearn.decomposition.sparse_encode : Sparse coding.
    Notes
    -----
    The number of degrees of freedom is computed as in [1]_.
    To have more details regarding the mathematical formulation of the
    AIC and BIC criteria, please refer to :ref:`User Guide <lasso_lars_ic>`.
    References
    ----------
    .. [1] :arxiv:`Zou, Hui, Trevor Hastie, and Robert Tibshirani.
            "On the degrees of freedom of the lasso."
            The Annals of Statistics 35.5 (2007): 2173-2192.
            <0712.0881>`
    .. [2] `Wikipedia entry on the Akaike information criterion
            <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_
    .. [3] `Wikipedia entry on the Bayesian information criterion
            <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_
    Examples
    --------
    >>> from sklearn import linear_model
    >>> reg = linear_model.LassoLarsIC(criterion='bic', normalize=False)
    >>> X = [[-2, 2], [-1, 1], [0, 0], [1, 1], [2, 2]]
    >>> y = [-2.2222, -1.1111, 0, -1.1111, -2.2222]
    >>> reg.fit(X, y)
    LassoLarsIC(criterion='bic', normalize=False)
    >>> print(reg.coef_)
    [ 0.  -1.11...]
    """

    def __init__(
        self,
        criterion="aic_c",
        *,
        fit_intercept=True,
        verbose=False,
        normalize="deprecated",
        precompute="auto",
        max_iter=500,
        eps=np.finfo(float).eps,
        copy_X=True,
        positive=False,
        noise_variance=None,
        use_noise_variance = True
    ):
        self.criterion = criterion
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.max_iter = max_iter
        self.verbose = verbose
        self.normalize = normalize
        self.copy_X = copy_X
        self.precompute = precompute
        self.eps = eps
        self.fit_path = True
        self.noise_variance = noise_variance
        self.use_noise_variance = use_noise_variance

    def _more_tags(self):
        return {"multioutput": False}

    def fit(self, X, y, copy_X=None):
        """Fit the model using X, y as training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.
        copy_X : bool, default=None
            If provided, this parameter will override the choice
            of copy_X made at instance creation.
            If ``True``, X will be copied; else, it may be overwritten.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        _normalize = _deprecate_normalize(
            self.normalize, default=True, estimator_name=self.__class__.__name__
        )

        if copy_X is None:
            copy_X = self.copy_X
        X, y = self._validate_data(X, y, y_numeric=True)

        X, y, Xmean, ymean, Xstd = _preprocess_data(
            X, y, self.fit_intercept, _normalize, copy_X
        )

        Gram = self.precompute

        alphas_, _, coef_path_, self.n_iter_ = lars_path(
            X,
            y,
            Gram=Gram,
            copy_X=copy_X,
            copy_Gram=True,
            alpha_min=0.0,
            method="lasso",
            verbose=self.verbose,
            max_iter=self.max_iter,
            eps=self.eps,
            return_n_iter=True,
            positive=self.positive,
        )

        n_samples = X.shape[0]

        if self.criterion == "aic":
            criterion_factor = 2
        elif self.criterion == "bic":
            criterion_factor = log(n_samples)
#        else:
#            raise ValueError(
#                f"criterion should be either bic or aic, got {self.criterion!r}"
#            )

        residuals = y[:, np.newaxis] - np.dot(X, coef_path_)
        residuals_sum_squares = np.sum(residuals**2, axis=0)
        
        TSS = np.sum((y - np.average(y))**2)
        FSS = TSS - residuals_sum_squares
        R_squared = 1 - residuals_sum_squares/TSS
        
        degrees_of_freedom = np.zeros(coef_path_.shape[1], dtype=int)
        for k, coef in enumerate(coef_path_.T):
            mask = np.abs(coef) > np.finfo(coef.dtype).eps
            if not np.any(mask):
                continue
            # get the number of degrees of freedom equal to:
            # Xc = X[:, mask]
            # Trace(Xc * inv(Xc.T, Xc) * Xc.T) ie the number of non-zero coefs
            degrees_of_freedom[k] = np.sum(mask)

        self.alphas_ = alphas_

        if self.noise_variance is None:
            self.noise_variance_ = self._estimate_noise_variance(
                X, y, positive=self.positive
            )
        else:
            self.noise_variance_ = self.noise_variance
        
        if self.criterion == "aic_c":
            if self.use_noise_variance == True:
                self.criterion_ = (
                    n_samples * np.log(2 * np.pi * self.noise_variance_)
                    + residuals_sum_squares / self.noise_variance_
                    + (2 * degrees_of_freedom + 2 * degrees_of_freedom**2)/(n_samples - degrees_of_freedom - 1)
                )
            else:
                self.criterion_ = (
                    n_samples*np.log(residuals_sum_squares) +2*degrees_of_freedom +  
                    (2 * degrees_of_freedom + 2 * degrees_of_freedom**2)/(n_samples - degrees_of_freedom - 1)
                )
        elif self.criterion == "gMDL":
            if self.use_noise_variance == True:
                criterion_ = np.zeros(len(degrees_of_freedom))
                for i in range(len(degrees_of_freedom)):
                    if FSS[i] > degrees_of_freedom[i]*self.noise_variance_:
                        criterion_[i] = residuals_sum_squares[i]/(2*self.noise_variance_) + 0.5*degrees_of_freedom[i]*(1.0 + np.log(FSS[i]/(degrees_of_freedom[i]*self.noise_variance_))) + 0.5*np.log(n_samples)
                    else:
                        criterion_[i] = TSS/(2*self.noise_variance_)
                self.criterion_ = criterion_
            else:
                criterion_ = np.zeros(len(degrees_of_freedom))   #self.criterion_ = np.zeros(len(degrees_of_freedom))
                for i in range(len(degrees_of_freedom)):
                    if R_squared[i] >= degrees_of_freedom[i]/n_samples:
                        S = residuals_sum_squares[i]/(n_samples-degrees_of_freedom[i])
                        F = FSS[i]/(degrees_of_freedom[i]*S)
                        criterion_[i] = 0.5*n_samples*np.log(S) + degrees_of_freedom[i]*0.5*np.log(F) + np.log(n_samples)
                    else:
                         criterion_[i] = 0.5*n_samples*np.log(TSS/n_samples) + 0.5*np.log(n_samples)
                self.criterion_ = criterion_
        else:
            if self.use_noise_variance == True:
                self.criterion_ = (
                    n_samples * np.log(2 * np.pi * self.noise_variance_)
                    + residuals_sum_squares / self.noise_variance_
                    + criterion_factor * degrees_of_freedom
                )
            else:
                self.criterion_ = ( 
                    n_samples*np.log(residuals_sum_squares) + criterion_factor*degrees_of_freedom
                )
        n_best = np.argmin(self.criterion_)

        self.alpha_ = alphas_[n_best]
        self.coef_ = coef_path_[:, n_best]
        self._set_intercept(Xmean, ymean, Xstd)
        return self

    def _estimate_noise_variance(self, X, y, positive):
        """Compute an estimate of the variance with an OLS model.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to be fitted by the OLS model. We expect the data to be
            centered.
        y : ndarray of shape (n_samples,)
            Associated target.
        positive : bool, default=False
            Restrict coefficients to be >= 0. This should be inline with
            the `positive` parameter from `LassoLarsIC`.
        Returns
        -------
        noise_variance : float
            An estimator of the noise variance of an OLS model.
        """
        if X.shape[0] <= X.shape[1] + self.fit_intercept:
            raise ValueError(
                f"You are using {self.__class__.__name__} in the case where the number "
                "of samples is smaller than the number of features. In this setting, "
                "getting a good estimate for the variance of the noise is not "
                "possible. Provide an estimate of the noise variance in the "
                "constructor."
            )
        # X and y are already centered and we don't need to fit with an intercept
        ols_model = LinearRegression(positive=positive, fit_intercept=False)
        y_pred = ols_model.fit(X, y).predict(X)
        return np.sum((y - y_pred) ** 2) / (
            X.shape[0] - X.shape[1] - self.fit_intercept
        )

    
    
    
    


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    sample_weight=None,
    check_input=True,
):
    """Center and scale data.
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    fit_intercept=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
        If normalize is True, then X_out is rescaled (dense and sparse case)
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Likely performed inplace on input y.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    X_scale : ndarray of shape (n_features,)
        The standard deviation per column of input X.
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order="K")

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            if normalize:
                X_offset, X_var, _ = _incremental_mean_and_var(
                    X,
                    last_mean=0.0,
                    last_variance=0.0,
                    last_sample_count=0.0,
                    sample_weight=sample_weight,
                )
            else:
                X_offset = np.average(X, axis=0, weights=sample_weight)

            X_offset = X_offset.astype(X.dtype, copy=False)
            X -= X_offset

        if normalize:
            X_var = X_var.astype(X.dtype, copy=False)
            # Detect constant features on the computed variance, before taking
            # the np.sqrt. Otherwise constant features cannot be detected with
            # sample weights.
            constant_mask = _is_constant_feature(X_var, X_offset, X.shape[0])
            if sample_weight is None:
                X_var *= X.shape[0]
            else:
                X_var *= sample_weight.sum()
            X_scale = np.sqrt(X_var, out=X_var)
            X_scale[constant_mask] = 1.0
            if sp.issparse(X):
                inplace_column_scale(X, 1.0 / X_scale)
            else:
                X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale



def _deprecate_normalize(normalize, default, estimator_name):
    """Normalize is to be deprecated from linear models and a use of
    a pipeline with a StandardScaler is to be recommended instead.
    Here the appropriate message is selected to be displayed to the user
    depending on the default normalize value (as it varies between the linear
    models and normalize value selected by the user).
    Parameters
    ----------
    normalize : bool,
        normalize value passed by the user
    default : bool,
        default normalize value used by the estimator
    estimator_name : str
        name of the linear estimator which calls this function.
        The name will be used for writing the deprecation warnings
    Returns
    -------
    normalize : bool,
        normalize value which should further be used by the estimator at this
        stage of the depreciation process
    Notes
    -----
    This function should be updated in 1.2 depending on the value of
    `normalize`:
    - True, warning: `normalize` was deprecated in 1.2 and will be removed in
      1.4. Suggest to use pipeline instead.
    - False, `normalize` was deprecated in 1.2 and it will be removed in 1.4.
      Leave normalize to its default value.
    - `deprecated` - this should only be possible with default == False as from
      1.2 `normalize` in all the linear models should be either removed or the
      default should be set to False.
    This function should be completely removed in 1.4.
    """

    if normalize not in [True, False, "deprecated"]:
        raise ValueError(
            "Leave 'normalize' to its default value or set it to True or False"
        )

    if normalize == "deprecated":
        _normalize = default
    else:
        _normalize = normalize

    pipeline_msg = (
        "If you wish to scale the data, use Pipeline with a StandardScaler "
        "in a preprocessing stage. To reproduce the previous behavior:\n\n"
        "from sklearn.pipeline import make_pipeline\n\n"
        "model = make_pipeline(StandardScaler(with_mean=False), "
        f"{estimator_name}())\n\n"
        "If you wish to pass a sample_weight parameter, you need to pass it "
        "as a fit parameter to each step of the pipeline as follows:\n\n"
        "kwargs = {s[0] + '__sample_weight': sample_weight for s "
        "in model.steps}\n"
        "model.fit(X, y, **kwargs)\n\n"
    )

    if estimator_name == "Ridge" or estimator_name == "RidgeClassifier":
        alpha_msg = "Set parameter alpha to: original_alpha * n_samples. "
    elif "Lasso" in estimator_name:
        alpha_msg = "Set parameter alpha to: original_alpha * np.sqrt(n_samples). "
    elif "ElasticNet" in estimator_name:
        alpha_msg = (
            "Set parameter alpha to original_alpha * np.sqrt(n_samples) if "
            "l1_ratio is 1, and to original_alpha * n_samples if l1_ratio is "
            "0. For other values of l1_ratio, no analytic formula is "
            "available."
        )
    elif estimator_name in ("RidgeCV", "RidgeClassifierCV", "_RidgeGCV"):
        alpha_msg = "Set parameter alphas to: original_alphas * n_samples. "
    else:
        alpha_msg = ""

    if default and normalize == "deprecated":
        warnings.warn(
            "The default of 'normalize' will be set to False in version 1.2 "
            "and deprecated in version 1.4.\n"
            + pipeline_msg
            + alpha_msg,
            FutureWarning,
        )
    elif normalize != "deprecated" and normalize and not default:
        warnings.warn(
            "'normalize' was deprecated in version 1.0 and will be removed in 1.2.\n"
            + pipeline_msg
            + alpha_msg,
            FutureWarning,
        )
    elif not normalize and not default:
        warnings.warn(
            "'normalize' was deprecated in version 1.0 and will be "
            "removed in 1.2. "
            "Please leave the normalize parameter to its default value to "
            "silence this warning. The default behavior of this estimator "
            "is to not do any normalization. If normalization is needed "
            "please use sklearn.preprocessing.StandardScaler instead.",
            FutureWarning,
        )

    return _normalize

