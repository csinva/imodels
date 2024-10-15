import numpy as np
import logging
from neuro.data.utils import mult_diag, counter
import random
import joblib
import itertools as itools


def zs(v): return (v-v.mean(0))/v.std(0)  # z-score function


ridge_logger = logging.getLogger("ridge_corr")


def gen_temporal_chunk_splits(num_splits: int, num_examples: int, chunk_len: int, num_chunks: int):
    all_indexes = range(num_examples)
    index_chunks = list(zip(*[iter(all_indexes)] * chunk_len))
    splits_list = []
    for _ in range(num_splits):
        random.shuffle(index_chunks)
        tune_indexes_ = list(itools.chain(*index_chunks[:num_chunks]))
        train_indexes_ = list(set(all_indexes)-set(tune_indexes_))
        splits_list.append((train_indexes_, tune_indexes_))
    return splits_list


def _ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    U, S, Vh = np.linalg.svd(stim, full_matrices=False)
    UR = np.dot(U.T, np.nan_to_num(resp))

    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float, int)):
        alpha = np.ones(resp.shape[1]) * alpha

    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas == ua)[0]
        # awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = Vh.T.dot(np.diag(S/(S**2+ua**2))).dot(UR[:, selvox])
        wt[:, selvox] = awt

    return wt


def _ridge_corr_pred(X_train, X_test, y_train, y_test, valphas, normalpha=False,
                     singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. Returns the correlation 
    between predicted and actual [Presp], without ever computing the regression weights.
    This function assumes that each voxel is assigned a separate alpha in [valphas].

    Parameters
    ----------
    X_train : array_like, shape (n_train, n_features)
    X_test : array_like, shape (n_test, n_features)
    y_train : array_like, shape (n_train, n_targets)
    y_test : array_like, shape (n_test, n_targets)
    valphas : list or array_like, shape (M,)
        Ridge parameter for each voxel.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    corr : array_like, shape (M,)
        The correlation between each predicted response and each column of Presp.

    """
    # Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    U, S, Vh = np.linalg.svd(X_train, full_matrices=False)

    # Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:, :ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)" %
                (nbad, str(U.shape)))

    # Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f" % norm)
    if normalpha:
        nalphas = valphas * norm
    else:
        nalphas = valphas

    # Precompute some products for speed
    UR = np.dot(U.T, y_train)  # Precompute this matrix product for speed
    PVh = np.dot(X_test, Vh.T)  # Precompute this matrix product for speed

    # Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(y_test)
    # Prespvar = Presp.var(0)
    Prespvar_actual = y_test.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (
        Prespvar_actual - Prespvar).mean())

    ualphas = np.unique(nalphas)
    corr = np.zeros((y_train.shape[1],))
    for ua in ualphas:
        selvox = np.nonzero(nalphas == ua)[0]
        alpha_pred = PVh.dot(np.diag(S/(S**2+ua**2))).dot(UR[:, selvox])

        if use_corr:
            corr[selvox] = (zPresp[:, selvox] * zs(alpha_pred)).mean(0)
        else:
            resvar = (y_test[:, selvox] - alpha_pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            corr[selvox] = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

    return corr


def _ridge_corr(
    X_train, X_test, y_train, y_test, alphas, normalpha=False, corrmin=0.2,
        singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    Assumes features are Z-scored across time.

    Parameters
    ----------
    X_train : array_like, shape (n_train, n_features)
    X_test : array_like, shape (n_test, n_features)
    y_train : array_like, shape (n_train, n_targets)
    y_test : array_like, shape (n_test, n_targets)
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.

    """
    # Calculate SVD of stimulus matrix
    logger.debug("Doing SVD...")
    try:
        U, S, Vh = np.linalg.svd(X_train, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.debug("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U, S, Vh = svd_dgesvd(X_train, full_matrices=False)

    # Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:, :ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.debug("Dropped %d tiny singular values.. (U is now %s)" %
                 (nbad, str(U.shape)))

    # Normalize alpha by the LSV norm
    norm = S[0]
    logger.debug("Training stimulus has LSV norm: %0.03f" % norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    # Precompute some products for speed
    UR = np.dot(U.T, y_train)  # Precompute this matrix product for speed
    PVh = np.dot(X_test, Vh.T)  # Precompute this matrix product for speed

    # Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(y_test)
    # Prespvar = Presp.var(0)
    Prespvar_actual = y_test.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.debug("Average difference between actual & assumed Prespvar: %0.3f" % (
        Prespvar_actual - Prespvar).mean())
    Rcorrs = []  # Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        # D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter
        # Reweight singular vectors by the (normalized?) ridge parameter
        D = S / (S ** 2 + na ** 2)

        # Best (1.75 seconds to prediction in test)
        pred = np.dot(mult_diag(D, PVh, left=False), UR)

        if use_corr:
            # prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            # Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            # Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            # Compute variance explained
            resvar = (y_test - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)

        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr > corrmin).sum()-(-Rcorr > corrmin).sum())
        logger.debug(log_msg)

    return Rcorrs


def bootstrap_ridge(
        X_train, y_train, X_test, y_test, alphas, nboots, chunklen, nchunks,
        corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
        use_corr=True, return_wt=True, logger=ridge_logger):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.

    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist 
    will be given the same regularization parameter (the one that is the best on average).

    Parameters
    ----------
    X_train : array_like, shape (n_train, n_features)
    X_test : array_like, shape (n_test, n_features)
    y_train : array_like, shape (n_train, n_targets)
    y_test : array_like, shape (n_test, n_targets)
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times 
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this 
        product should be about 20 percent of the total length of the training data.
    corrmin : float in [0..1], default 0.2
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    joined : None or list of array_like indices, default None
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    singcutoff : float, default 1e-10
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean, default False
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    return_wt : boolean, default True
        If True, this function will compute and return the regression weights after finding the best
        alpha parameter for each voxel. However, for very large models this can lead to memory issues.
        If false, this function will _not_ compute weights, but will still compute prediction performance
        on the prediction dataset (Pstim, Presp).

    Returns
    -------
    wt : array_like, shape (N, M)
        If [return_wt] is True, regression weights for N features and M responses. If [return_wt] is False, [].
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = y_train.shape
    splits = gen_temporal_chunk_splits(
        nboots, nresp, chunklen, nchunks)
    valinds = [splits[1] for splits in splits]

    correlation_matrices = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.debug("Selecting held-out test set..")

        # get indices for training / testing
        train_indexes_, tune_indexes_ = splits[bi]

        # Select data
        X_train_ = X_train[train_indexes_, :]
        X_test_ = X_train[tune_indexes_, :]
        y_train_ = y_train[train_indexes_, :]
        y_test_ = y_train[tune_indexes_, :]

        # Run ridge regression using this test set
        correlation_matrix_ = _ridge_corr(
            X_train_, X_test_, y_train_, y_test_, alphas,
            corrmin=corrmin, singcutoff=singcutoff,
            normalpha=normalpha, use_corr=use_corr,
            logger=logger)
        correlation_matrices.append(correlation_matrix_)

    # Find best alphas
    if nboots > 0:
        all_correlation_matrices = np.dstack(correlation_matrices)
    else:
        all_correlation_matrices = None

    if not single_alpha:
        if nboots == 0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")

        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = all_correlation_matrices.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = all_correlation_matrices[:, jl, :].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
    else:
        logger.debug("Finding single best alpha..")
        if nboots == 0:
            if len(alphas) == 1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = all_correlation_matrices.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]

        valphas = np.array([bestalpha]*nvox)
        logger.debug("Best alpha = %0.3f" % bestalpha)

    if return_wt:
        # Find weights
        logger.debug(
            "Computing weights for each response using entire training set..")
        wt = _ridge(X_train, y_train, valphas,
                    singcutoff=singcutoff, normalpha=normalpha)

        # Predict responses on prediction set
        logger.debug("Predicting responses for predictions set..")
        pred = np.dot(X_test, wt)

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        if use_corr:
            corrs = np.nan_to_num(np.array([np.corrcoef(y_test[:, ii], nnpred[:, ii].ravel())[0, 1]
                                            for ii in range(y_test.shape[1])]))
        else:
            residual_variance = (y_test-pred).var(0)
            residual_sum_of_squares = 1 - \
                (residual_variance / y_test.var(0))
            corrs = np.sqrt(np.abs(residual_sum_of_squares)) * \
                np.sign(residual_sum_of_squares)

        return wt, corrs, valphas, all_correlation_matrices, valinds
    else:
        # get correlations for prediction dataset directly
        corrs = _ridge_corr_pred(
            X_train, X_test, y_train, X_test, valphas,
            normalpha=normalpha, use_corr=use_corr, logger=logger, singcutoff=singcutoff)

        return [], corrs, valphas, all_correlation_matrices, valinds


if __name__ == '__main__':
    # sample data for ridge regression
    np.random.seed(0)

    # set logging to debug
    logging.basicConfig(level=logging.DEBUG)

    params = joblib.load('example_params.joblib')
    print(params.keys())
    wt, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
        X_train=params['features_train_delayed'], y_train=params['y_train'],
        X_test=params['features_test_delayed'], y_test=params['y_test'],
        alphas=params['alphas'], nboots=params['nboots'],
        chunklen=params['chunklen'], nchunks=params['nchunks'],
        singcutoff=params['singcutoff'], single_alpha=params['single_alpha'])
