'''Original code by Alex Huth and Huth lab for predicting fMRI responses
(see https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/ridge.py)

These functions help to predict many outputs using ridge regression efficiently.
'''

import time
import numpy as np
from tqdm import tqdm
import logging
import joblib
import itertools as itools
from imodels.algebraic.ridge_multi_utils import mult_diag, _z_score, ridge_logger, _counter
from sklearn.utils.extmath import randomized_svd


def _gen_temporal_chunk_splits(num_splits: int, num_examples: int, chunk_len: int, num_chunks: int, seed=42):
    '''Make a list of splits for cross-validation, where splits are temporal chunks of data.
    '''
    rng = np.random.RandomState(seed)
    all_indexes = range(num_examples)
    index_chunks = list(zip(*[iter(all_indexes)] * chunk_len))
    splits_list = []
    for _ in range(num_splits):
        rng.shuffle(index_chunks)
        tune_indexes_ = list(itools.chain(*index_chunks[:num_chunks]))
        train_indexes_ = list(set(all_indexes)-set(tune_indexes_))
        splits_list.append((train_indexes_, tune_indexes_))
    return splits_list


def _ridge(X, y, alpha, singcutoff=1e-10, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    X : array_like, shape (n_train, n_features)
    y : array_like, shape (n_train, n_targets)
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
    # ridge = Ridge(alpha=alpha**2, fit_intercept=False)
    # ridge.fit(X_train, y_train)
    # return ridge.coef_.T

    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    UR = np.dot(U.T, np.nan_to_num(y))

    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float, int)):
        alpha = np.ones(y.shape[1]) * alpha

    # Normalize alpha by the LSV norm
    nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((X.shape[1], y.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas == ua)[0]
        # awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = Vh.T.dot(np.diag(S/(S**2+ua**2))).dot(UR[:, selvox])
        wt[:, selvox] = awt

    return wt


def _ridge_correlations_per_voxel(X_train, X_test, y_train, y_test, valphas,
                                  singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Returns the correlation between y_test and ridge-regression predictions for y_test.
    Never actually needs to compute the regression weights (for speed).
    Assume every target is assigned a separate alpha.

    Parameters
    ----------
    X_train : array_like, shape (n_train, n_features)
    X_test : array_like, shape (n_test, n_features)
    y_train : array_like, shape (n_train, n_targets)
    y_test : array_like, shape (n_test, n_targets)
    valphas : list or array_like, shape (n_targets,)
        Ridge parameter for each voxel.
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
    corr : array_like, shape (n_targets,)
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

    # Precompute some products for speed
    UR = np.dot(U.T, y_train)  # Precompute this matrix product for speed
    PVh = np.dot(X_test, Vh.T)  # Precompute this matrix product for speed

    # Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    y_test_normalized = _z_score(y_test)
    # y_test_var = Presp.var(0)
    y_test_var_actual = y_test.var(0)
    y_test_var = (np.ones_like(y_test_var_actual) + y_test_var_actual) / 2.0
    logger.info("Average difference between actual & assumed y_test_var: %0.3f" % (
        y_test_var_actual - y_test_var).mean())

    ualphas = np.unique(valphas)
    corr = np.zeros((y_train.shape[1],))
    for ua in ualphas:
        selvox = np.nonzero(valphas == ua)[0]
        alpha_pred = PVh.dot(np.diag(S/(S**2+ua**2))).dot(UR[:, selvox])

        if use_corr:
            corr[selvox] = (y_test_normalized[:, selvox]
                            * _z_score(alpha_pred)).mean(0)
        else:
            resvar = (y_test[:, selvox] - alpha_pred).var(0)
            Rsq = 1 - (resvar / y_test_var)
            corr[selvox] = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

    return corr


def _ridge_correlations_per_voxel_per_alpha(
    X_train, X_test, y_train, y_test, alphas, corrmin=0.2,
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
    alphas : list or array_like, shape (n_alphas,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
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
    y_corrs : array_like, shape (n_alphas, n_targets)
        The correlation between each predicted response and each column of y_test for each alpha.

    """
    # Calculate SVD of stimulus matrix
    logger.debug("Doing SVD...")
    U, S, Vh = np.linalg.svd(X_train, full_matrices=False)

    # Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:, :ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.debug("Dropped %d tiny singular values.. (U is now %s)" %
                 (nbad, str(U.shape)))

    # Precompute some products for speed
    UR = np.dot(U.T, y_train)  # Precompute this matrix product for speed
    PVh = np.dot(X_test, Vh.T)  # Precompute this matrix product for speed

    # Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    y_test_normalized = _z_score(y_test)
    # y_test_var = Presp.var(0)
    y_test_var_actual = y_test.var(0)
    y_test_var = (np.ones_like(y_test_var_actual) + y_test_var_actual) / 2.0
    logger.debug("Average difference between actual & assumed y_test_var: %0.3f" % (
        y_test_var_actual - y_test_var).mean())
    Rcorrs = []  # Holds training correlations for each alpha
    for na, a in zip(alphas, alphas):
        # Reweight singular vectors by the (normalized?) ridge parameter
        D = S / (S ** 2 + na ** 2)

        pred = np.dot(mult_diag(D, PVh, left=False), UR)

        if use_corr:
            Rcorr = (y_test_normalized * _z_score(pred)).mean(0)
        else:
            # Compute variance explained
            resvar = (y_test - pred).var(0)
            Rsq = 1 - (resvar / y_test_var)
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
        corrmin=0.2, joined=None, singcutoff=1e-10, single_alpha=False,
        use_corr=True, return_wt=True, logger=ridge_logger, decrease_alpha: int = 0):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.

    First, [nchunks] random chunks of length [chunklen] will be taken from [X_train] and [y_train] for each regression
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
    n_train, n_targets = y_train.shape
    splits = _gen_temporal_chunk_splits(
        nboots, n_train, chunklen, nchunks)
    valinds = [splits[1] for splits in splits]

    correlation_matrices = []
    for idx_bootstrap in _counter(range(nboots), countevery=1, total=nboots):
        logger.debug("Selecting held-out test set..")

        # get indices for training / testing
        train_indexes_, tune_indexes_ = splits[idx_bootstrap]

        # Select data
        X_train_ = X_train[train_indexes_, :]
        X_tune_ = X_train[tune_indexes_, :]
        y_train_ = y_train[train_indexes_, :]
        y_tune_ = y_train[tune_indexes_, :]

        # Run ridge regression using this test set
        correlation_matrix_ = _ridge_correlations_per_voxel_per_alpha(
            X_train_, X_tune_, y_train_, y_tune_, alphas,
            corrmin=corrmin, singcutoff=singcutoff,
            use_corr=use_corr,
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

            # decrease inds by one clipped at max index
            if decrease_alpha > 0:
                print('decrease alpha mean', np.mean(bestalphainds))
                bestalphainds = np.clip(
                    bestalphainds - decrease_alpha, 0, len(alphas)-1)
                print('decrease alpha mean', np.mean(bestalphainds))
            valphas = alphas[bestalphainds]

        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((n_targets,))
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

        valphas = np.array([bestalpha]*n_targets)
        logger.debug("Best alpha = %0.3f" % bestalpha)

    if return_wt:
        # Find weights
        logger.debug(
            "Computing weights for each response using entire training set..")
        # wt = _ridge_sklearn(X_train, y_train, valphas)
        wt = _ridge(X_train, y_train, valphas, singcutoff=singcutoff)

        # Predict responses on prediction set
        logger.debug("Predicting responses for predictions set..")
        pred = np.dot(X_test, wt)

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        if use_corr:
            corrs_test = np.nan_to_num(np.array([np.corrcoef(y_test[:, ii], nnpred[:, ii].ravel())[0, 1]
                                                 for ii in range(y_test.shape[1])]))
        else:
            residual_variance = (y_test-pred).var(0)
            residual_sum_of_squares = 1 - \
                (residual_variance / y_test.var(0))
            corrs_test = np.sqrt(np.abs(residual_sum_of_squares)) * \
                np.sign(residual_sum_of_squares)

        return wt, corrs_test, valphas, all_correlation_matrices, valinds
    else:
        # get correlations for prediction dataset directly
        corrs_test = _ridge_correlations_per_voxel(
            X_train, X_test, y_train, X_test, valphas,
            use_corr=use_corr, logger=logger, singcutoff=singcutoff)

        return [], corrs_test, valphas, all_correlation_matrices, valinds


def lowrank_ridge(X, Y, alpha, r):
    """
    Perform ridge regression with many inputs and outputs using a rank-r approximation.

    Parameters:
    X : numpy.ndarray
        Input features matrix of shape (n_samples, n_features).
    Y : numpy.ndarray
        Output targets matrix of shape (n_samples, n_outputs).
    alpha : float
        Regularization parameter (alphaa).
    r : int
        Rank for the truncated SVD.

    Returns:
    B : numpy.ndarray
        Coefficient matrix of shape (n_features, n_outputs).
    """
    # Step 1: Compute truncated SVD of X
    U_r, Sigma_r, V_r_T = randomized_svd(X, n_components=r)

    # Step 2: Compute T = U_r^T Y
    T = U_r.T @ Y  # Shape: (r, n_outputs)

    # Step 3: Compute D = (Σ_r^2 + λ I_r)^{-1} Σ_r
    denom = Sigma_r ** 2 + alpha
    D = Sigma_r / denom  # Shape: (r,)

    # Step 4: Compute B ≈ V_r D T
    DT = D[:, np.newaxis] * T  # Element-wise multiplication
    B = V_r_T.T @ DT  # Shape: (n_features, n_outputs)

    return B


def bootstrap_low_rank_ridge(
        X_train, y_train, alphas, ranks, nboots, chunklen, nchunks,
        logger=ridge_logger):

    n_train, n_targets = y_train.shape
    splits = _gen_temporal_chunk_splits(
        nboots, n_train, chunklen, nchunks)
    valinds = [splits[1] for splits in splits]

    correlation_matrices = []
    for idx_bootstrap in _counter(range(nboots), countevery=1, total=nboots):
        logger.debug("Selecting held-out test set..")

        # get indices for training / testing
        train_indexes_, tune_indexes_ = splits[idx_bootstrap]

        # Select data
        X_train_ = X_train[train_indexes_, :]
        X_tune_ = X_train[tune_indexes_, :]
        y_train_ = y_train[train_indexes_, :]
        y_tune_ = y_train[tune_indexes_, :]

        # Run ridge regression using this test set
        t0 = time.time()
        correlation_matrix = np.zeros((len(ranks), len(alphas), n_targets))
        for i, rank in enumerate(ranks):
            for j, alpha in enumerate(tqdm(alphas)):
                wt = lowrank_ridge(X_train_, y_train_, alpha, rank)
                pred_tune = X_tune_ @ wt
                correlation_matrix[i, j] = np.array([np.corrcoef(y_tune_[:, ii], pred_tune[:, ii].ravel())[0, 1]
                                                     for ii in range(y_tune_.shape[1])])

        correlation_matrices.append(correlation_matrix.reshape(-1, n_targets))

    # Find best settings for each voxel
    all_correlation_matrices = np.dstack(correlation_matrices)
    meanbootcorrs = all_correlation_matrices.mean(2)
    best_indexes = np.argmax(meanbootcorrs, 0)

    # Fit full model for everything
    wts_full = []
    for i, rank in enumerate(ranks):
        for j, alpha in enumerate(tqdm(alphas)):
            wts_full.append(lowrank_ridge(
                X_train, y_train, alpha, rank))

    # select best weight per voxel
    wts_final = np.zeros_like(wts_full[0])
    for i in tqdm(range(n_targets)):
        wts_final[:, i] = wts_full[best_indexes[i]][:, i]
    logger.debug(f"\ttime elapsed: {time.time()-t0}")

    return wts_final, meanbootcorrs


def boostrap_ridge_with_lowrank(
        X_train, y_train, X_test, y_test, alphas_ridge, alphas_lowrank,
        ranks, nboots, chunklen, nchunks,
        singcutoff=1e-10, single_alpha=False, logger=ridge_logger
):
    wt_ridge, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
        X_train, y_train, X_test, y_test,
        alphas=alphas_ridge,
        nboots=nboots,
        chunklen=chunklen, nchunks=nchunks,
        singcutoff=singcutoff, single_alpha=single_alpha, logger=logger)

    # pred_test = X_test @ wt_ridge
    # corrs_test = np.array([np.corrcoef(y_test[:, ii], pred_test[:, ii].ravel())[0, 1]
    #    for ii in range(y_test.shape[1])])
    logger.debug(f'mean test corrs ridge {corrs_test.mean():.5f}')

    wt_lowrank, meanbootcorrs = bootstrap_low_rank_ridge(
        X_train, y_train, alphas=alphas_lowrank, ranks=ranks,
        nboots=nboots, chunklen=chunklen, nchunks=nchunks, logger=logger)
    pred_test = X_test @ wt_lowrank
    corrs_test = np.array([np.corrcoef(y_test[:, ii], pred_test[:, ii].ravel())[0, 1]
                           for ii in range(y_test.shape[1])])
    # print('mean test corrs', corrs_test.mean())
    logger.debug(f'mean test corrs lowrank {corrs_test.mean():.5f}')

    # select best weights based on bootstrap results
    mean_boot_corrs_ridge = corrs_tune.mean(2).max(axis=0)
    mean_boot_corrs_lowrank = meanbootcorrs.max(axis=0)
    wt_hybrid = np.zeros_like(wt_ridge)
    for i in range(y_train.shape[1]):
        if mean_boot_corrs_ridge[i] > mean_boot_corrs_lowrank[i]:
            wt_hybrid[:, i] = wt_ridge[:, i]
        else:
            wt_hybrid[:, i] = wt_lowrank[:, i]

    pred_test = X_test @ wt_hybrid
    corrs_test = np.array([np.corrcoef(y_test[:, ii], pred_test[:, ii].ravel())[0, 1]
                           for ii in range(y_test.shape[1])])
    logger.debug(f'mean test corrs hybrid {corrs_test.mean():.5f}')
    corrs_tune = np.maximum(mean_boot_corrs_ridge, mean_boot_corrs_lowrank)

    return wt_hybrid, corrs_test, corrs_tune, valinds


    ###########################################################
if __name__ == '__main__':
    # sample data for ridge regression
    np.random.seed(0)

    # set logging to debug
    logging.basicConfig(level=logging.DEBUG)

    # params = joblib.load('example_params.joblib')
    params = joblib.load('/home/chansingh/fmri/example_params_full.joblib')
    print(params.keys())

    X_train = params['features_train_delayed']
    y_train = params['resp_train']
    X_test = params['features_test_delayed']
    y_test = params['resp_test']
    alphas = params['alphas']
    # nboots=params['nboots'],
    nboots = 10
    chunklen = params['chunklen']
    nchunks = params['nchunks']
    singcutoff = params['singcutoff']
    single_alpha = params['single_alpha']

    # wt_hybrid, corrs_test, corrs_tune, valinds = boostrap_ridge_with_lowrank(
    #     X_train, y_train, X_test, y_test,
    #     alphas_ridge=alphas,
    #     alphas_lowrank=alphas,
    #     ranks=[100],
    #     nboots=nboots, chunklen=chunklen, nchunks=nchunks)

    print('alphas', alphas)

    # baseline call (with decrease_alpha=1)
    t0 = time.time()
    wt_ridge, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
        X_train, y_train, X_test, y_test,
        alphas=alphas,
        nboots=nboots,
        chunklen=params['chunklen'], nchunks=params['nchunks'],
        singcutoff=params['singcutoff'], single_alpha=params['single_alpha'],
        decrease_alpha=1
    )
    print('time elapsed', time.time()-t0)

    pred_test = X_test @ wt_ridge
    corrs_test = np.array([np.corrcoef(y_test[:, ii], pred_test[:, ii].ravel())[0, 1]
                           for ii in range(y_test.shape[1])])
    print('mean test corrs', corrs_test.mean())
    pred_train = X_train @ wt_ridge
    corrs_train = np.array([np.corrcoef(y_train[:, ii], pred_train[:, ii].ravel())[0, 1]
                            for ii in range(y_train.shape[1])])
    print('mean train corrs', corrs_train.mean())

    # # call 2
    # t0 = time.time()
    # wt_lowrank, meanbootcorrs = bootstrap_low_rank_ridge(
    #     X_train, y_train, alphas=alphas[::2], ranks=[25, 100], nboots=nboots, chunklen=chunklen, nchunks=nchunks)
    # print('time elapsed', time.time()-t0)
    # pred_train = X_train @ wt_lowrank

    # pred_test = X_test @ wt_lowrank
    # corrs_test = np.array([np.corrcoef(y_test[:, ii], pred_test[:, ii].ravel())[0, 1]
    #                        for ii in range(y_test.shape[1])])
    # print('mean test corrs', corrs_test.mean())
    # pred_train = X_train @ wt_lowrank
    # corrs_train = np.array([np.corrcoef(y_train[:, ii], pred_train[:, ii].ravel())[0, 1]
    #                         for ii in range(y_train.shape[1])])
    # print('mean train corrs', corrs_train.mean())

    # # select weights between wt and wt_lowrank based on bootstrap results
    # try:
    #     meanbootcorrs_ridge = corrs_tune.mean(2).max(axis=0)
    #     meanbootcorrs_lowrank = meanbootcorrs.max(axis=0)
    #     wt_hybrid = np.zeros_like(wt_ridge)
    #     for i in range(y_train.shape[1]):
    #         if meanbootcorrs_ridge[i] > meanbootcorrs_lowrank[i]:
    #             wt_hybrid[:, i] = wt_ridge[:, i]
    #         else:
    #             wt_hybrid[:, i] = wt_lowrank[:, i]

    #     pred_test = X_test @ wt_hybrid
    #     corrs_test = np.array([np.corrcoef(y_test[:, ii], pred_test[:, ii].ravel())[0, 1]
    #                            for ii in range(y_test.shape[1])])
    #     print('mean test corrs', corrs_test.mean())
    #     pred_train = X_train @ wt_hybrid
    #     corrs_train = np.array([np.corrcoef(y_train[:, ii], pred_train[:, ii].ravel())[0, 1]
    #                             for ii in range(y_train.shape[1])])
    #     print('mean train corrs', corrs_train.mean())
    # except Exception as e:
    #     print(e)
    #     breakpoint()
