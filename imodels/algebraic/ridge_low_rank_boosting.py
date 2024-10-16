from sklearn.linear_model import Ridge
import time
import numpy as np
from tqdm import tqdm
import logging
import random
import joblib
import itertools as itools
from imodels.algebraic.ridge_multi import _ridge_correlations_per_voxel_per_alpha, _ridge
from sklearn.utils.extmath import randomized_svd


ridge_logger = logging.getLogger("ridge_corr")
def _z_score(v): return (v-v.mean(0))/v.std(0)  # z-score function


def _gen_temporal_chunk_splits(num_splits: int, num_examples: int, chunk_len: int, num_chunks: int):
    all_indexes = range(num_examples)
    index_chunks = list(zip(*[iter(all_indexes)] * chunk_len))
    splits_list = []
    for _ in range(num_splits):
        random.shuffle(index_chunks)
        tune_indexes_ = list(itools.chain(*index_chunks[:num_chunks]))
        train_indexes_ = list(set(all_indexes)-set(tune_indexes_))
        splits_list.append((train_indexes_, tune_indexes_))
    return splits_list


def low_rank_ridge_regression(X, Y, alpha, r):
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


def bootstrap_lowrank_ridge(
        X_train, y_train, X_test, y_test, alphas, ranks, nboots, chunklen, nchunks,
        corrmin=0.2, joined=None, singcutoff=1e-10, single_alpha=False,
        use_corr=True, return_wt=True, logger=ridge_logger):

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
                wt = low_rank_ridge_regression(X_train_, y_train_, alpha, rank)
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
            wts_full.append(low_rank_ridge_regression(
                X_train, y_train, alpha, rank))

    wts_final = np.zeros_like(wts_full[0])

    for i, wt in enumerate(wts_full):
        # wts_final += wt * (best_indexes == i)[:, np.newaxis]
        wts_final[]

    logger.debug(f"\ttime elapsed: {time.time()-t0}")

    return wts_final, meanbootcorrs


def _counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    # Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)

    for count, thing in enumerate(iterable):
        yield thing

        if not count % countevery:
            current_time = time.time()
            rate = float(count+1)/(current_time-start_time)

            if rate > 1:  # more than 1 item/second
                ratestr = "%0.2f items/second" % rate
            else:  # less than 1 item/second
                ratestr = "%0.2f seconds/item" % (rate**-1)

            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime(
                    '%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d" % (count+1, total)
            else:
                timestr = ""
                itemstr = "%d" % (count+1)

            formatted_str = "%s items complete (%s%s)" % (
                itemstr, ratestr, timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)


###########################################################
if __name__ == '__main__':
    # sample data for ridge regression
    np.random.seed(0)

    # set logging to debug
    logging.basicConfig(level=logging.DEBUG)

    params = joblib.load('example_params.joblib')
    print(params.keys())

    X_train = params['features_train_delayed']
    y_train = params['resp_train']
    X_test = params['features_test_delayed']
    y_test = params['resp_test']
    alphas = params['alphas']
    # nboots=params['nboots'],
    nboots = 2
    chunklen = params['chunklen']
    nchunks = params['nchunks']
    singcutoff = params['singcutoff']
    single_alpha = params['single_alpha']

    n_train, n_targets = y_train.shape
    splits = _gen_temporal_chunk_splits(
        nboots, n_train, chunklen, nchunks)
    valinds = [splits[1] for splits in splits]

    # get indices for training / testing
    train_indexes_, tune_indexes_ = splits[0]

    # Select data
    X_train_ = X_train[train_indexes_, :]
    X_tune_ = X_train[tune_indexes_, :]
    y_train_ = y_train[train_indexes_, :]
    y_tune_ = y_train[tune_indexes_, :]

    # run baseline
    # t0 = time.time()
    # correlation_matrix_ = _ridge_correlations_per_voxel_per_alpha(
    # X_train_, X_tune_, y_train_, y_tune_, alphas,
    # singcutoff=singcutoff)

    # print('\tRunning', algo)
    t0 = time.time()

    # wt = low_rank_ridge_regression(X_train_, y_train_, alphas[3], 400)
    # wt = low_rank_ridge_regression(X_train, y_train, alphas[3], 15)
    # wt = _ridge(X_train_, y_train_, alphas[4])
    # pred_test = X_test @ wt

    # wt = _ridge_sklearn(params['features_train_delayed'],
    #                     params['resp_train'], alpha)

    # pred_train = X_train @ wt
    wts_final, meanbootcorrs = bootstrap_lowrank_ridge(
        X_train, y_train, X_test, y_test, alphas[::3], [15], nboots, chunklen, nchunks)
    pred_train = X_train @ wts_final

    corrs_train = np.array([np.corrcoef(params['resp_train'][:, ii], pred_train[:, ii].ravel())[0, 1]
                            for ii in range(params['resp_train'].shape[1])])
    print('\tmean train corr', corrs_train.mean())

    pred_test = X_test @ wts_final
    corrs_test = np.array([np.corrcoef(params['resp_test'][:, ii], pred_test[:, ii].ravel())[0, 1]
                           for ii in range(params['resp_test'].shape[1])])
    print('\tmean test corr', corrs_test.mean())
    print('\ttime elapsed', time.time()-t0)

    # print('\tmean train corr', corrs_train.mean())
