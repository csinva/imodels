from sklearn.linear_model import Ridge
import time
import numpy as np
from tqdm import tqdm
import logging
import random
import joblib
import itertools as itools


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


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx
