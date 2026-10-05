"""A one-time notice for models whose search is compiled with numba.

The first fit of such a model on a new machine compiles its search, which takes from seconds to
minutes, and the result is cached on disk, so later processes start in about a second. Without a
word that first fit looks hung; ``notify_first_compile`` says what is happening, once per process
and only when no compiled code is cached yet.
"""

import glob
import os
import sys

__all__ = ['notify_first_compile']

_NOTIFIED = set()


def _cache_dirs(module_file):
    """Where numba keeps the disk cache of the kernels defined in ``module_file``."""
    here = os.path.dirname(os.path.abspath(module_file))
    dirs = [os.path.join(here, '__pycache__')]  # next to the module, when writable
    rel = os.path.splitdrive(here)[1].lstrip(os.sep)
    user = os.environ.get('NUMBA_CACHE_DIR')
    if user:
        dirs.append(os.path.join(user, rel))
    dirs.append(os.path.join(os.path.expanduser('~'), '.cache', 'numba', rel))  # numba's user-wide fallback
    return dirs


def is_cached(module_file):
    """Whether numba has compiled code on disk for the kernels of ``module_file``."""
    stem = os.path.splitext(os.path.basename(module_file))[0]
    return any(glob.glob(os.path.join(d, f'{stem}.*.nbi')) for d in _cache_dirs(module_file))


def notify_first_compile(module_file, model_name, duration, cache_enabled=True, cache_env=None):
    """Print a notice to stderr before the first fit of ``model_name`` in this process compiles its
    numba search, unless the compiled code is already cached on disk.

    Params
    ------
    module_file: str
        ``__file__`` of the module that defines the kernels
    model_name: str
        the estimator's name, e.g. 'FastRiskScoreClassifier'
    duration: str
        how long the compilation takes, e.g. 'about 2 minutes'
    cache_enabled: bool
        whether the kernels are cached on disk
    cache_env: str
        the environment variable that turns the cache off, named in the notice
    """
    if model_name in _NOTIFIED:
        return
    _NOTIFIED.add(model_name)
    if cache_enabled and is_cached(module_file):
        return
    if cache_enabled:
        after = 'The result is cached, so later fits on this machine start in about a second.'
    else:
        off = f' ({cache_env}=0)' if cache_env else ''
        after = f'The disk cache is off{off}, so every new process compiles it again.'
    print(f'{model_name}: compiling its numba search for the first time, which takes {duration}. {after}',
          file=sys.stderr, flush=True)
