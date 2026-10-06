"""A one-time notice for models whose search is compiled with numba.

The first fit of such a model on a new machine compiles its search, which takes from seconds to
minutes, and the result is cached on disk, so later processes start in about a second. Without a
word that first fit looks hung; ``notify_first_compile`` says what is happening, once per process
and only when no compiled code is cached yet.
"""

import glob
import os
import pickle
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


def _numba_version():
    numba = sys.modules.get('numba')
    if numba is not None:
        return numba.__version__
    try:
        from importlib.metadata import version
        return version('numba')
    except Exception:
        return None


def _index_is_fresh(path, stamp, numba_version):
    """Whether the numba index file ``path`` was written by this numba for the source file as it is now.

    numba's index is a pickled numba version followed by a pickled ``((st_mtime, st_size) of the source,
    overloads)``; numba ignores an index whose version or stamp differs, and so does this."""
    try:
        with open(path, 'rb') as f:
            if pickle.load(f) != numba_version:
                return False
            saved_stamp = pickle.loads(f.read())[0]
        return tuple(saved_stamp) == stamp
    except Exception:  # unreadable or in another format: say not cached, so the notice is shown
        return False


def is_cached(module_file):
    """Whether numba has compiled code on disk for the kernels of ``module_file`` that this Python and numba can
    load: an index for this Python version (``<module>.<kernel>-<line>.py<major><minor>.nbi``) written by the
    installed numba for the current source file (same modification time and size). When unsure, False."""
    stem = os.path.splitext(os.path.basename(module_file))[0]
    tag = f'py{sys.version_info.major}{sys.version_info.minor}'
    try:
        st = os.stat(module_file)
    except OSError:
        return False
    stamp = (st.st_mtime, st.st_size)
    numba_version = _numba_version()
    if numba_version is None:
        return False
    for d in _cache_dirs(module_file):
        for path in glob.glob(os.path.join(glob.escape(d), f'{glob.escape(stem)}.*.{tag}.nbi')):
            if _index_is_fresh(path, stamp, numba_version):
                return True
    return False


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
