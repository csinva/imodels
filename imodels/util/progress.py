"""Progress bars for model fitting.

Fitting an interpretable model is often slow enough to want a sign of life --
one bar tick per rule, tree, boosting round, or MCMC iteration. Models here
share the two helpers below so that the bars look and behave the same
everywhere:

- they appear only when the model's `verbose` argument is truthy, so nothing
  changes for the default silent fit
- they are written to stderr, leaving stdout free for the textual output some
  models already print at higher verbosity
- `bar.write(...)` prints a line without tearing the bar, so text and bar can
  be used together

Use `progress_iter` to wrap a loop whose length is known up front, and
`progress_bar` as a context manager for a loop that has to update by hand
(a `while` loop, or one whose total is only known part-way through).
"""

import sys

from tqdm import tqdm

__all__ = ['progress_iter', 'progress_bar']


class _NullBar:
    """Stand-in used when verbose is falsy, so callers need no `if verbose`."""

    def update(self, n=1):
        pass

    def set_description(self, desc=None, refresh=True):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def write(self, s, file=None, end='\n'):
        # matches tqdm.write's default target, so quiet models stay quiet
        # only because verbose gates the call, not because output is dropped
        print(s, file=file if file is not None else sys.stdout, end=end)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _tqdm(iterable=None, total=None, desc=None, leave=False):
    return tqdm(iterable, total=total, desc=desc, leave=leave, file=sys.stderr)


def progress_iter(iterable, verbose=False, desc=None, total=None, leave=False):
    """Wrap `iterable` in a progress bar when `verbose` is truthy.

    Params
    ------
    iterable
        what the fit loop iterates over
    verbose: bool or int
        the model's verbosity; the bar is shown when this is truthy
    desc: str
        label shown to the left of the bar, e.g. 'fitting rules'
    total: int
        number of iterations, when `iterable` has no length
    leave: bool
        whether to keep the finished bar on screen
    """
    if not verbose:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, leave=leave)


def progress_bar(total=None, verbose=False, desc=None, leave=False):
    """A bar to update by hand, for loops that aren't a simple `for`.

    Returns a context manager exposing the part of tqdm's interface models
    here use (`update`, `set_postfix`, `write`, `close`). When `verbose` is
    falsy it is a do-nothing stand-in, so the call site needs no branch.
    """
    if not verbose:
        return _NullBar()
    return _tqdm(total=total, desc=desc, leave=leave)
