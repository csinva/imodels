"""Fast-and-frugal trees (Phillips, Neth, Woike & Gaissmaier 2017).

A fast-and-frugal tree asks one question per level and, at every level, is able
to stop and decide: each cue has an *exit* to one class, and whatever it doesn't
catch falls through to the next cue. The last cue exits both ways, so the tree
always decides within `max_depth` questions.

That single descending path is why this lives with the rule lists rather than
the trees, and it is what makes these models usable from memory -- they are
widely used for triage decisions in medicine.

References
----------
Phillips, Neth, Woike & Gaissmaier (2017), "FFTrees: A toolbox to create,
visualize, and evaluate fast-and-frugal decision trees",
Judgment and Decision Making 12(4). http://journal.sjdm.org/17/17217/jdm17217.html
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

from imodels.rule_list.rule_list import RuleList
from imodels.util.arguments import (check_binary_target, check_fit_arguments,
                                    check_predict_X, decode_labels)


class FastFrugalTreeClassifier(BaseEstimator, RuleList, ClassifierMixin):
    """A fast-and-frugal tree: one cue per level, each with an exit.

    Parameters
    ----------
    max_depth : int, default=4
        Number of cues the tree may ask. Fast-and-frugal trees are deliberately
        shallow; the literature typically uses 4 or fewer.
    max_thresholds : int, default=64
        Candidate thresholds considered per feature. Thresholds are taken from
        quantiles of the feature, so this bounds the search on large datasets.
    min_samples_exit : int, default=1
        A cue must send at least this many samples to its exit to be used.

    Attributes
    ----------
    rules_ : list of dict
        One entry per cue, in the order they are asked, plus a final catch-all.
        Uses the same representation as the other rule lists, so `get_rules()`
        works on it.
    classes_ : ndarray
    """

    def __init__(self, max_depth: int = 4, max_thresholds: int = 64,
                 min_samples_exit: int = 1):
        self.max_depth = max_depth
        self.max_thresholds = max_thresholds
        self.min_samples_exit = min_samples_exit

    def fit(self, X, y, feature_names=None):
        check_binary_target(self, y)
        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)

        remaining = np.ones(X.shape[0], dtype=bool)
        self.rules_ = []

        for depth in range(self.max_depth):
            # the last cue has to decide both ways, so stop asking questions
            if depth == self.max_depth - 1 or remaining.sum() == 0:
                break
            cue = self._best_cue(X, y, remaining, feature_names)
            if cue is None:  # no cue earns its place
                break
            cue['depth'] = depth
            self.rules_.append(cue)
            remaining = remaining & ~cue.pop('_exits')

        self.rules_.append(self._final_exit(y, remaining))
        self.complexity_ = len(self.rules_)
        return self

    def _candidate_thresholds(self, column):
        values = np.unique(column)
        if len(values) <= self.max_thresholds:
            # midpoints between observed values
            return (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
        quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
        return np.unique(np.quantile(column, quantiles))

    def _best_cue(self, X, y, remaining, feature_names):
        """The cue whose exit gets the most samples right, net of its mistakes."""
        y_remaining = y[remaining]
        best, best_score = None, 0

        for feature in range(X.shape[1]):
            column = X[remaining, feature]
            for threshold in self._candidate_thresholds(column):
                above = column > threshold
                for exit_above in (True, False):
                    exits = above if exit_above else ~above
                    n_exits = int(exits.sum())
                    if n_exits < self.min_samples_exit:
                        continue
                    exited_y = y_remaining[exits]
                    for exit_class in (0, 1):
                        correct = int((exited_y == exit_class).sum())
                        # reward correct exits, penalize wrong ones
                        score = 2 * correct - n_exits
                        if score > best_score:
                            full = np.zeros(len(y), dtype=bool)
                            full[np.flatnonzero(remaining)[exits]] = True
                            best_score = score
                            best = {
                                'col': str(feature_names[feature]),
                                'index_col': feature,
                                'cutoff': threshold,
                                # 'flip' marks the exit as the <= side, matching
                                # how the other rule lists record their direction
                                'flip': not exit_above,
                                'val_right': float(exit_class),
                                'num_pts': int(remaining.sum()),
                                'num_pts_right': n_exits,
                                '_exits': full,
                            }
        return best

    def _final_exit(self, y, remaining):
        """Everything still undecided takes the majority class of what's left."""
        if remaining.sum() == 0:
            value = float(np.round(y.mean())) if len(y) else 0.0
            return {'val': value, 'num_pts': 0}
        return {'val': float(np.round(y[remaining].mean())),
                'num_pts': int(remaining.sum())}

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_predict_X(self, check_array(X))

        probs = np.zeros(X.shape[0])
        undecided = np.ones(X.shape[0], dtype=bool)
        for rule in self.rules_:
            if 'col' not in rule:  # the final catch-all
                probs[undecided] = rule['val']
                break
            above = X[:, rule['index_col']] > rule['cutoff']
            exits = (~above if rule['flip'] else above) & undecided
            probs[exits] = rule['val_right']
            undecided = undecided & ~exits

        return np.vstack((1 - probs, probs)).transpose()

    def predict(self, X):
        return decode_labels(self, np.argmax(self.predict_proba(X), axis=1))

    def __str__(self):
        if not hasattr(self, 'rules_'):
            return f'{type(self).__name__}(max_depth={self.max_depth})'
        s = '> ------------------------------\n'
        s += '> Fast-and-frugal tree\n'
        s += '> \tEach cue either decides, or passes the case to the next cue\n'
        s += '> ------------------------------\n'
        for rule in self.rules_:
            if 'col' not in rule:
                s += (f"> else | predict {int(rule['val'])} "
                      f"({rule['num_pts']} obs)\n")
                continue
            comparison = '<=' if rule['flip'] else '>'
            s += (f"> if {rule['col']} {comparison} {rule['cutoff']:.3g} | "
                  f"predict {int(rule['val_right'])} "
                  f"({rule['num_pts_right']} obs)\n")
        s += '> ------------------------------\n'
        return s
