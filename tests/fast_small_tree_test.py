"""Checks for FastSmallTreeClassifier, whose selling point is that it is exact.

The important test here is `test_matches_brute_force`: the model claims the tree
it returns is the best one in existence for its objective, so a test that only
checked accuracy would not be testing the claim. These enumerate every tree on a
small problem and compare.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, export_text

from imodels import FastSmallTreeClassifier
from imodels.tree.optimal_tree.solver import TreeClassifier
from imodels.util.arguments import decode_labels


def brute_force_optimum(Xb, y, lam):
    """Risk of the best tree on binary features Xb, by exhaustive enumeration.

    The same objective the model minimises: misclassification rate over all n
    rows, plus lam per leaf. Memoised on the set of rows reaching a node, which
    is what makes enumerating every tree tractable at this size.
    """
    n, m = Xb.shape
    memo = {}

    def best(rows):
        key = frozenset(rows)
        if key in memo:
            return memo[key]
        labels = y[list(rows)]
        _, counts = np.unique(labels, return_counts=True)
        value = (len(rows) - counts.max()) / n + lam  # keep it as one leaf
        for j in range(m):
            left = [i for i in rows if Xb[i, j] == 1]
            right = [i for i in rows if Xb[i, j] == 0]
            if not left or not right:  # not a tree, just the same node again
                continue
            value = min(value, best(tuple(left)) + best(tuple(right)))
        memo[key] = value
        return value

    return best(tuple(range(n)))


def objective_of(model, X, y, lam):
    """Score any fitted classifier on the model's objective, for comparison."""
    preds = np.asarray(model.predict(X))
    n_leaves = (model.get_n_leaves() if hasattr(model, "get_n_leaves")
                else model.n_leaves_)
    return float(np.mean(preds != np.asarray(y))) + lam * n_leaves


@pytest.fixture
def binary_data():
    """Every combination of 4 binary features, labelled by XOR of the first two.

    XOR is the textbook case for an exact solver: neither of the two features
    that matter looks useful on its own, so the first split a greedy tree makes
    is a coin toss. The label is balanced, so leaves have to earn their keep.
    """
    X = np.array(list(itertools.product([0, 1], repeat=4)))
    y = (X[:, 0] ^ X[:, 1]).astype(int)
    return X, y


class TestOptimality:
    @pytest.mark.parametrize("lam", [0.02, 0.05, 0.15])
    def test_matches_brute_force(self, binary_data, lam):
        """The certified objective equals the best of every tree that exists."""
        X, y = binary_data
        model = FastSmallTreeClassifier(regularization=lam, time_limit=60).fit(X, y)
        assert model.optimal_, "search did not certify optimality on 16x4 data"
        assert model.objective_ == pytest.approx(brute_force_optimum(X, y, lam))

    def test_bounds_meet_when_certified(self, binary_data):
        """A certified run closes its interval on the optimum."""
        X, y = binary_data
        model = FastSmallTreeClassifier(regularization=0.05, time_limit=60).fit(X, y)
        assert model.lowerbound_ == pytest.approx(model.upperbound_)
        assert model.objective_ == pytest.approx(model.upperbound_)

    def test_at_least_as_good_as_greedy(self, binary_data):
        """Greedy trees are a lower bar by construction; pin that they are."""
        X, y = binary_data
        lam = 0.05
        model = FastSmallTreeClassifier(regularization=lam, time_limit=60).fit(X, y)
        greedy = DecisionTreeClassifier(random_state=0).fit(X, y)
        assert model.objective_ <= objective_of(greedy, X, y, lam) + 1e-12

    def test_reported_objective_matches_the_returned_tree(self, binary_data):
        """objective_ describes the tree handed back, not an internal bound."""
        X, y = binary_data
        lam = 0.05
        model = FastSmallTreeClassifier(regularization=lam, time_limit=60).fit(X, y)
        assert model.objective_ == pytest.approx(objective_of(model, X, y, lam))


class TestRegularization:
    def test_larger_penalty_never_grows_the_tree(self, binary_data):
        """Leaves cost more, so the optimal tree cannot gain any."""
        X, y = binary_data
        sizes = [FastSmallTreeClassifier(regularization=lam, time_limit=60)
                 .fit(X, y).n_leaves_ for lam in (0.01, 0.05, 0.2, 0.5)]
        assert sizes == sorted(sizes, reverse=True), sizes

    def test_heavy_penalty_gives_a_single_leaf(self, binary_data):
        """Priced above any gain, the majority-class stump wins."""
        X, y = binary_data
        model = FastSmallTreeClassifier(regularization=1.0, time_limit=60).fit(X, y)
        assert model.n_leaves_ == 1
        assert model.complexity_ == 0
        assert len(np.unique(model.predict(X))) == 1


class TestApi:
    def test_predict_proba_agrees_with_predict(self, binary_data):
        X, y = binary_data
        model = FastSmallTreeClassifier(regularization=0.05, time_limit=60).fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), len(model.classes_))
        assert np.allclose(proba.sum(axis=1), 1)
        assert (model.classes_[proba.argmax(axis=1)] == model.predict(X)).all()

    def test_multiclass_with_string_labels(self):
        X, y = np.repeat(np.arange(3), 6).reshape(-1, 1), np.repeat(list("abc"), 6)
        model = FastSmallTreeClassifier(regularization=0.02, time_limit=60).fit(X, y)
        assert list(model.classes_) == ["a", "b", "c"]
        assert (model.predict(X) == y).all()
        assert model.predict_proba(X).shape == (18, 3)

    def test_feature_names_reach_the_printed_model(self, binary_data):
        X, y = binary_data
        frame = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
        model = FastSmallTreeClassifier(regularization=0.05, time_limit=60).fit(frame, y)
        printed = str(model)
        assert "alpha" in printed or "beta" in printed
        assert "certified optimal" in printed

    def test_time_limit_reports_uncertified(self):
        """A limit that cannot be met must say so rather than claim optimality.

        Random labels over continuous features is the expensive case: there is
        no structure to prune against, and each column contributes one candidate
        split per distinct value.
        """
        rng = np.random.RandomState(0)
        X = rng.randn(200, 6)
        y = rng.randint(0, 2, size=200)
        model = FastSmallTreeClassifier(regularization=0.005, time_limit=0.3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
        assert not model.optimal_
        assert any("optimality" in str(w.message) for w in caught)
        assert model.lowerbound_ <= model.upperbound_ + 1e-12


class TestSklearnTree:
    """The certified tree is stored as an sklearn tree, which `predict` uses."""

    @staticmethod
    def certified_predict(model, X):
        """Predictions from the solver's own traversal of the certified rules."""
        frame = pd.DataFrame(np.asarray(X, dtype=float), columns=list(model.feature_names_))
        return decode_labels(model, TreeClassifier(model.tree_).predict_fast(frame).astype(int))

    def test_estimator_is_a_fitted_sklearn_tree(self, binary_data):
        X, y = binary_data
        model = FastSmallTreeClassifier(regularization=0.02, time_limit=60).fit(X, y)
        assert isinstance(model.estimator_, DecisionTreeClassifier)
        assert model.estimator_.get_n_leaves() == model.n_leaves_
        assert (model.predict(X) == model.estimator_.predict(X)).all()

    @pytest.mark.parametrize("kwargs", [
        {},
        {"balance": True},
        {"costs": [[0, 1, 4], [2, 0, 1], [1, 3, 0]]},
    ], ids=["default", "balance", "costs"])
    def test_sklearn_tree_routes_like_the_certified_tree(self, kwargs):
        """sklearn predicts argmax of a leaf's value, which must be the certified class.

        Three classes on continuous columns, so every rule is a threshold between
        training values and the leaves' class choice depends on the objective.
        """
        rng = np.random.RandomState(0)
        X = rng.rand(150, 3)
        y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.6).astype(int)
        model = FastSmallTreeClassifier(regularization=0.02, time_limit=60, **kwargs).fit(X, y)
        assert (model.predict(X) == self.certified_predict(model, X)).all()
        X_new = rng.rand(500, 3)
        assert (model.predict(X_new) == self.certified_predict(model, X_new)).all()

    def test_sklearn_tooling_reads_the_feature_names(self, binary_data):
        X, y = binary_data
        frame = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
        model = FastSmallTreeClassifier(regularization=0.02, time_limit=60).fit(frame, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no feature-name warnings from sklearn
            model.predict(frame)
        text = export_text(model.estimator_, feature_names=list(frame.columns))
        assert any(name in text for name in frame.columns)


# ---------------------------------------------------------------------------
# Exactness under conditions the development benchmark never exercised: wider feature
# sets, three classes, cost matrices with nonzero diagonals, and a node store so small
# that it is reallocated many times mid-search (including inside a single-column chain).
# The guarantee rests on the proofs in the documentation; these are its tripwires.
# ---------------------------------------------------------------------------

def brute_force_with_costs(Xb, y, K, C, lam):
    """Exhaustive optimum of sum_i C[pred_i, y_i] + lam * leaves over trees on Xb."""
    n, m = Xb.shape
    memo = {}

    def leaf(rows):
        cnt = np.bincount(y[list(rows)], minlength=K)
        return float(min(C[p] @ cnt for p in range(K)))

    def best(rows):
        key = frozenset(rows)
        if key in memo:
            return memo[key]
        value = leaf(rows) + lam
        for j in range(m):
            left = tuple(i for i in rows if Xb[i, j] == 1)
            right = tuple(i for i in rows if Xb[i, j] == 0)
            if not left or not right:
                continue
            value = min(value, best(left) + best(right))
        memo[key] = value
        return value

    return best(tuple(range(n)))


def objective_with_costs(model, X, y, C, lam):
    preds = model.target_encoder_.transform(model.predict(X))
    return float(sum(C[p, t] for p, t in zip(preds, y))) + lam * model.n_leaves_


@pytest.mark.parametrize("seed", range(12))
def test_exhaustive_wide_binary_and_multiclass(seed):
    """Up to six binary features and three classes; every certificate equals the
    exhaustive optimum and the returned tree scores exactly the certified value."""
    rng = np.random.default_rng(seed)
    n, m, K = int(rng.integers(8, 25)), int(rng.integers(4, 7)), int(rng.integers(2, 4))
    Xb = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
    y = rng.integers(0, K, size=n)
    for lam in (0.005, 0.02, 0.05, 0.12, 0.3):
        model = FastSmallTreeClassifier(regularization=lam).fit(Xb, y)
        assert model.optimal_
        truth = brute_force_optimum(Xb, y, lam)
        assert abs(model.upperbound_ - truth) < 1e-9
        assert abs(objective_of(model, Xb, y, lam) - truth) < 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_exhaustive_cost_matrix_nonzero_diagonal(seed):
    """Random nonnegative cost matrices, a nonzero diagonal included: the certified
    objective is the exhaustive optimum under those costs and the tree achieves it."""
    rng = np.random.default_rng(100 + seed)
    n, m, K = int(rng.integers(8, 21)), int(rng.integers(3, 6)), 3
    Xb = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
    y = rng.integers(0, K, size=n)
    C = rng.uniform(0.0, 1.0, size=(K, K))
    if seed % 2 == 0:
        np.fill_diagonal(C, 0.0)
    for lam in (0.2, 0.7, 1.5):
        model = FastSmallTreeClassifier(regularization=lam, costs=C).fit(Xb, y)
        assert model.optimal_
        truth = brute_force_with_costs(Xb, y, K, C, lam)
        assert abs(model.upperbound_ - truth) < 1e-9
        assert abs(objective_with_costs(model, Xb, y, C, lam) - truth) < 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_tiny_store_reallocated_mid_search(seed, monkeypatch):
    """A node store of capacity 4 is grown dozens of times during one fit, including in
    the middle of a single-column chain; the certificate and the tree must not depend on
    where the growth happened. One numeric column with many thresholds plus binary
    features makes the chain path and the pairwise path both run."""
    from imodels.tree.optimal_tree import solver
    monkeypatch.setattr(solver, "STORE_CAPACITY", 4)
    rng = np.random.default_rng(200 + seed)
    n = int(rng.integers(12, 25))
    X = pd.DataFrame({"x": rng.normal(size=n).round(3), "a": rng.integers(0, 2, size=n),
                      "b": rng.integers(0, 2, size=n)})
    y = ((X["x"] > 0.2) ^ (X["a"] == 1) ^ (rng.random(n) < 0.15)).astype(int).to_numpy()
    for lam in (0.01, 0.04, 0.1):
        model = FastSmallTreeClassifier(regularization=lam).fit(X, y)
        assert model.optimal_
        Xb = model.encoder_.transform(X).astype(np.uint8)
        truth = brute_force_optimum(Xb, y, lam)
        assert abs(model.upperbound_ - truth) < 1e-9
        assert abs(objective_of(model, X, y, lam) - truth) < 1e-9


def test_rejects_inputs_the_proofs_exclude():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    with pytest.raises(ValueError):
        FastSmallTreeClassifier(regularization=-0.01).fit(X, y)
    with pytest.raises(ValueError):
        FastSmallTreeClassifier(regularization=0.05, costs=[[0.0, -1.0], [1.0, 0.0]]).fit(X, y)
    with pytest.raises(ValueError):
        FastSmallTreeClassifier(regularization=0.05, costs=[[0.0, np.inf], [1.0, 0.0]]).fit(X, y)
