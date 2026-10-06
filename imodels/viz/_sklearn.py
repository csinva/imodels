"""Adapters for scikit-learn models beyond single decision trees.

- linear models (regression, logistic / linear classifiers, GLMs) -> AdditiveView of linear terms
- forests (RandomForest, ExtraTrees) -> several trees whose predictions are averaged
- gradient boosting (GradientBoosting, HistGradientBoosting) -> several trees whose leaf values add up
- sums of regression trees (FIGS, exported by imodels.to_sklearn) -> the same view as boosting
- IsotonicRegression -> one shape function
- Pipelines whose earlier steps only scale features are folded into raw units.
Every adapter is checked against the model's own predictions in tests/viz_sklearn_test.py.
"""

import numpy as np

from ._extract import sk_threshold
from ._ir import make_info
from ._views import AdditiveView, Term

SCALERS = ("StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler")
MAX_SHOWN_TREES = 6


# ---------------------------------------------------------------- helpers
def _names(est, X, feature_names, n):
    if feature_names is not None:
        return [str(f) for f in feature_names]
    if X is not None and hasattr(X, "columns"):
        return [str(c) for c in X.columns]
    names = getattr(est, "feature_names_in_", None)
    return [str(c) for c in names] if names is not None else [f"x{i}" for i in range(n)]


def _class_names(classes, class_names):
    if class_names is None:
        return [str(c) for c in classes]
    if isinstance(class_names, dict):
        return [str(class_names.get(c, c)) for c in classes]
    return [str(c) for c in class_names]


def _y_index(y, classes):
    if y is None:
        return None
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[v] for v in np.asarray(y)])


def unwrap_pipeline(model):
    """(final estimator, affine map or None). The map takes raw x to what the estimator sees:
    z = (x - shift) / scale, composed over every scaling step."""
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:  # pragma: no cover
        return model, None
    if not isinstance(model, Pipeline):
        return model, None
    shift = scale = None
    for name, step in model.steps[:-1]:
        if step is None or step == "passthrough":
            continue
        kind = type(step).__name__
        if kind not in SCALERS:
            raise NotImplementedError(
                f"Pipeline step {name!r} ({kind}) changes the features; only scaling steps "
                f"({', '.join(SCALERS)}) can be folded into the drawing. Draw the final estimator on transformed X instead.")
        n = step.n_features_in_
        if kind == "StandardScaler":
            sh = step.mean_ if step.with_mean else np.zeros(n)
            sc = step.scale_ if step.with_std else np.ones(n)
        elif kind == "MinMaxScaler":  # z = x * scale_ + min_  ->  z = (x - (-min_/scale_)) / (1/scale_)
            sh, sc = -step.min_ / step.scale_, 1 / step.scale_
        elif kind == "MaxAbsScaler":
            sh, sc = np.zeros(n), step.scale_
        else:  # RobustScaler
            sh = step.center_ if step.with_centering else np.zeros(n)
            sc = step.scale_ if step.with_scaling else np.ones(n)
        sh, sc = np.asarray(sh, float), np.asarray(sc, float)
        if shift is None:
            shift, scale = sh, sc
        else:  # z2 = (z1 - sh) / sc = (x - shift - sh*scale) / (scale*sc)
            shift, scale = shift + sh * scale, scale * sc
    final = model.steps[-1][1]
    return final, (None if shift is None else (shift, scale))


def _raw_threshold(t, f, affine):
    """A split ``z_f <= t`` on scaled features, as a threshold on raw x_f."""
    if affine is None:
        return t
    shift, scale = affine
    return t * scale[f] + shift[f]


# ---------------------------------------------------------------- linear models
LINK_EXP = ("PoissonRegressor", "GammaRegressor")


def _linear(est, affine, X, y, fn, cn, tn, model_name):
    coef = np.asarray(est.coef_, dtype=float)
    clf = hasattr(est, "classes_")
    if coef.ndim > 1 and coef.shape[0] > 1:
        raise NotImplementedError(f"Multiclass {model_name} is not supported yet (one coefficient row per class).")
    coef = coef.ravel()
    b = float(np.ravel(getattr(est, "intercept_", 0.0))[0]) if np.size(getattr(est, "intercept_", 0.0)) else 0.0
    if affine is not None:  # w . (x - shift) / scale + b  ->  raw coefficients
        shift, scale = affine
        b = b - float(np.sum(coef * shift / scale))
        coef = coef / scale
    names = _names(est, X, fn, len(coef))
    terms = [Term("linear", float(w), feature=j) for j, w in enumerate(coef) if w != 0]
    Xa = None if X is None else np.asarray(X, dtype=float)
    if clf:
        classes = est.classes_
        cls = _class_names(classes, cn)
        has_proba = hasattr(est, "predict_proba") and _proba_is_logistic(est)
        if has_proba:
            kw = dict(link="logistic", score_name="log-odds",
                      note=f"log-odds = intercept + coefficient × feature;  P({cls[-1]}) = sigmoid(log-odds).")
        else:
            kw = dict(link="threshold", threshold=0.0, score_name="decision",
                      note=f"decision = intercept + coefficient × feature;  predicts {cls[-1]} when it is above 0.")
        v = AdditiveView("ruleset", "classification", names, model_name, terms, intercept=b, class_names=cls, **kw)
        return v.attach(Xa, _y_index(y, classes))
    link = "identity"
    kind = type(est).__name__
    if kind in LINK_EXP or (kind == "TweedieRegressor" and _tweedie_log(est)):
        link = "exp"
    note = ("log(prediction) = intercept + coefficient × feature." if link == "exp"
            else "prediction = intercept + coefficient × feature.")
    v = AdditiveView("ruleset", "regression", names, model_name, terms, intercept=b, link=link,
                     target_name=tn or "target", score_name="log prediction" if link == "exp" else (tn or "prediction"),
                     note=note + (" Coefficients are in raw units (the pipeline's scaling is folded in)." if affine else ""))
    return v.attach(Xa, None if y is None else np.asarray(y, float))


def _proba_is_logistic(est):
    """Linear classifiers whose predict_proba is sigmoid(decision_function)."""
    kind = type(est).__name__
    if kind == "LogisticRegression" or kind == "LogisticRegressionCV":
        return True
    if kind == "SGDClassifier":
        return getattr(est, "loss", None) in ("log_loss", "log")
    return False


def _tweedie_log(est):
    link = getattr(est, "link", "auto")
    return link == "log" or (link == "auto" and getattr(est, "power", 0) > 0)


def is_linear(est):
    try:
        from sklearn import linear_model, svm
    except ImportError:  # pragma: no cover
        return False
    mods = (linear_model,)
    kind = type(est).__name__
    if kind in ("LinearSVC", "LinearSVR"):
        return True
    if type(est).__module__.startswith("sklearn.linear_model") and hasattr(est, "coef_"):
        return True
    return kind == "RANSACRegressor"


# ---------------------------------------------------------------- forests and boosting
def _sk_tree_specs(tree, affine, value_fn, base=0):
    """Specs for one fitted sklearn ``Tree`` (or arrays with the same fields), ids offset by ``base``."""
    specs = []
    left, right = tree["children_left"], tree["children_right"]
    nan_left = tree.get("missing_go_to_left")
    for i in range(len(left)):
        sp = dict(n=int(tree["n_node_samples"][i]), impurity=float(tree["impurity"][i]))
        if left[i] >= 0:
            f = int(tree["feature"][i])
            t = float(tree["threshold"][i])
            t = sk_threshold(t) if tree.get("float32", True) else t
            sp.update(feature=f, threshold=float(_raw_threshold(t, f, affine)),
                      left=int(left[i]) + base, right=int(right[i]) + base)
            if nan_left is not None:
                sp["nan_left"] = bool(nan_left[i])
        sp.update(value_fn(i))
        specs.append(sp)
    return specs


def _arrays(t):
    """Field dict for a sklearn Tree object."""
    out = dict(children_left=t.children_left, children_right=t.children_right, feature=t.feature,
               threshold=t.threshold, n_node_samples=t.n_node_samples, impurity=t.impurity, value=t.value)
    mg = getattr(t, "missing_go_to_left", None)
    if mg is not None:
        out["missing_go_to_left"] = np.asarray(mg).astype(bool)
    return out


def _ensemble(trees, *, task, combine, names, model_name, class_names, X, y, link="identity", intercept=0.0,
              target_name=None, max_shown=MAX_SHOWN_TREES):
    """trees: list of (field dict, value_fn)."""
    specs, roots = [], []
    for fields, value_fn in trees:
        roots.append(len(specs))
        specs += _sk_tree_specs(fields, fields.get("affine"), value_fn, base=len(specs))
    info = make_info(specs, task=task, feature_names=names, model_name=model_name, class_names=class_names,
                     X=X, y=y, roots=roots, combine=combine, link=link, intercept=intercept, target_name=target_name)
    for nd, sp in zip(info.nodes, specs):
        nd.value = sp.get("contrib")
        nd.nan_left = sp.get("nan_left")
        if task == "classification" and sp.get("proba") is not None and combine == "mean":
            nd.counts = np.asarray(sp["proba"], float) * max(nd.n, 1)  # the tree's own leaf estimate
    info.shown_roots = roots[:max_shown]
    if combine == "sum" and task == "regression":
        vals = [nd.value for nd in info.nodes if nd.value is not None and nd.is_leaf]
        info.value_range = (min(vals), max(vals)) if vals else info.value_range
    if combine == "sum" and task == "classification":
        vals = [nd.value for nd in info.nodes if nd.value is not None and nd.is_leaf]
        info.value_range = (min(vals), max(vals)) if vals else (0.0, 1.0)
    return info


def _forest(est, affine, X, y, fn, cn, tn, model_name):
    clf = hasattr(est, "classes_")
    if clf and isinstance(est.classes_, list):
        raise NotImplementedError("Multi-output forests are not supported.")
    names = _names(est, X, fn, est.n_features_in_)
    trees = []
    for e in est.estimators_:
        f = _arrays(e.tree_)
        f["affine"] = affine
        if clf:
            vals = f["value"][:, 0, :]
            vf = (lambda v: lambda i: dict(proba=v[i] / (v[i].sum() or 1)))(vals)
        else:
            vals = f["value"][:, 0, 0]
            vf = (lambda v: lambda i: dict(value=float(v[i]), contrib=float(v[i])))(vals)
        trees.append((f, vf))
    classes = est.classes_ if clf else None
    return _ensemble(trees, task="classification" if clf else "regression", combine="mean", names=names,
                     model_name=model_name, class_names=_class_names(classes, cn) if clf else None,
                     X=None if X is None else np.asarray(X, float), y=_y_index(y, classes) if clf else y,
                     target_name=tn)


def _gbm(est, affine, X, y, fn, cn, tn, model_name):
    clf = hasattr(est, "classes_")
    if clf and len(est.classes_) > 2:
        raise NotImplementedError("Multiclass gradient boosting is not supported yet (one tree per class per stage).")
    names = _names(est, X, fn, est.n_features_in_)
    lr = float(est.learning_rate)
    Xa = None if X is None else np.asarray(X, float)
    probe = np.zeros((1, est.n_features_in_))
    init = float(np.ravel(est._raw_predict_init(probe))[0])
    trees = []
    for e in est.estimators_[:, 0]:
        f = _arrays(e.tree_)
        f["affine"] = affine
        vals = f["value"][:, 0, 0] * lr
        trees.append((f, (lambda v: lambda i: dict(value=float(v[i]), contrib=float(v[i])))(vals)))
    classes = est.classes_ if clf else None
    return _ensemble(trees, task="classification" if clf else "regression", combine="sum", names=names,
                     model_name=model_name, class_names=_class_names(classes, cn) if clf else None,
                     X=Xa, y=_y_index(y, classes) if clf else y, link="logistic" if clf else "identity",
                     intercept=init, target_name=tn)


def _hgb(est, affine, X, y, fn, cn, tn, model_name):
    clf = hasattr(est, "classes_")
    if clf and len(est.classes_) > 2:
        raise NotImplementedError("Multiclass histogram gradient boosting is not supported yet.")
    if getattr(est, "is_categorical_", None) is not None and np.any(est.is_categorical_):
        raise NotImplementedError("HistGradientBoosting with categorical features is not supported yet.")
    names = _names(est, X, fn, est.n_features_in_)
    Xa = None if X is None else np.asarray(X, float)
    trees = []
    for preds in est._predictors:
        nodes = preds[0].nodes
        leaf = nodes["is_leaf"].astype(bool)
        f = dict(children_left=np.where(leaf, -1, nodes["left"].astype(np.int64)),  # unsigned in sklearn
                 children_right=np.where(leaf, -1, nodes["right"].astype(np.int64)),
                 feature=nodes["feature_idx"], threshold=nodes["num_threshold"], n_node_samples=nodes["count"],
                 impurity=np.full(len(nodes), np.nan), missing_go_to_left=nodes["missing_go_to_left"].astype(bool),
                 float32=False, affine=affine)
        vals = nodes["value"].astype(float)
        trees.append((f, (lambda v: lambda i: dict(value=float(v[i]), contrib=float(v[i])))(vals)))
    base = float(np.ravel(est._baseline_prediction)[0])
    classes = est.classes_ if clf else None
    return _ensemble(trees, task="classification" if clf else "regression", combine="sum", names=names,
                     model_name=model_name, class_names=_class_names(classes, cn) if clf else None,
                     X=Xa, y=_y_index(y, classes) if clf else y, link="logistic" if clf else "identity",
                     intercept=base, target_name=tn)


def sum_of_trees(trees, X, y, fn, cn, tn, model_name, classes=None):
    """Regression trees whose predictions add up: FIGS, as exported by ``imodels.to_sklearn``.

    Each tree predicts one score per class for a classifier (P = softmax of the summed scores, so for
    two classes P(class 1) = sigmoid of the summed differences) and the prediction for a regressor.
    """
    clf = classes is not None
    if clf and len(classes) != 2:
        raise NotImplementedError(f"Multiclass {model_name} is not supported yet; use a binary or regression model.")
    names = _names(trees[0], X, fn, trees[0].n_features_in_)
    out = []
    for e in trees:
        f = _arrays(e.tree_)
        f["affine"] = None
        v = f["value"][:, :, 0]  # (nodes, outputs)
        vals = v[:, 1] - v[:, 0] if clf else v[:, 0]
        out.append((f, (lambda v: lambda i: dict(value=float(v[i]), contrib=float(v[i])))(vals)))
    return _ensemble(out, task="classification" if clf else "regression", combine="sum", names=names,
                     model_name=model_name, class_names=_class_names(classes, cn) if clf else None,
                     X=None if X is None else np.asarray(X, float), y=_y_index(y, classes) if clf else y,
                     link="logistic" if clf else "identity", target_name=tn)


def _isotonic(est, affine, X, y, fn, cn, tn, model_name):
    if affine is not None:
        raise NotImplementedError("IsotonicRegression inside a scaling pipeline is not supported.")
    xs = np.asarray(est.X_thresholds_, float)
    ys = np.asarray(est.y_thresholds_, float)
    name = (fn[0] if fn is not None else None) or (str(X.columns[0]) if X is not None and hasattr(X, "columns")
                                                    else getattr(X, "name", None) or "x")
    term = Term("curve", feature=0, grid=xs, values=ys)
    if est.out_of_bounds == "nan":
        raise NotImplementedError("IsotonicRegression(out_of_bounds='nan') is not supported.")
    v = AdditiveView("gam", "regression", [name], model_name, [term], intercept=0.0, target_name=tn or "target",
                     score_name=tn or "prediction", note="A monotone step-and-ramp function of one feature.")
    Xa = None if X is None else np.asarray(X, float).reshape(-1, 1)
    return v.attach(Xa, None if y is None else np.asarray(y, float))


FOREST = ("RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor")
GBM = ("GradientBoostingClassifier", "GradientBoostingRegressor")
HGB = ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor")
OPAQUE = ("KNeighborsClassifier", "KNeighborsRegressor", "SVC", "SVR", "NuSVC", "NuSVR", "MLPClassifier",
          "MLPRegressor", "GaussianProcessClassifier", "GaussianProcessRegressor", "KernelRidge")


def adapt(model, X, y, feature_names, class_names, target_name):
    """View for a supported sklearn model (or Pipeline), or None to fall back to single trees."""
    est, affine = unwrap_pipeline(model)
    kind = type(est).__name__
    name = kind if affine is None else f"{kind} (in a pipeline)"
    args = (est, affine, X, y, feature_names, class_names, target_name, name)
    if kind in FOREST:
        return _forest(*args)
    if kind in GBM:
        return _gbm(*args)
    if kind in HGB:
        return _hgb(*args)
    if kind == "IsotonicRegression":
        return _isotonic(*args)
    if kind == "RANSACRegressor":
        return _linear(est.estimator_, affine, X, y, feature_names, class_names, target_name, name)
    if is_linear(est):
        return _linear(*args)
    if kind in OPAQUE:
        raise NotImplementedError(f"{kind} has no readable structure to draw (no rules, trees or coefficients).")
    if hasattr(est, "tree_") and affine is not None:
        return _single_tree_scaled(est, affine, X, y, feature_names, class_names, target_name, name)
    return None


def _single_tree_scaled(est, affine, X, y, fn, cn, tn, name):
    """A decision tree behind scaling steps: thresholds converted to raw units."""
    from ._extract import extract

    info = extract(est, None, None, _names(est, X, fn, est.n_features_in_), cn, tn)
    shift, scale = affine
    for nd in info.nodes:
        if not nd.is_leaf:
            nd.threshold = float(nd.threshold * scale[nd.feature] + shift[nd.feature])
    if X is not None:
        from ._extract import _feature_kind

        Xa = np.asarray(X, float)
        info.X = Xa
        info.route(Xa)
        info.feature_kind = {f: _feature_kind(Xa[:, f]) for f in {nd.feature for nd in info.nodes if not nd.is_leaf}}
        if y is not None:
            info.y = _y_index(y, est.classes_) if info.is_clf else np.asarray(y, float)
    info.model_name = name
    return info
