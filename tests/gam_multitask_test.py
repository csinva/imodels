from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV, LassoCV, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import _check_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from collections import defaultdict
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

import imodels
from sklearn.base import RegressorMixin, ClassifierMixin
import os
import os.path

path_to_tests = os.path.dirname(os.path.realpath(__file__))


def single_output_self_supervised():
    X, y, feature_names = imodels.get_clean_dataset("california_housing")
    # X, y, feature_names = imodels.get_clean_dataset("bike_sharing")

    # remove some features to speed things up
    X = X[:10, :4]

    # remove some outcomes to speed things up
    y = y[:10]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # unit test
    gam = MultiTaskGAMRegressor(multitask=False)
    gam.fit(X, y_train)
    gam2 = MultiTaskGAMRegressor(multitask=True)
    gam2.fit(X, y_train)
    preds_orig = gam.predict(X_test)
    assert np.allclose(preds_orig, gam2.ebms_[-1].predict(X_test))

    # extracted curves + intercept should sum to original predictions
    feats_extracted = gam2._extract_ebm_features(X_test)

    # get features for ebm that predicts target
    feats_extracted_target = feats_extracted[:,
                                             -len(gam2.term_names_list_[-1]):]
    # assert feats_extracted_target.shape == (num_samples, num_features)
    preds_extracted_target = np.sum(feats_extracted_target, axis=1) + \
        gam2.ebms_[-1].intercept_
    diff = preds_extracted_target - preds_orig
    assert np.allclose(preds_extracted_target, preds_orig), diff
    print('Single-output tests pass successfully')


def classification():
    X, y, feature_names = imodels.get_clean_dataset("heart")
    # X, y, feature_names = imodels.get_clean_dataset("bike_sharing")

    # remove some features to speed things up
    X = X[:30, :4]

    # remove some outcomes to speed things up
    y = y[:30]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # multi-task
    gam2 = MultiTaskGAMClassifier(multitask=True)
    gam2.fit(X, y_train)
    print('\tmultitask roc', roc_auc_score(
        y_test, gam2.predict_proba(X_test)[:, 1]))

    # non-multitask
    gam = MultiTaskGAMClassifier(multitask=False)
    gam.fit(X, y_train)
    preds = gam.predict(X_test)
    preds_proba = gam.predict_proba(X_test)
    assert preds.size == y_test.size, "predict() yields right size"
    assert preds_proba.shape[1] == 2, "preds_proba has 2 columns"
    assert np.max(preds_proba) < 1.1, "preds_proba has no values over 1"
    print('\tSingle-task roc', roc_auc_score(y_test, preds_proba[:, 1]))
    print('Classification tests passed')


def multi_output():
    X, y, feature_names = imodels.get_clean_dataset("water-quality_multitask")

    # remove some features to speed things up
    X = X[:10, :4]
    y = y[:10]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print('\tshapes', X.shape, y_train.shape, X_test.shape, y_test.shape)

    gam_mt = MultiTaskGAMRegressor(multitask=True)
    gam_mt.fit(X, y_train)
    print('\tmultitask r2_test', gam_mt.score(X_test, y_test))

    gam = MultiTaskGAMRegressor(multitask=False)
    gam.fit(X, y_train)
    print('\tsingle-task r2_test', gam.score(X_test, y_test))
    print('Multi-output tests passed')


def multi_output_classification():
    X, y, feature_names = imodels.get_clean_dataset("water-quality_multitask")

    def _roc_no_error(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0

    # remove some features to speed things up
    X = X[:30, :2]
    y = y[:30]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print('\tshapes', X.shape, y_train.shape, X_test.shape, y_test.shape)

    gam_mt = MultiTaskGAMClassifier(multitask=True)
    gam_mt.fit(X, y_train)
    preds = gam_mt.predict(X_test)
    preds_proba = gam_mt.predict_proba(X_test)
    preds_proba = np.vstack([p[:, 1] for p in preds_proba]).T
    acc = np.mean(preds == y_test)
    rocs = [_roc_no_error(y_test[:, i], preds_proba[:, i])
            for i in range(y_test.shape[1])]
    roc = np.mean(rocs)
    print('\tmultitask acc', acc)
    print('\tmultitask roc', roc)

    gam = MultiTaskGAMClassifier(multitask=False)
    gam.fit(X, y_train)
    preds = gam.predict(X_test)
    preds_proba = gam.predict_proba(X_test)
    preds_proba = np.vstack([p[:, 1] for p in preds_proba]).T
    acc = np.mean(preds == y_test)
    rocs = [_roc_no_error(y_test[:, i], preds_proba[:, i])
            for i in range(y_test.shape[1])]
    roc = np.mean(rocs)
    print('\tsingle-task acc', acc)
    print('\tsingle-task roc', roc)

    print('Multi-output classification tests passed')


def compare_models():
    # X, y, feature_names = imodels.get_clean_dataset("heart")
    X, y, feature_names = imodels.get_clean_dataset("bike_sharing")
    # X, y, feature_names = imodels.get_clean_dataset("water-quality_multitask")
    # X, y, feature_names = imodels.get_clean_dataset("diabetes")

    # remove some features to speed things up
    X = X[:, :5]
    X = X[:50]
    y = y[:50]
    X, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    results = defaultdict(list)
    for gam in tqdm([
        # MultiTaskGAMRegressor(use_correlation_screening_for_features=True),
        MultiTaskGAMRegressor(
            use_single_task_with_reweighting=True, fit_linear_frac=0.5),
        MultiTaskGAMRegressor(),
        MultiTaskGAMRegressor(fit_linear_frac=0.5),
        # MultiTaskGAMRegressor(fit_target_curves=False),
        # AdaBoostRegressor(
            # estimator=MultiTaskGAMRegressor(
        # ebm_kwargs={'max_rounds': 50}),
            # n_estimators=8),
            # AdaBoostRegressor(estimator=MultiTaskGAMRegressor(
        # multitask=True), n_estimators=2),
        # MultiTaskGAMRegressor(multitask=True, onehot_prior=True),
        # MultiTaskGAMRegressor(multitask=True, onehot_prior=False),
        # MultiTaskGAMRegressor(multitask=True, renormalize_features=True),
        # MultiTaskGAMRegressor(multitask=True, renormalize_features=False),
        # MultiTaskGAMRegressor(multitask=True, use_internal_classifiers=True),
        # ExplainableBoostingRegressor(n_jobs=1, interactions=0)
    ]):
        np.random.seed(42)
        results["model_name"].append(gam)
        print('Fitting', results['model_name'][-1])
        gam.fit(X, y_train)
        results['test_corr'].append(np.corrcoef(
            y_test, gam.predict(X_test))[0, 1].round(3))
        results['test_r2'].append(gam.score(X_test, y_test).round(3))
        if hasattr(gam, 'lin_model'):
            print('lin model coef', gam.lin_model.coef_)
        print(results)

    # don't round strings
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(pd.DataFrame(results).round(3))


if __name__ == '__main__':
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    from imodels.algebraic.gam_multitask import MultiTaskGAMRegressor, MultiTaskGAMClassifier
    # multi_output_classification()
    # classification()
    # single_output_self_supervised()
    # multi_output()
    compare_models()
