import os.path
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, roc_auc_score

from imodels.experimental import FIGSExtRegressor, FIGSExtClassifier
from imodels import FIGSRegressorCV, FIGSClassifierCV, get_clean_dataset
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("Clalit")
PTH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels/art/clalit"
N_REPS = 10
DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    # ("sonar", "sonar", "pmlb"),
    ("heart", "heart", 'imodels'),
    ("breast-cancer", "breast_cancer", 'imodels'),
    ("haberman", "haberman", 'imodels'),
    ("ionosphere", "ionosphere", 'pmlb'),
    ("diabetes", "diabetes", "pmlb"),
    # # #("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # # #("credit-g", "credit_g", 'imodels'), # like german-credit, but more feats
    ("german-credit", "german", "pmlb"),
    #
    # #clinical-decision rules
    # #("iai-pecarn", "iai_pecarn.csv", "imodels"),
    #
    # #popular classification datasets used in rule-based modeling / fairness
    # # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("juvenile", "juvenile_clean", 'imodels'),
    ("recidivism", "compas_two_year_clean", 'imodels'),
    # # ("credit", "credit_card_clean", 'imodels'),
    # # ("readmission", 'readmission_clean', 'imodels'),  # v big
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ('friedman1', 'friedman1', 'synthetic'),
    ('friedman2', 'friedman2', 'synthetic'),
    ('friedman3', 'friedman3', 'synthetic'),
    ('abalone', '183', 'openml'),
    ("diabetes-regr", "diabetes", 'sklearn'),
    ("california-housing", "california_housing", 'sklearn'),  # this replaced boston-housing due to ethical issues
    ("satellite-image", "294_satellite_image", 'pmlb'),
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'),  # this one is v big (100k examples)

]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    return parser.parse_args()


def get_n_rules(est):
    if type(est) == XGBClassifier:
        return np.sum([t.count("<") for t in est.get_booster().get_dump()])


def _get_estimator_performance(est, datas, met):
    X_train, X_test, y_train, y_test = datas
    est.fit(X_train, y_train)
    y_hat = est.predict(X_test)
    return met(y_hat, y_test)


def _get_n_samples(dataset):
    _, y, _ = get_clean_dataset(dataset[1], data_source=dataset[2])
    return len(y)


def compare_dataset(d, n_reps, figs, xgb, met):
    X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
    n, p = X.shape
    LOGGER.info(f"Working on {d[0]} (n,p) = ({n}, {p})")
    if len(y) > 9999:
        return

    performace = {"FIGS": [], "XGB": []}

    figs_mets = []
    xgb_mets = []

    for rep in range(n_reps):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.3, random_state=rep)

        datas = (X_train, X_test, y_train, y_test)

        xgb_perf = _get_estimator_performance(xgb, datas, met)
        figs_perf = _get_estimator_performance(figs, datas, met)

        figs_mets.append(figs_perf)
        xgb_mets.append(xgb_perf)

    performace['FIGS'] = {"mean": np.mean(figs_mets), "std": np.std(figs_mets),
                          "n": n, "p": p, "n_trees": len(figs.trees_), "n_rules": figs.complexity_}
    performace['XGB'] = {"mean": np.mean(xgb_mets), "std": np.std(xgb_mets),
                         "n_trees": len(xgb.get_booster().get_dump()), "n": n, "p": p, "n_rules": get_n_rules(xgb)}

    df = pd.DataFrame(performace)
    df.to_csv(os.path.join(PTH, f"{d[0]}.csv"))

    # return performace


def compare_performace(figs, xgb, datasets, met):
    for d in datasets:
        compare_dataset(d, N_REPS, figs, xgb, met)


def main():
    figs_cls = FIGSExtClassifier()
    xgb_cls = XGBClassifier()
    args = parse_args()
    ds = [d for d in DATASETS_CLASSIFICATION if d[0] == args.dataset]
    compare_performace(figs_cls, xgb_cls, ds, roc_auc_score)

    # figs_reg = FIGSExtRegressor()
    # xgb_reg = XGBRegressor()
    # compare_performace(figs_reg, xgb_reg, DATASETS_REGRESSION, mean_squared_error)


if __name__ == '__main__':
    main()
