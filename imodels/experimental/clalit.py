import os.path
import pickle

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, roc_auc_score

from imodels.experimental import FIGSExtRegressor, FIGSExtClassifier
from .. import FIGSRegressorCV, FIGSClassifierCV, get_clean_dataset
from xgboost import XGBClassifier, XGBRegressor

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
    # ("german-credit", "german", "pmlb"),
    #
    # #clinical-decision rules
    # #("iai-pecarn", "iai_pecarn.csv", "imodels"),
    #
    # #popular classification datasets used in rule-based modeling / fairness
    # # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("juvenile", "juvenile_clean", 'imodels'),
    # ("recidivism", "compas_two_year_clean", 'imodels'),
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
    if len(y) > 9999:
        return

    performace = {"FIGS": [], "XGB": []}

    figs_mets = []
    xgb_mets = []

    for rep in range(n_reps):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.3, random_state=rep)

        datas = (X_train, X_test, y_train, y_test)

        figs_perf = _get_estimator_performance(figs, datas, met)
        xgb_perf = _get_estimator_performance(xgb, datas, met)

        figs_mets.append(figs_perf)
        xgb_mets.append(xgb_perf)

    performace['FIGS'] = {"mean": np.mean(figs_mets), "std": np.std(figs_mets)}
    performace['XGB'] = {"mean": np.mean(xgb_mets), "std": np.std(xgb_mets)}

    return performace


def compare_performace(figs, xgb, datasets, met):
    perf = {d[0]: compare_dataset(d, N_REPS, figs, xgb, met) for d in datasets}

    performace = {k:v for k,v in perf.items() if v is not None}

    # fig, ax = plt.subplots(1)
    #
    # x_axis = [_get_n_samples(d) for d in datasets]
    #
    # y_axis_figs = [performace[d[0]]['FIGS']['mean'] for d in datasets]
    # y_axis_err_figs = [performace[d[0]]['FIGS']['std'] for d in datasets]
    #
    # y_axis_xgb = [performace[d[0]]['XGB']['mean'] for d in datasets]
    # y_axis_err_xgb = [performace[d[0]]['XGB']['std'] for d in datasets]

    with open(os.path.join(PTH, "cls.pkl"), "wb") as stream:
        pickle.dump(performace, stream)

    # ax.errorbar(x_axis, y_axis_figs, yerr=y_axis_err_figs, label="FIGS", color="blue")
    # ax.errorbar(x_axis, y_axis_xgb, yerr=y_axis_err_xgb, label="XGB", color="red")
    #
    # ax.legend()
    # plt.show()


def main():
    figs_cls = FIGSExtClassifier()
    xgb_cls = XGBClassifier()
    compare_performace(figs_cls, xgb_cls, DATASETS_CLASSIFICATION, roc_auc_score)

    # figs_reg = FIGSExtRegressor()
    # xgb_reg = XGBRegressor()
    # compare_performace(figs_reg, xgb_reg, DATASETS_REGRESSION, mean_squared_error)


if __name__ == '__main__':
    main()
