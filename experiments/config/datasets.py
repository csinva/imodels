from os.path import join as oj

from experiments.util import DATASET_PATH

DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ("sonar", "sonar", "pmlb"),
    ("heart", oj(DATASET_PATH, "heart.csv"), 'local'),
    ("breast-cancer", oj(DATASET_PATH, "breast_cancer.csv"), 'local'),
    ("haberman", oj(DATASET_PATH, "haberman.csv"), 'local'),
    ("ionosphere", "ionosphere", 'pmlb'),
    ("diabetes", "diabetes", "pmlb"),
    # ("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # ("credit-g", oj(DATASET_PATH, "credit_g.csv"), 'local'), # like german-credit, but more feats
    ("german-credit", "german", "pmlb"),

    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("juvenile", oj(DATASET_PATH, "ICPSR_03986/DS0001/juvenile_clean.csv"), 'local'),
    ("recidivism", oj(DATASET_PATH, "compas-analysis/compas_two_year_clean.csv"), 'local'),
    ("credit", oj(DATASET_PATH, "credit_card/credit_card_clean.csv"), 'local'),
    ("readmission", oj(DATASET_PATH, 'readmission/readmission_clean.csv'), 'local'), # v big
]

DATASETS_REGRESSION = [
    # leo-breiman paper random forest uses some UCI datasets as well
    # pg 23: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ('friedman1', 'friedman1', 'synthetic'),
    ('friedman2', 'friedman2', 'synthetic'),
    ('friedman3', 'friedman3', 'synthetic'),

    ("diabetes-regr", "diabetes", 'sklearn'),
    ("california-housing", "california_housing", 'sklearn'),
    ("satellite-image", "294_satellite_image", 'pmlb'),
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'), # this one is v big (100k examples)

]
