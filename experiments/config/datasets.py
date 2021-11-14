from experiments.util import DATASET_PATH
from os.path import join as oj

DATASETS_CLASSIFICATION = [
    # popular classification datasets used in rule-based modeling / fairness
    # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    ("recidivism", oj(DATASET_PATH, "compas-analysis/compas_two_year_clean.csv"), 'local'),
    ("credit", oj(DATASET_PATH, "credit_card/credit_card_clean.csv"), 'local'),
    ("juvenile", oj(DATASET_PATH, "ICPSR_03986/DS0001/juvenile_clean.csv"), 'local'),
    ("readmission", oj(DATASET_PATH, 'readmission/readmission_clean.csv'), 'local'),

    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    ("ionosphere", "59", 'openml'),
    ("breast-cancer", oj(DATASET_PATH, "breast_cancer.csv"), 'local'),
    ("credit-g", oj(DATASET_PATH, "credit_g.csv"), 'local'),
    ("haberman", oj(DATASET_PATH, "haberman.csv"), 'local'),
    ("heart", oj(DATASET_PATH, "heart.csv"), 'local'),
]

DATASETS_REGRESSION = [
    ("diabetes", "diabetes", 'sklearn'),
    ("california-housing", "diabetes", 'sklearn'),
    # ("breast-tumor", "1201_BNG_breastTumor", 'pmlb'), # this one is v big (100k examples)
    ("echo-months", "1199_BNG_echoMonths", 'pmlb'),
    ("satellite-image", "294_satellite_image", 'pmlb'),
]
