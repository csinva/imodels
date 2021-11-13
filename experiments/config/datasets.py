from experiments.util import DATASET_PATH
from os.path import join as oj

DATASETS_CLASSIFICATION = [
    ("recidivism", oj(DATASET_PATH, "compas-analysis/compas_two_year_clean.csv"), 'local'),
    ("credit", oj(DATASET_PATH, "credit_card/credit_card_clean.csv"), 'local'),
    ("juvenile", oj(DATASET_PATH, "ICPSR_03986/DS0001/juvenile_clean.csv"), 'local'),
    ("readmission", oj(DATASET_PATH, 'readmission/readmission_clean.csv'), 'local'),
    ("breast-cancer", oj(DATASET_PATH, "breast_cancer.csv"), 'local'),
    ("credit-g", oj(DATASET_PATH, "credit_g.csv"), 'local'),
    ("haberman", oj(DATASET_PATH, "haberman.csv"), 'local'),
    ("heart", oj(DATASET_PATH, "heart.csv"), 'local'),
]

DATASETS_REGRESSION = [
    ("diabetes", "diabetes", 'sklearn'),
    ("breastTumor", "1201_BNG_breastTumor", 'pmlb'),
]
