from experiments.util import DATASET_PATH
from os.path import join as oj

DATASETS = [
    ("recidivism", oj(DATASET_PATH, "compas-analysis/compas_two_year_clean.csv")),
    ("credit", oj(DATASET_PATH, "credit_card/credit_card_clean.csv")),
    ("juvenile", oj(DATASET_PATH, "ICPSR_03986/DS0001/juvenile_clean.csv")),
    ("readmission", oj(DATASET_PATH, 'readmission/readmission_clean.csv')),
    ("breast-cancer", oj(DATASET_PATH, "breast_cancer.csv")),
    ("credit-g", oj(DATASET_PATH, "credit_g.csv")),
    ("haberman", oj(DATASET_PATH, "haberman.csv")),
    ("heart", oj(DATASET_PATH, "heart.csv")),
]
