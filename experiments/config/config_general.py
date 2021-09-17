from experiments.util import DATASET_PATH


DATASETS = [
    ("recidivism", DATASET_PATH + "compas-analysis/compas_two_year_clean.csv"),
    ("credit", DATASET_PATH + "credit_card/credit_card_clean.csv"),
    ("juvenile", DATASET_PATH + "ICPSR_03986/DS0001/data_clean.csv"),
    ("readmission", DATASET_PATH + 'readmission/readmission_clean.csv'),
    ("breast-cancer", DATASET_PATH + "breast_cancer.csv"), 
    ("credit-g", DATASET_PATH + "credit_g.csv"), 
    ("haberman", DATASET_PATH + "haberman.csv"), 
    ("heart", DATASET_PATH + "heart.csv")
]

RULEFIT_KWARGS = {'random_state': 0, 'max_rules': None, 'include_linear': False}
FPL_KWARGS = {'disc_strategy': 'simple', 'max_rules': None, 'include_linear': False}
BRL_KWARGS = {'disc_strategy': 'simple', 'max_iter': 2000}


# EASY_ESTIMATORS = deepcopy(ALL_ESTIMATORS)

# EASY_ESTIMATORS[3] = (
#     [Model('rulefit - alpha_30', rfit, 'n_estimators', n, 'alpha', 30, RULEFIT_KWARGS) for n in np.arange(1, 92, 10)]
#     + [Model('rulefit - alpha_13', rfit, 'n_estimators', n, 'alpha', 13, RULEFIT_KWARGS) for n in [1, 3] + list(np.arange(5, 38, 4))]
#     + [Model('rulefit - alpha_5', rfit, 'n_estimators', n, 'alpha', 5, RULEFIT_KWARGS) for n in np.arange(1, 38, 4)]
#     + [Model('rulefit - alpha_neg', rfit, 'n_estimators', n, 'alpha', 2, RULEFIT_KWARGS) for n in np.arange(1, 20, 2)]
#     + [Model('rulefit - alpha_1', rfit, 'n_estimators', n, 'alpha', 1, RULEFIT_KWARGS) for n in np.arange(1, 11)]
# )
# EASY_ESTIMATORS[4] = (
#     [Model('fplasso - max_card_1', fpl, 'alpha', a, 'maxcardinality', 1, FPL_KWARGS) for a in np.logspace(-0.5, 1.2, 10)]
#     + [Model('fplasso - max_card_2', fpl, 'alpha', a, 'maxcardinality', 2, FPL_KWARGS) for a in np.logspace(0.5, 1.4, 10)]
# )
# EASY_ESTIMATORS[5] = (
#     [Model('fpskope - max_card_1_prec_0.3', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.3}) for n in np.linspace(0.1, 0.5, 10)]
#     + [Model('fpskope - max_card_1_prec_0.4', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.4}) for n in np.linspace(0.08, 0.5, 10)]
#     + [Model('fpskope - max_card_1_prec_0.5', fps, 'minsupport', n, 'maxcardinality', 1, {'disc_strategy': 'simple', 'precision_min': 0.5}) for n in np.linspace(0.08, 0.5, 10)]
#     + [Model('fpskope - max_card_2_prec_0.3', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.3}) for n in np.linspace(0.3, 0.6, 10)]
#     + [Model('fpskope - max_card_2_prec_0.4', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.4}) for n in np.linspace(0.3, 0.6, 10)]
#     + [Model('fpskope - max_card_2_prec_0.5', fps, 'minsupport', n, 'maxcardinality', 2, {'disc_strategy': 'simple', 'precision_min': 0.5}) for n in np.linspace(0.3, 0.6, 10)]
# )
