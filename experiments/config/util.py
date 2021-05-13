from experiments.config.config_readmission import READMISSION_ESTIMATORS, READMISSION_TEST_ESTIMATORS, get_ensembles_for_readmission
from experiments.config.config_credit import CREDIT_ESTIMATORS, CREDIT_TEST_ESTIMATORS, get_ensembles_for_credit
from experiments.config.config_recidivism import RECIDIVISM_ESTIMATORS, RECIDIVISM_TEST_ESTIMATORS, get_ensembles_for_recidivism
from experiments.config.config_juvenile import JUVENILE_ESTIMATORS, JUVENILE_TEST_ESTIMATORS, get_ensembles_for_juvenile


def get_estimators_for_dataset(dataset: str, test: bool = False):
    if dataset == 'readmission':
        return READMISSION_TEST_ESTIMATORS if test else READMISSION_ESTIMATORS
    elif dataset == 'credit':
        return CREDIT_TEST_ESTIMATORS if test else CREDIT_ESTIMATORS
    elif dataset == 'recidivism':
        return RECIDIVISM_TEST_ESTIMATORS if test else RECIDIVISM_ESTIMATORS
    elif dataset == 'juvenile':
        return JUVENILE_TEST_ESTIMATORS if test else JUVENILE_ESTIMATORS


def get_ensembles_for_dataset(dataset: str, test: bool = False):
    if dataset == 'readmission':
        return get_ensembles_for_readmission(test)
    elif dataset == 'credit':
        return get_ensembles_for_credit(test)
    elif dataset == 'recidivism':
        return get_ensembles_for_recidivism(test)
    elif dataset == 'juvenile':
        return get_ensembles_for_juvenile(test)
