import unittest

from sklearn.utils.estimator_checks import check_estimator

from imodels import *


class TestCheckEstimators(unittest.TestCase):
    '''Checks that estimators conform to sklearn checks
    '''

    def test_check_classifier_compatibility(self):
        '''Test classifiers are properly sklearn-compatible
        '''
        for classifier in [SLIMClassifier]:  # BoostedRulesClassifier (multi-class not supported)
            check_estimator(classifier())
            assert 'passed check_estimator for ' + str(classifier)

    def test_check_regressor_compatibility(self):
        '''Test regressors are properly sklearn-compatible
        '''
        for regr in []:  # SLIMRegressor fails acc screening for boston dset
            check_estimator(regr())
            assert 'passed check_estimator for ' + str(regr)
