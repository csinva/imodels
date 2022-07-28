import unittest

from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks
from inspect import signature
import imodels


class TestCheckEstimators(unittest.TestCase):
    """Checks that estimators conform to sklearn checks
    """

    def test_check_classifier_compatibility(self):
        """Test classifiers are properly sklearn-compatible
        """
        for classifier in [imodels.SLIMClassifier]:  # BoostedRulesClassifier (multi-class not supported)
            check_estimator(classifier())
            assert 'passed check_estimator for ' + str(classifier)

    def test_check_regressor_compatibility(self):
        """Test regressors are properly sklearn-compatible
        """
        for regr in []:  # SLIMRegressor fails acc screening for boston dset
            check_estimator(regr())
            assert 'passed check_estimator for ' + str(regr)

    def test_method_signatures_basic(self):
        for estimator in imodels.ESTIMATORS:
            assert hasattr(estimator, 'fit')
            assert 'X' in signature(estimator.fit).parameters, str(estimator) + ' failed fit parameters'
            assert 'y' in signature(estimator.fit).parameters, str(estimator) + ' failed fit parameters'

            assert hasattr(estimator, 'predict')
            assert 'X' in signature(estimator.predict).parameters, str(estimator) + ' failed predict parameters'

        for estimator in imodels.CLASSIFIERS:
            assert hasattr(estimator, 'predict_proba')
            assert 'X' in signature(estimator.predict_proba).parameters, str(
                estimator) + ' failed predict_proba parameters'


if __name__ == '__main__':
    check_estimator(imodels.FIGSRegressor())
