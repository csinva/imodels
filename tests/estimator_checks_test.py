import unittest
from inspect import signature

from sklearn.utils.estimator_checks import check_estimator

import imodels


class TestCheckEstimators(unittest.TestCase):
    """Checks that estimators conform to sklearn checks
    """

    def test_check_classifier_compatibility(self):
        """Test classifiers are properly sklearn-compatible
        """
        # BoostedRulesClassifier is excluded (multi-class not supported)
        for classifier in [imodels.SLIMClassifier]:
            check_estimator(classifier())

    def test_method_signatures_basic(self):
        """Every registered estimator exposes the standard methods, taking X (and y)"""
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
