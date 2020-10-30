import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted


class RuleSet:

    def decision_function(self, X) -> np.ndarray:
        """Average anomaly score of X of the base classifiers (rules).

        The anomaly score of an input sample is computed as
        the weighted sum of the binary rules outputs, the weight being
        the respective precision of each rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The anomaly score of the input samples.
            The higher, the more abnormal. Positive scores represent outliers,
            null scores represent inliers.

        """
        # Check if fit had been called
        # check_is_fitted(self, ['rules_', 'estimators_', 'estimators_samples_',
        #                        'max_samples_'])

        # Input validation
        X = check_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time."
                             " Please reshape your data."
                             % (X.shape[1], self.n_features_))

        df = pd.DataFrame(X, columns=self.feature_names_)
        selected_rules = self.rules_without_feature_names_

        scores = np.zeros(X.shape[0])
        for (r, w) in selected_rules:
            scores[list(df.query(r).index)] += w[0]

        return scores
