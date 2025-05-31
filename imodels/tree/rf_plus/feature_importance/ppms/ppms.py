# generic imports
import copy
from abc import ABC
import numpy as np
from scipy.special import expit
from joblib import Parallel, delayed

# imodels imports
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_regression import AloGLMRegressor
from imodels.tree.rf_plus.rf_plus_prediction_models.aloocv_classification import AloGLMClassifier

class MDIPlusGenericRegressorPPM(ABC):
    """
    Partial prediction model for arbitrary estimators. Parallelized for speedup.
    """

    def __init__(self, estimator):
        """
        Constructor for the MDIPlusGenericRegressorPPM class.
        
        Args:
            estimator (object): A regression estimator object with a
                                'predict' method.
        """
        self.estimator = copy.deepcopy(estimator)

    def predict_full(self, blocked_data):
        """
        Gets the full predictions for the model.
        
        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
        
        Returns:
            np.ndarray: Full predictions for the model.
        """
        return self.estimator.predict(blocked_data.get_all_data())
    
    def predict_partial(self, blocked_data, mode, l2norm):
        """
        Gets the partial predictions. To be used when we want to incorporate the
        intercept of the regression model into the partial predictions.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k(blocked_data, k, mode,
                                                      l2norm)
        return partial_preds
    
    def predict_partial_subtract_intercept(self, blocked_data, njobs = 1):
        """
        Gets the partial predictions. To be used when we do not want to consider
        the intercept of the regression model, such as the partial linear LMDI+
        implementation.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sign (bool): indicator for if we want to retain the direction of
                         the partial prediction.
            normalize (bool): indicator for if we want to normalize the partial
                              predictions by the size of the full prediction.
            njobs (int): number of jobs to run in parallel.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        n_blocks = blocked_data.n_blocks
        
        # helper function to parallelize the partial prediction computation
        def predict_wrapper(k):
            psibeta = self.predict_partial_k_subtract_intercept(blocked_data, k)
            return psibeta
        
        # delayed makes sure that predictions get arranged in the correct order
        partial_preds = Parallel(n_jobs=njobs)(delayed(predict_wrapper)(k)
                                               for k in range(n_blocks))
        
        # parse through the outputs of the parallel data structure
        partial_pred_storage = {}
        for k in range(len(partial_preds)):
            partial_pred_storage[k] = partial_preds[k]

        return partial_pred_storage
    
    def predict_partial_k(self, blocked_data, k, mode, l2norm):
        """
        Gets the partial predictions for an individual feature k, including the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        modified_data = blocked_data.get_modified_data(k, "keep_rest_zero")
        return self.estimator.predict(modified_data)
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k):
        """
        Gets the partial predictions for an individual feature k, omitting the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        psi_k = blocked_data.get_modified_data(k, "only_k")
        coefs = self.estimator.coef_
        return psi_k @ coefs

class MDIPlusGenericClassifierPPM(ABC):
    """
    Partial prediction model for arbitrary classification estimators. May be slow.
    """
    def __init__(self, estimator):
        """
        Constructor for the MDIPlusGenericClassifierPPM class.
        
        Args:
            estimator (object): A classification estimator object with a
                                'predict_proba' method.
        """
        self.estimator = copy.deepcopy(estimator)

    def predict_full(self, blocked_data):
        """
        Gets the full predictions for the model.
        
        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            
        Returns:
            np.ndarray: Full predictions for the model.
        """
        return self.estimator.predict_proba(blocked_data.get_all_data())[:,1]
    
    def predict_partial(self, blocked_data, mode, l2norm, sigmoid):
        """
        Gets the partial predictions. To be used when we want to incorporate the
        intercept of the regression model into the partial predictions.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        n_blocks = blocked_data.n_blocks
        partial_preds = {}
        for k in range(n_blocks):
            partial_preds[k] = self.predict_partial_k(blocked_data, k, mode,
                                                      l2norm, sigmoid)
        return partial_preds
    
    def predict_partial_subtract_intercept(self, blocked_data, njobs = 1):
        """
        Gets the partial predictions. To be used when we do not want to consider
        the intercept of the regression model, such as the partial linear LMDI+
        implementation.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sign (bool): indicator for if we want to retain the direction of
                         the partial prediction.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.
            normalize (bool): indicator for if we want to normalize the partial
                              predictions by the size of the full prediction.
            njobs (int): number of jobs to run in parallel.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        n_blocks = blocked_data.n_blocks
        
        # helper function to parallelize the partial prediction computation
        def predict_wrapper(k):
            psibeta = self.predict_partial_k_subtract_intercept(blocked_data, k)
            return psibeta
        
        # delayed makes sure that predictions get arranged in the correct order
        partial_preds = Parallel(n_jobs=njobs)(delayed(predict_wrapper)(k)
                                               for k in range(n_blocks))
        
        # parse through the outputs of the parallel data structure
        partial_pred_storage = {}
        for k in range(len(partial_preds)):
            partial_pred_storage[k] = partial_preds[k]

        return partial_pred_storage
    
    def predict_partial_k_subtract_intercept(self, blocked_data, k):
        """
        Gets the partial predictions for an individual feature k, omitting the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        psi_k = blocked_data.get_modified_data(k, "only_k")
        coefs = self.estimator.coef_
        # reshape coefs if necessary
        if coefs.shape[0] != psi_k.shape[1]:
            coefs = coefs.reshape(-1,1).flatten()
        return psi_k @ coefs

    def predict_partial_k(self, blocked_data, k, mode, l2norm, sigmoid):
        """
        Gets the partial predictions for an individual feature k, including the
        regression intercept in the predictions for the model.

        Args:
            blocked_data (BlockedPartitionData): Psi(X) data.
            k (int): feature index.
            mode (str): either {"keep_k", "keep_rest"}, see BlockPartitionedData
            l2norm (bool): indicator for if we want to take the l2-normed
                           product of the data and the coefficients.
            sigmoid (bool): indicator for if we want to apply the sigmoid
                            function to our classification outcome.

        Returns:
            dict: mapping of feature index to partial predictions.
        """
        
        modified_data = blocked_data.get_modified_data(k, mode)
        coefs = self.estimator.coef_
        return modified_data @ coefs + self.estimator.intercept_
