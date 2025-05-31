# generic imports
import numpy as np

# imports from imodels
from imodels.tree.rf_plus.feature_importance.ppms.ppms import MDIPlusGenericRegressorPPM, MDIPlusGenericClassifierPPM

class LMDIPlus():
    """
    Local MDI+ (LMDI+) Explainer for tree-based models.

    Parameters:
    ----------
    rf_plus_model : RFPlusModel
        A trained RF+ model.
    evaluate_on : str
        Specifies which samples to use when computing local importances:
        - 'all' (default): evaluate on all samples.
        - 'oob': only on out-of-bag samples per tree.
        - 'inbag': only on in-bag samples per tree.
    """

    def __init__(self, rf_plus_model, evaluate_on = 'all'):
        self.rf_plus_model = rf_plus_model
        self.mode = 'only_k'
        self.oob_indices = self.rf_plus_model._oob_indices
        self.evaluate_on = evaluate_on
        if self.rf_plus_model._task == "classification":
            self.tree_explainers = [MDIPlusGenericClassifierPPM(rf_plus_model.estimators_[i]) 
                                    for i in range(len(rf_plus_model.estimators_))]
        else:
            self.tree_explainers = [MDIPlusGenericRegressorPPM(rf_plus_model.estimators_[i]) 
                                    for i in range(len(rf_plus_model.estimators_))]
            

    def get_lmdi_plus_scores(self, X, y = None, njobs = 1, ranking = False):
        """
        Compute LMDI+ scores for each sample in X.

        If `y` is provided, the evaluation is assumed to be on the training set, and if `y` is None, 
        it assumes LFI computation is being performed on unseen test data.

        Parameters:
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        y : np.ndarray or None
            Target values for the training set or None for test data.
        njobs : int
            Number of parallel jobs to use for prediction (if supported).
        ranking : bool
            If True, converts the LFI scores to feature rankings per sample.

        Returns:
        -------
        local_feature_importances : np.ndarray
            Matrix of shape (n_samples, n_features) containing the averaged LMDI+ scores across trees.
        """

        local_feature_importances = np.full((X.shape[0], X.shape[1],len(self.tree_explainers)), np.nan)
        if y is None:
            evaluate_on = None
        else:
            evaluate_on = self.evaluate_on
        
        lfi_scores = self._get_LFI_subtract_intercept(X, njobs)

        for i in range(lfi_scores.shape[-1]):
            ith_tree_scores = lfi_scores[:, :, i]
            oob_indices = np.unique(self.oob_indices[i])
            if evaluate_on == 'oob':
                local_feature_importances[oob_indices, :, i] = \
                    ith_tree_scores[oob_indices, :]
            elif evaluate_on == 'inbag':
                inbag_indices = np.arange(X.shape[0])
                inbag_indices = np.setdiff1d(inbag_indices, oob_indices)
                local_feature_importances[inbag_indices, :, i] = \
                    ith_tree_scores[inbag_indices, :]
            else:
                local_feature_importances[:, :, i] = ith_tree_scores
        
        if ranking:
            local_feature_importances = np.abs(local_feature_importances)
            rank_matrix = np.zeros_like(local_feature_importances)
            for i in range(local_feature_importances.shape[-1]):
                lfi_treei = local_feature_importances[:,:,i]
                indices_of_zero_columns = np.where(np.all(lfi_treei==0, axis=0))[0]
                lfi_treei[:, indices_of_zero_columns] = -1
                ranks = np.argsort(np.argsort(lfi_treei, kind="stable"), kind = "stable")
                ranks = np.array(ranks, dtype=np.float32)
                ranks[:, indices_of_zero_columns] = np.nan
                rank_matrix[:,:,i] = ranks
            local_feature_importances = rank_matrix

        local_feature_importances = np.nanmean(local_feature_importances, axis=-1)
        local_feature_importances[np.isnan(local_feature_importances)] = 0
        return local_feature_importances

    
    def _get_LFI_subtract_intercept(self, X, njobs):
        """
        Compute per-tree LMDI+ scores for each sample in X.

        Parameters:
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        njobs : int
            Number of parallel jobs to use for prediction (if supported).

        Returns:
        -------
        LFIs : np.ndarray
            Matrix of shape (n_samples, n_features, n_trees) containing the per-tree
            local feature importance scores.
        """
        LFIs = np.zeros((X.shape[0],X.shape[1],len(self.tree_explainers)))
        for i, tree_explainer in enumerate(self.tree_explainers):
            blocked_data_ith_tree = self.rf_plus_model.transformers_[i].transform(X)
            if self.rf_plus_model._task == "classification":
                ith_partial_preds = tree_explainer.predict_partial_subtract_intercept(blocked_data_ith_tree, njobs=njobs)
            else:
                ith_partial_preds = tree_explainer.predict_partial_subtract_intercept(blocked_data_ith_tree, njobs=njobs)
            ith_partial_preds = np.array([ith_partial_preds[j] for j in range(X.shape[1])]).T
            LFIs[:,:,i] = ith_partial_preds
        return LFIs