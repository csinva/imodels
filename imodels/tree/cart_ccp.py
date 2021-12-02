from sklearn.tree import DecisionTreeClassifier, export_text, DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn import datasets
from sklearn.datasets import make_friedman1,make_friedman2,make_friedman3
from imodels.util.tree import compute_tree_complexity
from copy import deepcopy



class DecisionTreeClassifierCCP(DecisionTreeClassifier):
    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1):
        self.desired_complexity = desired_complexity
        #print('est', estimator_)
        self.estimator_ = estimator_
    
    def fit(self,X,y,sample_weight=None,*args, **kwargs):
        path = self.estimator_.cost_complexity_pruning_path(X,y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        complexities = {}
        for alpha in ccp_alphas: 
            est_params = self.estimator_.get_params()
            est_params['ccp_alpha'] = alpha
            copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
            copied_estimator.fit(X, y)
            complexities[alpha] = self._get_complexity(copied_estimator)
        closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
        params_for_fitting = self.estimator_.get_params()
        params_for_fitting['ccp_alpha'] = closest_alpha
        self.estimator_.set_params(**params_for_fitting)
        self.estimator_.fit(X,y,*args, **kwargs)
    
    def _get_complexity(self,BaseEstimator):
        return compute_tree_complexity(BaseEstimator.tree_)
    
    def predict_proba(self, *args, **kwargs):
        if hasattr(self.estimator_, 'predict_proba'):
            return self.estimator_.predict_proba(*args, **kwargs)
        else:
            return NotImplemented

    
    def predict(self,X,*args, **kwargs):
        return self.estimator_.predict(X,*args, **kwargs)
    
    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented
        
        
class DecisionTreeRegressorCCP(BaseEstimator):
    
    def __init__(self, estimator_: BaseEstimator, desired_complexity: int = 1):
        self.desired_complexity = desired_complexity
        #print('est', estimator_)
        self.estimator_ = estimator_
    
    def fit(self,X,y,sample_weight=None,*args, **kwargs):
        path = self.estimator_.cost_complexity_pruning_path(X,y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        complexities = {}
        for alpha in ccp_alphas: 
            est_params = self.estimator_.get_params()
            est_params['ccp_alpha'] = alpha
            copied_estimator =  deepcopy(self.estimator_).set_params(**est_params)
            copied_estimator.fit(X, y)
            complexities[alpha] = self._get_complexity(copied_estimator)
        closest_alpha, closest_leaves = min(complexities.items(), key=lambda x: abs(self.desired_complexity - x[1]))
        params_for_fitting = self.estimator_.get_params()
        params_for_fitting['ccp_alpha'] = closest_alpha
        self.estimator_.set_params(**params_for_fitting)
        self.estimator_.fit(X,y,*args, **kwargs)
    
    def _get_complexity(self,BaseEstimator):
        return compute_tree_complexity(BaseEstimator.tree_)
    
    def predict(self,X,*args, **kwargs):
        return self.estimator_.predict(X,*args, **kwargs)
    
    def score(self, *args, **kwargs):
        if hasattr(self.estimator_, 'score'):
            return self.estimator_.score(*args, **kwargs)
        else:
            return NotImplemented

if __name__ == '__main__':
    m = DecisionTreeClassifierCCP(estimator_=DecisionTreeClassifier(min_samples_leaf = 5),desired_complexity = 10)
    #X,y = make_friedman1()
    X, y = datasets.load_breast_cancer(return_X_y=True)
    m.fit(X,y)
    m.predict(X)
    m.score(X,y)

