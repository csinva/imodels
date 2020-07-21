import numpy as np
import math

class GreedyRuleList(object):
    def __init__(self, max_depth=1e3):
        self.depth = 0
        self.max_depth = max_depth
        self.feature_names=None
        
    def find_best_split(self, col, y):
        """
        col: the column we split on
        y: target var
        """
        min_entropy = 10    
        n = len(y)
        for value in set(col):  # iterating through each value in the column
            y_predict = col < value  # separate y into 2 groups
            my_entropy = get_entropy(y_predict, y)  # get entropy of this split
            if my_entropy <= min_entropy:  # check if it's the best one so far
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
    
    def find_best_split_of_all(self, x, y):
        """
        Find the best split from all features
        returns: the column to split on, the cutoff value, and the actual entropy
        """
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):  # iterating through each feature
            entropy, cur_cutoff = self.find_best_split(c, y)  # find the best split of that feature
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:  # check if it's best so far
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    
    def fit(self, x, y, depth=0, feature_names=None, verbose=True):
        """
        x
            Feature set
        y
            target variable
        par_node
            will be the tree generated for this x and y. 
        depth
            the depth of the current layer
        """
        if 'pandas' in str(type(x)):
            self.feature_names = x.columns
            x = x.values
        else:
            if self.feature_names is None:
                self.feature_names = ['feat ' + str(i) for i in range(x.shape[1])]
        if feature_names is not None:
            self.feature_names = feature_names
        
        if 'pandas' in str(type(y)):
            y = y.values
            
        if len(y) == 0:   # base case 1: no data in this group
            return []
        elif self.all_same(y):   # base case 2: all y is the same in this group
            return [{'val': y[0], 'num_pts': y.size}]
        elif depth >= self.max_depth:   # base case 4: max depth reached 
            return []
        else:   # Recursively generate rule list! 
            
            # find one split given an information gain 
            col, cutoff, entropy = self.find_best_split_of_all(x, y)  
            if verbose:
                print('mean', np.mean(y))
            y_left = y[x[:, col] < cutoff]  # left hand side data
            y_right = y[x[:, col] >= cutoff]  # left hand side data
            par_node = [{
                'col': self.feature_names[col],
                'index_col': col,
                'cutoff': cutoff,
                'val': np.mean(y),
                'val_right': np.mean(y_right),
                'num_pts': y.size, 
                'num_pts_right': y_right.size
            }]  # save the information 
            
            # generate tree for the left hand side data
            par_node = par_node + self.fit(x[x[:, col] < cutoff], y_left, depth + 1)   
            
            self.depth += 1   # increase the depth since we call fit once
            self.rules_ = par_node  
            return par_node
    
    def all_same(self, items):
        return all(x == items[0] for x in items)
    
    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros(n)
        for i in range(n):
            x = X[i]
            for j, rule in enumerate(self.rules_):
                if j == len(self.rules_) - 1:
                    probs[i] = rule['val']
                elif x[rule['index_col']] >= rule['cutoff']:
                    probs[i] = rule['val']
                    break
        return np.vstack((1 - probs, probs)).transpose() # probs (n, 2)
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
    
    def __str__(self):
        s = ''
        for rule in self.rules_:
            s += f"mean {rule['val'].round(3)} ({rule['num_pts']} pts)\n"
            if 'col' in rule:
                s += f"if {rule['col']} >= {rule['cutoff']} then {rule['val_right'].round(3)} ({rule['num_pts_right']} pts)\n"
        return s
            
    

def entropy_func(c, n):
    """
    The math formula
    """
    return -(c*1.0/n)*math.log(c*1.0/n, 2)

def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)

# get the entropy of one big circle showing above
def entropy_of_one_division(division): 
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division==c)
        e = n_c * 1.0 / n * entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
        s += e
    return s, n

# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict]) # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
    return s