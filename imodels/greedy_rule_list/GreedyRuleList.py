import numpy as np
import math
from copy import deepcopy

class GreedyRuleList(object):
    def __init__(self, max_depth=1e3, class_weight=None, criterion='gini'):
        self.depth = 0
        self.max_depth = max_depth
        self.feature_names = None
        self.class_weight = class_weight
        self.criterion = criterion
    
    def fit(self, x, y, depth=0, feature_names=None, verbose=False):
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
        
        # base case 1: no data in this group
        if len(y) == 0:
            return []
        
        # base case 2: all y is the same in this group
        elif self.all_same(y):
            return [{'val': y[0], 'num_pts': y.size}]
        
        # base case 4: max depth reached 
        elif depth >= self.max_depth:   
            return []
        
        # Recursively generate rule list! 
        else:   
            
            # find one split given an information gain 
            col, cutoff, criterion_val = self.find_best_split(x, y)  
            if verbose:
                print('mean', (100 * np.mean(y)).round(3))
            
            y_left = y[x[:, col] < cutoff]  # left-hand side data
            y_right = y[x[:, col] >= cutoff]  # right-hand side data
            if np.mean(y_left) > np.mean(y_right):
                flip = True
                tmp = deepcopy(y_left)
                y_left = deepcopy(y_right)
                y_right = tmp
                x_left = x[x[:, col] >= cutoff]
            else:
                flip = False
                x_left = x[x[:, col] < cutoff]
            par_node = [{
                'col': self.feature_names[col],
                'index_col': col,
                'cutoff': cutoff,
                'val': np.mean(y),
                'flip': flip,
                'val_right': np.mean(y_right),
                'num_pts': y.size, 
                'num_pts_right': y_right.size
            }]  # save the information 
            
            # generate tree for the left hand side data
            par_node = par_node + self.fit(x_left, y_left, depth + 1, verbose=verbose)   
            
            self.depth += 1   # increase the depth since we call fit once
            self.rules_ = par_node  
            return par_node
    
    def predict_proba(self, X):
        if 'pandas' in str(type(X)):
            X = X.values
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
    
    def print_list(self):
        '''Print out the list in a nice way
        '''
        s = ''
        def red(s):
            return f"\033[91m{s}\033[00m"
        def cyan(s): 
            return f"\033[96m{s}\033[00m"
        def rule_name(rule):
            if rule['flip']:
                return '~' + rule['col']
            return rule['col']
        rule = self.rules_[0]
    #     s += f"{red((100 * rule['val']).round(3))}% IwI ({rule['num_pts']} pts)\n"
        for rule in self.rules_:
            s += f"\t{'':>35} => {cyan((100 * rule['val']).round(2)):>6}% risk ({rule['num_pts']} pts)\n"
    #         s += f"\t{'Else':>45} => {cyan((100 * rule['val']).round(2)):>6}% IwI ({rule['val'] * rule['num_pts']:.0f}/{rule['num_pts']} pts)\n"
            if 'col' in rule:
    #             prefix = f"if {rule['col']} >= {rule['cutoff']}"
                prefix = f"if {rule_name(rule)}"
                val = f"{100 * rule['val_right'].round(3):.4}"
                s += f"{prefix:>43} ===> {red(val)}% risk ({rule['num_pts_right']} pts)\n"
        rule = self.rules_[-1]
    #     s += f"{red((100 * rule['val']).round(3))}% IwI ({rule['num_pts']} pts)\n"
        print(s)
    
    def all_same(self, items):
        return all(x == items[0] for x in items)
            
    def find_best_split(self, x, y):
        """
        Find the best split from all features
        returns: the column to split on, the cutoff value, and the actual criterion_value
        """
        col = None
        min_criterion_val = 1e10
        cutoff = None

        # iterating through each feature
        for i, c in enumerate(x.T):

            # find the best split of that feature
            criterion_val, cur_cutoff = self.split_on_feature(c, y)  

            # found perfect cutoff
            if criterion_val == 0:    
                return i, cur_cutoff, criterion_val

            # check if it's best so far
            elif criterion_val <= min_criterion_val:  
                min_criterion_val = criterion_val
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_criterion_val    

    def split_on_feature(self, col, y):
        """
        col: the column we split on
        y: target var
        """
        min_criterion_val = 1e10    
        n = len(y)
        
        # iterate through each value in the column
        for value in set(col):
            
            # separate y into 2 groups
            y_predict = col < value  

            # get criterion val of this split
            criterion_val = self.weighted_criterion(y_predict, y)

            # check if it's the best one so far
            if criterion_val <= min_criterion_val:
                min_criterion_val = criterion_val
                cutoff = value
        return min_criterion_val, cutoff    

    def weighted_criterion(self, y_predict, y_real):
        """Returns criterion calculated over a split
        y_predict is the split decision, True/False, and y_true can be multi class
        """
        if len(y_predict) != len(y_real):
            print('They have to be the same length')
            return None

        if self.criterion == 'entropy':
            criterion_func = self.entropy_one_node
        elif self.criterion == 'gini':
            criterion_func = self.gini_one_node
        
        # left-hand side criterion
        s_left = criterion_func(y_real[y_predict])
        
        # right-hand side criterion
        s_right = criterion_func(y_real[~y_predict])
        
        # overall criterion, again weighted average
        n = len(y_real)
        sample_weights = np.ones(n)
        if self.class_weight is not None:
            for c in self.class_weight.keys():
                idxs_c = y_real == c
                sample_weights[idxs_c] = self.class_weight[c]
        tot_weight = np.sum(sample_weights)
        weight_left = np.sum(sample_weights[y_predict]) / tot_weight
        weight_right = np.sum(sample_weights[~y_predict]) / tot_weight
        s =  weight_left * s_left + weight_right * s_right
        return s

    def gini_one_node(self, division):
        '''Returns gini index for one node
        = sum(pc * (1 – pc))
        '''
        s = 0
        n = len(division)
        classes = set(division)
        
        # for each class, get entropy
        for c in classes:   
            
            # weights for each class
            n_c = sum(division==c)
            p_c = n_c / n
            
            # weighted avg
            s += p_c * (1 - p_c)

        return s
    
    def entropy_one_node(self, division): 
        """Returns entropy of a divided group of data
        Data may have multiple classes
        """
        s = 0
        n = len(division)
        classes = set(division)
        
        # for each class, get entropy
        for c in classes:   
            
            # weights for each class
            weight = sum(division==c) / n
                
            # weighted avg
            s += weight * entropy_from_counts(sum(division==c), sum(division!=c))
        return s

def entropy_from_counts(c1, c2):
    """Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1 == 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    
    def entropy_func(p): return -p * math.log(p, 2)
    
    p1 = c1 * 1.0 / (c1 + c2)
    p2 = c2 * 1.0 / (c1 + c2)
    return entropy_func(p1) + entropy_func(p2)

