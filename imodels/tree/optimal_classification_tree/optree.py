import pandas as pd
from pyomo.environ import *  # !! Please don't delete this
from pyomo.core.base.PyomoModel import *
from pyomo.core.base.constraint import *
from pyomo.core.base.objective import *
from pyomo.core.base.var import *
from pyomo.core.kernel.set_types import *
from pyomo.opt.base.solvers import *
import logging
import numpy as np
from abc import abstractmethod, ABCMeta
from sklearn.tree import DecisionTreeClassifier
from inspect import getmembers
from .tree import TreeModel, WholeTree, Tree
from .localsearch import OptimalTreeModelOptimizer, OptimalHyperTreeModelOptimizer
import multiprocessing


class AbstractOptimalTreeModel(metaclass=ABCMeta):
    def __init__(self, tree_depth: int = 3, N_min: int = 1, alpha: float = 0,
                 x_cols: list = ["x1", "x2"], y_col: str = "y", M: int = 10, 
                 epsilon: float = 1e-4, solver_name: str = "gurobi"):
        self.y_col = y_col
        self.P = len(x_cols)
        self.P_range = x_cols
        self.K_range = None
        self.solver_name = solver_name
        self.D = tree_depth
        self.Nmin = N_min
        self.M = M
        self.epsilon = epsilon
        self.alpha = alpha
        self.is_trained = False
        self.parent_nodes, self.leaf_ndoes = self.generate_nodes(self.D)
        self.normalizer = {}
        self.pool = None

        # optimization model
        self.model = None

        # root node is "1"
        # below it are "2" and "3" and so on for later depths
        
        # solutions
        self.l = None
        self.c = None # values of leaves
        self.d = None # depth
        self.a = None # which variable to split on
        self.b = None # value to split on
        self.Nt = None
        self.Nkt = None
        self.Lt = None
        
        
        assert tree_depth > 0, "Tree depth must be greater than 0! (Actual: {0})".format(tree_depth)

    def train(self, data: pd.DataFrame, train_method: str = "ls",
              show_training_process: bool = True, warm_start: bool = True,
              num_initialize_trees: int = 10):
        if train_method == "ls":
            self.fast_train(data, num_initialize_trees)
        elif train_method == "mio":
            self.exact_train(data, show_training_process, warm_start)
        else:
            raise ValueError("Illegal train_method! You should use one of 'ls' for local search(fast but local optima)"
                             "or "
                             "'mio' for Mixed Integer Optimization (global optima but much slow)")

    def exact_train(self, data: pd.DataFrame, show_training_process: bool = True, warm_start: bool = True):
        data = self.normalize_data(data)

        solver = SolverFactory(self.solver_name)

        start_tree_depth = 1 if warm_start else self.D

        global_status = "Not started"
        global_loss = np.inf
        previous_depth_params = None
        for d in range(start_tree_depth, self.D + 1):
            if d < self.D:
                logging.info("Warm starting the optimization with tree depth {0} / {1}...".format(d, self.D))
            else:
                logging.info("Optimizing the tree with depth {0}...".format(self.D))

            cart_params = self._get_cart_params(data, d)
            warm_start_params = self._select_better_warm_start_params([previous_depth_params, cart_params], data)

            parent_nodes, leaf_nodes = self.generate_nodes(d)
            model = self.generate_model(data, parent_nodes, leaf_nodes, warm_start_params)

            # model.pprint()

            res = solver.solve(model, tee=show_training_process, warmstart=True)
            status = str(res.solver.termination_condition)
            loss = value(model.obj)

            previous_depth_params = self._generate_warm_start_params_from_previous_depth(model, data.shape[0],
                                                                                         parent_nodes, leaf_nodes)

            logging.debug("Previous solution: ")
            for k in previous_depth_params:
                logging.debug("{0}: {1}".format(k, previous_depth_params[k]))

            if d == self.D:
                global_status = status
                global_loss = loss
                self.model = model
                self.is_trained = True
                self.l = {t: value(model.l[t]) for t in self.leaf_ndoes}
                self.c = {t: [value(model.c[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
                self.d = {t: value(model.d[t]) for t in self.parent_nodes}
                self.a = {t: [value(model.a[j, t]) for j in self.P_range] for t in self.parent_nodes}
                self.b = {t: value(model.bt[t]) for t in self.parent_nodes}
                self.Nt = {t: value(model.Nt[t]) for t in self.leaf_ndoes}
                self.Nkt = {t: [value(model.Nkt[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
                self.Lt = {t: value(model.Lt[t]) for t in self.leaf_ndoes}

        logging.info("Training done. Loss: {1}. Optimization status: {0}".format(global_status, global_loss))
        logging.info("Training done(Contd.): training accuracy: {0}".format(1 - sum(self.Lt.values()) / data.shape[0]))

    def fit(self, X, y, **kwargs):
        self.P = X.shape[1]
        self.P_range = ["x" + str(i + 1) for i in range(self.P)]
        data = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), 
                            columns=self.P_range + ["y"])
        self.train(data, train_method="ls", warm_start=False, show_training_process=False, **kwargs)         
        
    def print_tree(self, feature_names):
        all_nodes = self.parent_nodes + self.leaf_ndoes
        for depth in range(int(np.floor(np.log2(len(all_nodes) + 1)))):
            print(f'depth {depth}:')
            offset = 2 ** depth
            for t in range(offset, 2 ** (depth + 1)):
                try:
                    print('\t', feature_names[self.a[t]==1][0], '>', self.b[t])
                except:
                    print('\tnode', t, 'undefined')
            print()        
        
    @abstractmethod
    def fast_train_helper(self, train_x, train_y, tree: Tree, Nmin: int, return_tree_list):
        pass

    def fast_train(self, data: pd.DataFrame, num_initialize_trees: int = 10):
        data = self.normalize_data(data)
        self.K_range = sorted(list(set(data[self.y_col])))
        self.pool = multiprocessing.Pool()

        alpha = self.alpha
        initialize_trees = []
        for l in range(num_initialize_trees):
            tree = self.initialize_tree_for_fast_train(data, alpha)
            initialize_trees.append(tree)

        train_x = data.ix[::, self.P_range].values
        train_y = data.ix[::, self.y_col].values

        manager = multiprocessing.Manager()
        return_tree_list = manager.list()
        jobs = []
        for tree in initialize_trees:
            job = multiprocessing.Process(target=self.fast_train_helper,
                                          args=(train_x, train_y, tree, self.Nmin, return_tree_list))
            jobs.append(job)
            job.start()

        for job in jobs:
            job.join()

        logging.info("Training done.")

        min_loss_tree_index = 0
        min_loss = np.inf
        for i in range(len(return_tree_list)):
            tree = return_tree_list[i]
            loss = tree.loss(train_x, train_y)
            logging.info("Loss for tree [{0}] is {1}".format(i, loss))
            if loss < min_loss:
                min_loss = loss
                min_loss_tree_index = i

        logging.info("The final loss is {0}".format(min_loss))

        optimized_tree = return_tree_list[min_loss_tree_index]
        self.a = optimized_tree.a
        self.b = optimized_tree.b
        self.is_trained = True
        self.c = {}
        for t in optimized_tree.c:
            yt = optimized_tree.c[t]
            ct = [0 for i in self.K_range]
            index = self.K_range.index(yt)
            ct[index] = 1
            self.c[t] = ct

    def normalize_data(self, data: pd.DataFrame):
        data = data.copy().reset_index(drop=True)
        for col in self.P_range:
            col_max = max(data[col])
            col_min = min(data[col])
            self.normalizer[col] = (col_max, col_min)
            if col_max != col_min:
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 1
        return data

    def initialize_tree_for_fast_train(self, data: pd.DataFrame, alpha: float):
        # Use CART as the primary initialization method. If failed, use random initialization.
        is_cart_initialization_failed = False
        try:
            # Initialize by cart

            # Only train CART with sqrt(P) number of features
            P_range_bk = self.P_range
            np.random.shuffle(self.P_range)
            subset_features = self.P_range[0:int(np.sqrt(self.P))]
            self.P_range = subset_features
            subset_data = pd.concat([data.ix[::, subset_features], data.ix[::, self.y_col]], axis=1)
            cart_params = self._get_cart_params(subset_data, self.D)
            self.P_range = P_range_bk
            cart_a = {}
            for t in self.parent_nodes:
                at = []
                for j in self.P_range:
                    if j in subset_features:
                        at.append(cart_params["a"][j, t])
                    else:
                        at.append(0)
                cart_a[t] = np.array(at)
            cart_b = {t: cart_params["bt"][t] for t in self.parent_nodes}
            tree = WholeTree(self.D, a=cart_a, b=cart_b, alpha=alpha)
        except Exception as e:
            # Initialize randomly
            is_cart_initialization_failed = True
            tree = TreeModel(self.D, self.P, alpha=alpha)

        if is_cart_initialization_failed:
            logging.info("Initialized randomly")
        else:
            logging.info("Initialized by CART")

        return tree

    @abstractmethod
    def generate_model(self, data: pd.DataFrame, parent_nodes: list, leaf_nodes: list, warm_start_params: dict = None):
        """Generate the corresponding model instance"""
        pass

    def _get_cart_params(self, data: pd.DataFrame, depth: int):
        cart_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=self.Nmin)
        clf = cart_model.fit(data[self.P_range].values.tolist(), data[[self.y_col]].values.tolist())
        members = getmembers(clf.tree_)
        members_dict = {m[0]: m[1] for m in members}

        if members_dict["max_depth"] < depth:
            return None

        self.K_range = sorted(list(set(data[self.y_col])))
        cart_params = self._convert_skcart_to_params(members_dict)

        parent_nodes, leaf_nodes = self.generate_nodes(depth)
        z = {(i, t): 0 for i in range(0, data.shape[0]) for t in leaf_nodes}
        epsilon = self.epsilon
        for i in range(data.shape[0]):
            xi = np.array([data.ix[i, j] for j in self.P_range])
            node = 1
            current_depth = 0
            while current_depth < depth:
                current_b = xi.dot(np.array([cart_params["a"][j, node] for j in self.P_range]))
                if current_b + self.epsilon <= cart_params["bt"][node]:
                    epsilon = epsilon if cart_params["bt"][node] - current_b > epsilon else cart_params["bt"][
                                                                                                node] - current_b
                    node = 2 * node
                else:
                    node = 2 * node + 1
                current_depth += 1
                if current_depth == depth:
                    z[i, node] = 1

        self.epsilon = epsilon
        cart_params["z"] = z

        logging.debug("Cart solution: ")
        for k in cart_params:
            logging.debug("{0}: {1}".format(k, cart_params[k]))

        return cart_params

    @abstractmethod
    def _convert_skcart_to_params(self, clf: dict):
        pass

    def pprint(self):
        logging.info("l: {0}".format(self.l))
        logging.info("c: {0}".format(self.c))
        logging.info("d: {0}".format(self.d))
        logging.info("a: {0}".format(self.a))
        logging.info("b: {0}".format(self.b))
        logging.info("Lt: {0}".format(self.Lt))
        logging.info("Nkt: {0}".format(self.Nkt))
        logging.info("Nt: {0}".format(self.Nt))

    def _select_better_warm_start_params(self, params_list: list, data: pd.DataFrame):
        params_list = [p for p in params_list if p is not None]
        if len(params_list) == 0:
            return None

        n = data.shape[0]
        label = data[[self.y_col]].copy()
        label["__value__"] = 1
        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        best_params = None
        current_loss = np.inf
        for i, params in enumerate(params_list):
            loss = self._get_solution_loss(params, L_hat)
            logging.info("Loss of the {0}th warmstart solution is: {1}. The current best loss is: {2}.".format(i, loss,
                                                                                                               current_loss))
            if loss < current_loss:
                current_loss = loss
                best_params = params

        return best_params

    @abstractmethod
    def _get_solution_loss(self, params, L_hat: float):
        pass

    @abstractmethod
    def _generate_warm_start_params_from_previous_depth(self, model, n_training_data: int,
                                                        parent_nodes: list, leaf_nodes: list):
        pass

    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet! Please use `train()` to train the model first!")

        return self._feature_importance()

    @abstractmethod
    def _feature_importance(self):
        pass

    def predict(self, data):
        if not type(data) == pd.DataFrame:
            data = pd.DataFrame(data,
                                columns=["x" + str(i + 1) for i in range(data.shape[1])])
        
        if not self.is_trained:
            raise ValueError("Model has not been trained yet! Please use `train()` to train the model first!")

        new_data = data.copy()
        new_data_cols = data.columns
        for col in self.P_range:
            if col not in new_data_cols:
                raise ValueError("Column {0} is not in the given data for prediction! ".format(col))
            col_max, col_min = self.normalizer[col]
            if col_max != col_min:
                new_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                new_data[col] = 1

        prediction = []
        # loop over data points
        for j in range(new_data.shape[0]):
            # get one data point
            x = np.array([new_data.ix[j, i] for i in self.P_range])
            
            t = 1 # t is which node we are currently at
            d = 0 # depth
            
            # while not at the bottom of the tree
            while d < self.D:
                
                # a is which variable to split on
                at = np.array(self.a[t])
                
                # bt is threshold for this split
                bt = self.b[t]
                
                # go down-left
                if at.dot(x) < bt:
                    t = t * 2
                    
                # go down-right
                else:
                    t = t * 2 + 1
                    
                d = d + 1 # increase depth
            
            # c stores values
            y_hat = self.c[t]
            prediction.append(self.K_range[y_hat.index(max(y_hat))])
        return prediction

    def _parent(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have parent! "
        assert i <= 2 ** (self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2 ** (self.D + 1), i)
        return int(i / 2)

    def _ancestors(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have ancestors! "
        assert i <= 2 ** (self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2 ** (self.D + 1), i)
        left_ancestors = []
        right_ancestors = []
        j = i
        while j > 1:
            if j % 2 == 0:
                left_ancestors.append(int(j / 2))
            else:
                right_ancestors.append(int(j / 2))
            j = int(j / 2)
        return left_ancestors, right_ancestors

    @staticmethod
    def generate_nodes(tree_depth: int):
        nodes = list(range(1, int(round(2 ** (tree_depth + 1)))))
        parent_nodes = nodes[0: 2 ** (tree_depth + 1) - 2 ** tree_depth - 1]
        leaf_ndoes = nodes[-2 ** tree_depth:]
        return parent_nodes, leaf_ndoes

    @staticmethod
    def positive_or_zero(i: float):
        if i >= 0:
            return i
        else:
            return 0

    @staticmethod
    def convert_to_complete_tree(incomplete_tree: dict):
        children_left = incomplete_tree["children_left"]
        children_right = incomplete_tree["children_right"]
        depth = incomplete_tree["max_depth"]

        mapping = {1: 0}
        for t in range(2, 2 ** (depth + 1)):
            parent_of_t = int(t / 2)
            parent_in_original_tree = mapping[parent_of_t]
            is_left_child = t % 2 == 0

            if is_left_child:
                node_in_original_tree = children_left[parent_in_original_tree]
            else:
                node_in_original_tree = children_right[parent_in_original_tree]
            mapping[t] = node_in_original_tree

        return mapping

    @staticmethod
    def get_leaf_mapping(tree_nodes_mapping: dict):
        number_nodes = len(tree_nodes_mapping)
        depth = int(round(np.log2(number_nodes + 1) - 1))
        nodes = list(range(1, number_nodes + 1))
        leaf_nodes = nodes[-2 ** depth:]
        leaf_nodes_mapping = {}
        for t in leaf_nodes:
            tt = t
            while tt >= 1:
                original_t = tree_nodes_mapping[tt]
                if original_t != -1:
                    leaf_nodes_mapping[t] = original_t
                    break
                else:
                    tt = int(tt / 2)
        return leaf_nodes_mapping


class OptimalTreeModel(AbstractOptimalTreeModel):
    def fast_train_helper(self, train_x, train_y, tree: Tree, Nmin: int, return_tree_list):
        optimizer = OptimalTreeModelOptimizer(Nmin)
        optimized_tree = optimizer.local_search(tree, train_x, train_y)
        return_tree_list.append(optimized_tree)

    def generate_model(self, data: pd.DataFrame, parent_nodes: list, leaf_ndoes: list, warm_start_params: dict = None):
        model = ConcreteModel(name="OptimalTreeModel")
        n = data.shape[0]
        label = data[[self.y_col]].copy()
        label["__value__"] = 1
        Y = label.pivot(columns=self.y_col, values="__value__")

        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        Y.fillna(value=-1, inplace=True)

        n_range = range(n)
        K_range = sorted(list(set(data[self.y_col])))
        P_range = self.P_range

        self.K_range = K_range

        warm_start_params = {} if warm_start_params is None else warm_start_params

        # Variables
        model.z = Var(n_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("z"))
        model.l = Var(leaf_ndoes, within=Binary, initialize=warm_start_params.get("l"))
        model.c = Var(K_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("c"))
        model.d = Var(parent_nodes, within=Binary, initialize=warm_start_params.get("d"))
        model.a = Var(P_range, parent_nodes, within=Binary, initialize=warm_start_params.get("a"))

        model.Nt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nt"))
        model.Nkt = Var(K_range, leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nkt"))
        model.Lt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Lt"))
        model.bt = Var(parent_nodes, within=NonNegativeReals, initialize=warm_start_params.get("bt"))

        # Constraints
        model.integer_relationship_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.c[k, t] for k in K_range]) == model.l[t]
            )
        for i in n_range:
            for t in leaf_ndoes:
                model.integer_relationship_constraints.add(
                    expr=model.z[i, t] <= model.l[t]
                )
        for i in n_range:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for t in leaf_ndoes]) == 1
            )
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for i in n_range]) >= model.l[t] * self.Nmin
            )
        for t in parent_nodes:
            model.integer_relationship_constraints.add(
                expr=sum([model.a[j, t] for j in P_range]) == model.d[t]
            )
        for t in parent_nodes:
            if t != 1:
                model.integer_relationship_constraints.add(
                    expr=model.d[t] <= model.d[self._parent(t)]
                )

        model.leaf_samples_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.leaf_samples_constraints.add(
                expr=model.Nt[t] == sum([model.z[i, t] for i in n_range])
            )
        for t in leaf_ndoes:
            for k in K_range:
                model.leaf_samples_constraints.add(
                    expr=model.Nkt[k, t] == sum([model.z[i, t] * (1 + Y.loc[i, k]) / 2.0 for i in n_range])
                )

        model.leaf_error_constraints = ConstraintList()
        for k in K_range:
            for t in leaf_ndoes:
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] >= model.Nt[t] - model.Nkt[k, t] - (1 - model.c[k, t]) * n
                )
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] <= model.Nt[t] - model.Nkt[k, t] + model.c[k, t] * n
                )

        model.parent_branching_constraints = ConstraintList()
        for i in n_range:
            for t in leaf_ndoes:
                left_ancestors, right_ancestors = self._ancestors(t)
                for m in right_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) >= model.bt[m] - (1 - model.z[
                            i, t])
                    )
                for m in left_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) + self.epsilon <= model.bt[m] + (1 -
                                                                                                                     model.z[
                                                                                                                         i, t]) * (
                                                                                                                        1 + self.epsilon)
                    )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=model.bt[t] <= model.d[t]
            )

        # Objective
        model.obj = Objective(
            expr=sum([model.Lt[t] for t in leaf_ndoes]) / L_hat + sum([model.d[t] for t in parent_nodes]) * self.alpha
        )

        return model

    def _feature_importance(self):
        importance_scores = np.array([self.a[t] for t in self.a]).sum(axis=0)
        return {x: s for x, s in zip(self.P_range, importance_scores)}

    def _generate_warm_start_params_from_previous_depth(self, model, n_training_data: int,
                                                        parent_nodes: list, leaf_nodes: list):
        ret = {}
        D = int(round(np.log2(len(leaf_nodes))) + 1)
        new_parent_nodes, new_leaf_nodes = self.generate_nodes(D)
        n_range = range(n_training_data)

        ret["z"] = {(i, t): round(value(model.z[i, int(t / 2)])) if t % 2 == 1 else 0 for i in n_range for t in
                    new_leaf_nodes}
        ret["l"] = {t: round(value(model.l[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["c"] = {(k, t): round(value(model.c[k, int(t / 2)])) if t % 2 == 1 else 0 for k in self.K_range for t in
                    new_leaf_nodes}
        ret_d_1 = {t: round(value(model.d[t])) for t in parent_nodes}
        ret_d_2 = {t: 0 for t in leaf_nodes}
        ret["d"] = {**ret_d_1, **ret_d_2}
        ret_a_1 = {(j, t): round(value(model.a[j, t])) for j in self.P_range for t in parent_nodes}
        ret_a_2 = {(j, t): 0 for j in self.P_range for t in leaf_nodes}
        ret["a"] = {**ret_a_1, **ret_a_2}
        ret["Nt"] = {t: self.positive_or_zero(value(model.Nt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["Nkt"] = {(k, t): self.positive_or_zero(value(model.Nkt[k, int(t / 2)])) if t % 2 == 1 else 0 for k in
                      self.K_range for t in new_leaf_nodes}
        ret["Lt"] = {t: self.positive_or_zero(value(model.Lt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret_b_1 = {t: self.positive_or_zero(value(model.bt[t])) for t in parent_nodes}
        ret_b_2 = {t: 0 for t in leaf_nodes}
        ret["bt"] = {**ret_b_1, **ret_b_2}
        return ret

    def _convert_skcart_to_params(self, members: dict):
        complete_incomplete_nodes_mapping = self.convert_to_complete_tree(members)
        leaf_nodes_mapping = self.get_leaf_mapping(complete_incomplete_nodes_mapping)
        D = members["max_depth"]

        ret = {}
        parent_nodes, leaf_nodes = self.generate_nodes(D)

        ret["l"] = {t: self.extract_solution_l(complete_incomplete_nodes_mapping, t) for t in leaf_nodes}
        ret_c_helper = {
            t: self.extract_solution_c(self.K_range, members, t, complete_incomplete_nodes_mapping, leaf_nodes_mapping)
            for t in leaf_nodes}
        ret["c"] = {(k, t): ret_c_helper[t][kk] for kk, k in enumerate(self.K_range) for t in leaf_nodes}

        ret["d"] = {t: 1 if (complete_incomplete_nodes_mapping[t] != -1 and
                             members["children_left"][complete_incomplete_nodes_mapping[t]] != -1 and
                             members["children_right"][complete_incomplete_nodes_mapping[t]] != -1) else 0
                    for t in parent_nodes}
        ret["a"] = {(j, t): self.extract_solution_a(members, complete_incomplete_nodes_mapping, j, t) for j in
                    self.P_range for t in parent_nodes}
        ret["Nt"] = {t: self.extract_solution_Nt(members, t, complete_incomplete_nodes_mapping, leaf_nodes_mapping) for
                     t in leaf_nodes}
        ret["Nkt"] = {
            (k, t): self.extract_solution_Nkt(members, t, kk, complete_incomplete_nodes_mapping, leaf_nodes_mapping)
            for kk, k in enumerate(self.K_range) for t in leaf_nodes}
        ret["Lt"] = {
            t: OptimalTreeModel.extract_solution_Lt(members, t, complete_incomplete_nodes_mapping, leaf_nodes_mapping)
            for t in leaf_nodes}
        ret["bt"] = {t: 0 if members["threshold"][complete_incomplete_nodes_mapping[t]] <= 0 else
        members["threshold"][complete_incomplete_nodes_mapping[t]] for t in parent_nodes}

        return ret

    def _get_solution_loss(self, params: dict, L_hat: float):
        return sum(params["Lt"].values()) / L_hat + self.alpha * sum(params["d"].values())

    def extract_solution_a(self, members: dict, nodes_mapping: dict, j: str, t: int):
        if nodes_mapping[t] == -1:
            return 0

        feature = members["feature"][nodes_mapping[t]]
        if feature < 0:
            return 0

        if self.P_range[feature] == j:
            return 1
        else:
            return 0

    @staticmethod
    def extract_solution_l(nodes_mapping: dict, t: int):
        if nodes_mapping[t] >= 0:
            return 1

        p = t
        while p > 1:
            pp = int(p / 2)
            if nodes_mapping[pp] >= 0:
                if p % 2 == 1:
                    return 1
                else:
                    return 0
            else:
                if p % 2 == 1:
                    p = pp
                else:
                    return 0

    @staticmethod
    def extract_solution_c(K_range, members: dict, t: int, nodes_mapping: dict, leaf_nodes_mapping: dict):
        samples_count_in_the_node = np.array(members["value"][leaf_nodes_mapping[t]][0])
        max_class = max(samples_count_in_the_node)

        ret = []
        for s in samples_count_in_the_node:
            if s == max_class:
                ret.append(1)
                max_class += 1
            else:
                ret.append(0)

        if nodes_mapping[t] >= 0:
            return ret

        p = t
        while p > 1:
            pp = int(p / 2)
            if nodes_mapping[pp] >= 0:
                if p % 2 == 1:
                    return ret
                else:
                    return [0 for i in K_range]
            else:
                if p % 2 == 1:
                    p = pp
                else:
                    return [0 for i in K_range]

    @staticmethod
    def extract_solution_Nt(members: dict, t: int, nodes_mapping: dict, leaf_nodes_mapping: dict):
        if nodes_mapping[t] >= 0:
            return members["n_node_samples"][nodes_mapping[t]]

        p = t
        while p > 1:
            pp = int(p / 2)
            if nodes_mapping[pp] >= 0:
                if p % 2 == 1:
                    return members["n_node_samples"][leaf_nodes_mapping[t]]
                else:
                    return 0
            else:
                if p % 2 == 1:
                    p = pp
                else:
                    return 0

    @staticmethod
    def extract_solution_Nkt(members: dict, t: int, kk: int, nodes_mapping: dict, leaf_nodes_mapping: dict):
        if nodes_mapping[t] >= 0:
            return members["value"][nodes_mapping[t]][0][kk]

        p = t
        while p > 1:
            pp = int(p / 2)
            if nodes_mapping[pp] >= 0:
                if p % 2 == 1:
                    return members["value"][leaf_nodes_mapping[t]][0][kk]
                else:
                    return 0
            else:
                if p % 2 == 1:
                    p = pp
                else:
                    return 0

    @staticmethod
    def extract_solution_Lt(members: dict, t, nodes_mapping: dict, leaf_nodes_mapping: dict):
        samples_count_in_the_node = np.array(members["value"][leaf_nodes_mapping[t]][0])
        max_class = max(samples_count_in_the_node)

        n_max_count = 0
        for c in samples_count_in_the_node:
            if c == max_class:
                n_max_count += 1

        if n_max_count == 1:
            ret = sum([s for s in samples_count_in_the_node if s != max_class])
        else:
            ret = sum([s for s in samples_count_in_the_node if s != max_class]) + max_class

        if nodes_mapping[t] >= 0:
            return ret

        p = t
        while p > 1:
            pp = int(p / 2)
            if nodes_mapping[pp] >= 0:
                if p % 2 == 1:
                    return ret
                else:
                    return 0
            else:
                if p % 2 == 1:
                    p = pp
                else:
                    return 0

    @staticmethod
    def extract_solution_bt(members: dict, t: int, nodes_mapping: dict):
        original_node = nodes_mapping[t]
        children_left = members["children_left"]

        if original_node == -1 or children_left[original_node] == -1:
            return 0

        return abs(members["threshold"][original_node])


class OptimalHyperTreeModel(AbstractOptimalTreeModel):
    def __init__(self, x_cols: list, y_col: str, tree_depth: int, N_min: int, alpha: float = 0,
                 num_random_tree_restart: int = 2, M: int = 10, epsilon: float = 1e-4,
                 solver_name: str = "gurobi"):
        self.H = num_random_tree_restart
        super(OptimalHyperTreeModel, self).__init__(x_cols, y_col, tree_depth, N_min, alpha, M, epsilon, solver_name)       
        
    def train(self, data: pd.DataFrame, train_method: str = "ls",
              show_training_process: bool = True, warm_start: bool = True,
              num_initialize_trees: int = 1):
        super(OptimalHyperTreeModel, self).train(data, train_method, show_training_process,
                                                 warm_start, num_initialize_trees)

    def fast_train_helper(self, train_x, train_y, tree: Tree, Nmin: int, return_tree_list):
        optimizer = OptimalHyperTreeModelOptimizer(Nmin, self.H)
        optimized_tree = optimizer.local_search(tree, train_x, train_y)
        return_tree_list.append(optimized_tree)

    def generate_model(self, data: pd.DataFrame, parent_nodes: list, leaf_ndoes: list, warm_start_params: dict = None):
        model = ConcreteModel(name="OptimalTreeModel")
        n = data.shape[0]
        label = data[[self.y_col]].copy()
        label["__value__"] = 1
        Y = label.pivot(columns=self.y_col, values="__value__")

        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        Y.fillna(value=-1, inplace=True)

        n_range = range(n)
        K_range = sorted(list(set(data[self.y_col])))
        P_range = self.P_range

        self.K_range = K_range

        warm_start_params = {} if warm_start_params is None else warm_start_params

        # Variables
        model.z = Var(n_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("z"))
        model.l = Var(leaf_ndoes, within=Binary, initialize=warm_start_params.get("l"))
        model.c = Var(K_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("c"))
        model.d = Var(parent_nodes, within=Binary, initialize=warm_start_params.get("d"))
        model.s = Var(P_range, parent_nodes, within=Binary, initialize=warm_start_params.get("s"))

        model.Nt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nt"))
        model.Nkt = Var(K_range, leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nkt"))
        model.Lt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Lt"))
        model.a = Var(P_range, parent_nodes, initialize=warm_start_params.get("a"))
        model.bt = Var(parent_nodes, initialize=warm_start_params.get("bt"))
        model.a_hat_jt = Var(P_range, parent_nodes, within=NonNegativeReals,
                             initialize=warm_start_params.get("a_hat_jt"))

        # Constraints
        model.integer_relationship_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.c[k, t] for k in K_range]) == model.l[t]
            )
        for i in n_range:
            for t in leaf_ndoes:
                model.integer_relationship_constraints.add(
                    expr=model.z[i, t] <= model.l[t]
                )
        for i in n_range:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for t in leaf_ndoes]) == 1
            )
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for i in n_range]) >= model.l[t] * self.Nmin
            )
        for j in P_range:
            for t in parent_nodes:
                model.integer_relationship_constraints.add(
                    expr=model.s[j, t] <= model.d[t]
                )
        for t in parent_nodes:
            model.integer_relationship_constraints.add(
                expr=sum([model.s[j, t] for j in P_range]) >= model.d[t]
            )
        for t in parent_nodes:
            if t != 1:
                model.integer_relationship_constraints.add(
                    expr=model.d[t] <= model.d[self._parent(t)]
                )

        model.leaf_samples_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.leaf_samples_constraints.add(
                expr=model.Nt[t] == sum([model.z[i, t] for i in n_range])
            )
        for t in leaf_ndoes:
            for k in K_range:
                model.leaf_samples_constraints.add(
                    expr=model.Nkt[k, t] == sum([model.z[i, t] * (1 + Y.loc[i, k]) / 2.0 for i in n_range])
                )

        model.leaf_error_constraints = ConstraintList()
        for k in K_range:
            for t in leaf_ndoes:
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] >= model.Nt[t] - model.Nkt[k, t] - (1 - model.c[k, t]) * n
                )
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] <= model.Nt[t] - model.Nkt[k, t] + model.c[k, t] * n
                )

        model.parent_branching_constraints = ConstraintList()
        for i in n_range:
            for t in leaf_ndoes:
                left_ancestors, right_ancestors = self._ancestors(t)
                for m in right_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) >= model.bt[m] - (1 - model.z[
                            i, t]) * self.M
                    )
                for m in left_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) + self.epsilon <= model.bt[m] + (
                                                                                                                        1 -
                                                                                                                        model.z[
                                                                                                                            i, t]) * (
                                                                                                                        self.M + self.epsilon)
                    )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=sum([model.a_hat_jt[j, t] for j in P_range]) <= model.d[t]
            )
        for j in P_range:
            for t in parent_nodes:
                model.parent_branching_constraints.add(
                    expr=model.a_hat_jt[j, t] >= model.a[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a_hat_jt[j, t] >= -model.a[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a[j, t] >= -model.s[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a[j, t] <= model.s[j, t]
                )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=model.bt[t] >= -model.d[t]
            )
            model.parent_branching_constraints.add(
                expr=model.bt[t] <= model.d[t]
            )

        # Objective
        model.obj = Objective(
            expr=sum([model.Lt[t] for t in leaf_ndoes]) / L_hat + sum(
                [model.a_hat_jt[j, t] for j in P_range for t in parent_nodes]) * self.alpha
        )

        return model

    def _feature_importance(self):
        importance_scores = np.array(
            [[value(self.model.s[j, t]) * value(self.model.a_hat_jt[j, t]) for j in self.P_range] for t in
             self.parent_nodes]).sum(axis=0)
        return {x: s for x, s in zip(self.P_range, importance_scores)}

    def _generate_warm_start_params_from_previous_depth(self, model, n_training_data: int,
                                                        parent_nodes: list, leaf_nodes: list):
        ret = {}
        D = int(round(np.log2(len(leaf_nodes))) + 1)
        new_parent_nodes, new_leaf_nodes = self.generate_nodes(D)
        n_range = range(n_training_data)

        ret["z"] = {(i, t): round(value(model.z[i, int(t / 2)])) if t % 2 == 1 else 0 for i in n_range for t in
                    new_leaf_nodes}
        ret["l"] = {t: round(value(model.l[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["c"] = {(k, t): round(value(model.c[k, int(t / 2)])) if t % 2 == 1 else 0 for k in self.K_range for t in
                    new_leaf_nodes}
        ret_d_1 = {t: round(value(model.d[t])) for t in parent_nodes}
        ret_d_2 = {t: 0 for t in leaf_nodes}
        ret["d"] = {**ret_d_1, **ret_d_2}
        ret_s_1 = {(j, t): round(value(model.s[j, t])) for j in self.P_range for t in parent_nodes}
        ret_s_2 = {(j, t): 0 for j in self.P_range for t in leaf_nodes}
        ret["s"] = {**ret_s_1, **ret_s_2}
        ret["Nt"] = {t: self.positive_or_zero(value(model.Nt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["Nkt"] = {(k, t): self.positive_or_zero(value(model.Nkt[k, int(t / 2)])) if t % 2 == 1 else 0 for k in
                      self.K_range for t in new_leaf_nodes}
        ret["Lt"] = {t: self.positive_or_zero(value(model.Lt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret_a_1 = {(j, t): value(model.a[j, t]) for j in self.P_range for t in parent_nodes}
        ret_a_2 = {(j, t): 0 for j in self.P_range for t in leaf_nodes}
        ret["a"] = {**ret_a_1, **ret_a_2}
        ret_b_1 = {t: value(model.bt[t]) for t in parent_nodes}
        ret_b_2 = {t: 0 for t in leaf_nodes}
        ret["bt"] = {**ret_b_1, **ret_b_2}
        ret["a_hat_jt"] = {(j, t): abs(ret["a"][j, t]) for (j, t) in ret["a"]}
        return ret

    def _convert_skcart_to_params(self, members: dict):
        complete_incomplete_nodes_mapping = self.convert_to_complete_tree(members)
        leaf_nodes_mapping = self.get_leaf_mapping(complete_incomplete_nodes_mapping)
        D = members["max_depth"]

        logging.debug("Children left of cart: {0}".format(members["children_left"]))
        logging.debug("Children right of cart: {0}".format(members["children_right"]))

        ret = {}
        parent_nodes, leaf_nodes = self.generate_nodes(D)

        ret["l"] = {t: OptimalTreeModel.extract_solution_l(complete_incomplete_nodes_mapping, t) for t in leaf_nodes}
        ret_c_helper = {
            t: OptimalTreeModel.extract_solution_c(self.K_range, members, t, complete_incomplete_nodes_mapping,
                                                   leaf_nodes_mapping)
            for t in leaf_nodes}
        ret["c"] = {(k, t): ret_c_helper[t][kk] for kk, k in enumerate(self.K_range) for t in leaf_nodes}

        ret["d"] = {t: 1 if (complete_incomplete_nodes_mapping[t] >= 0 and
                             members["children_left"][complete_incomplete_nodes_mapping[t]] >= 0) else 0
                    for t in parent_nodes}
        ret["s"] = {(j, t): self.extract_solution_s(members, complete_incomplete_nodes_mapping, j, t) for j in
                    self.P_range
                    for t in parent_nodes}
        ret["Nt"] = {
            t: OptimalTreeModel.extract_solution_Nt(members, t, complete_incomplete_nodes_mapping, leaf_nodes_mapping)
            for
            t in leaf_nodes}
        ret["Nkt"] = {(k, t): OptimalTreeModel.extract_solution_Nkt(members, t, kk, complete_incomplete_nodes_mapping,
                                                                    leaf_nodes_mapping)
                      for kk, k in enumerate(self.K_range) for t in leaf_nodes}
        ret["Lt"] = {
            t: OptimalTreeModel.extract_solution_Lt(members, t, complete_incomplete_nodes_mapping, leaf_nodes_mapping)
            for t in leaf_nodes}
        ret["a"] = ret["s"]
        ret["a_hat_jt"] = ret["s"]
        ret["bt"] = {t: OptimalTreeModel.extract_solution_bt(members, t, complete_incomplete_nodes_mapping) for t in
                     parent_nodes}

        return ret

    def _get_solution_loss(self, params: dict, L_hat: float):
        return sum(params["Lt"].values()) / L_hat + self.alpha * sum(params["a_hat_jt"].values())

    def extract_solution_s(self, members: dict, nodes_mapping: dict, j: str, t: int):
        children_left = members["children_left"]

        if nodes_mapping[t] == -1 or children_left[nodes_mapping[t]] < 0:
            return 0

        feature = members["feature"][nodes_mapping[t]]
        if feature < 0:
            return 0

        if self.P_range[feature] == j:
            return 1
        else:
            return 0



if __name__ == "__main__":
    data = pd.DataFrame({
        "index": ['A', 'C', 'D', 'E', 'F'],
        "x1": [1, 2, 2, 2, 3],
        "x2": [1, 2, 1, 0, 1],
        "y": [1, 1, -1, -1, -1]
    })
    test_data = pd.DataFrame({
        "index": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        "x1": [1, 1, 2, 2, 2, 3, 3],
        "x2": [1, 2, 2, 1, 0, 1, 0],
        "y": [1, 1, 1, -1, -1, -1, -1]
    })
    model = OptimalHyperTreeModel(["x1", "x2"], "y", tree_depth=2, N_min=1, alpha=0.1)
    model.train(data)

    print(model.predict(test_data))
