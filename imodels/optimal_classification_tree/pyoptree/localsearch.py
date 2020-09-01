import numpy as np
from abc import abstractmethod, ABCMeta
from .tree import Tree, TreeModel
# from pyoptree.tree import Tree, TreeModel
import logging
import random
import multiprocessing
from tqdm import tqdm

# compress "Invalid Value Encountered Error "
np.seterr(divide='ignore', invalid='ignore')


class AbstractOptimalTreeModelOptimizer(metaclass=ABCMeta):
    def __init__(self, Nmin: int):
        self.Nmin = Nmin
        self.sorted_x = None
        self.pool = multiprocessing.Pool()

    @staticmethod
    def shuffle(index_set: list):
        index_set_bk = index_set.copy()
        np.random.shuffle(index_set_bk)
        return index_set_bk

    @staticmethod
    def sort_x(x):
        res = []
        for j in range(x.shape[1]):
            xj = x[::, j]
            res.append(sorted(xj))
        res = np.array(res)
        return res.T

    def local_search(self, tree: Tree, x, y):
        self.sorted_x = self.sort_x(x)
        tree = tree.copy()
        error_previous = tree.loss(x, y)
        error_current = np.inf

        logging.info("Current error of the whole tree: {0}".format(error_previous))
        i = 1
        while True:
            for t in AbstractOptimalTreeModelOptimizer.shuffle(tree.get_parent_nodes()):
                subtree = tree.subtree(t)
                res = tree.evaluate(x)
                L = []
                for tt in subtree.get_leaf_nodes():
                    L.extend(res[tt])
                if len(L) > 0:
                    logging.info("Visiting node {0}, there are {1} data points in this subtree.".format(t, len(L)))
                    logging.info("Training in {0}th iteration...".format(i))
                    i += 1
                    new_subtree = self.optimize_node(subtree, x, y, L)
                    tree.a[t] = new_subtree.a[t]
                    tree.b[t] = new_subtree.b[t]

                    error_current = tree.loss(x, y)

                    logging.info("Current error of the whole tree: {0}".format(error_current))

            if round(error_current, 5) == round(error_previous, 5):
                tree.generate_majority_leaf_class(x, y)
                return tree

            error_previous = error_current

    @abstractmethod
    def optimize_node(self, subtree: Tree, x, y, L):
        pass

    @abstractmethod
    def best_split(self, lower_tree: Tree, upper_tree: Tree, x, y, L: list):
        pass

    @staticmethod
    def _check_best_split_input(lower_tree: Tree, upper_tree: Tree):
        assert lower_tree.root_node % 2 == 0, "Illegal lower tree! (root node: {0})".format(lower_tree.root_node)
        assert upper_tree.root_node == lower_tree.root_node + 1, "Illegal upper tree! (lower tree root node: {0}, " \
                                                                 "upper tree root node: {1})".format(
            lower_tree.root_node,
            upper_tree.root_node)
        assert lower_tree.depth == upper_tree.depth, "Unequal depth of lower tree and upper tree! ({0} != {1})".format(
            lower_tree.depth, upper_tree.depth
        )


class OptimalTreeModelOptimizer(AbstractOptimalTreeModelOptimizer):
    def optimize_node(self, subtree: Tree, x, y, L):
        new_sub_tree = subtree.copy()
        sub_x = x[L, ::]
        sub_y = y[L]

        p = sub_x.shape[1]

        lower_tree, upper_tree = new_sub_tree.children()
        error_best = new_sub_tree.loss(sub_x, sub_y)

        logging.debug("Current best error of the subtree: {0}".format(error_best))

        updated = False

        error_lower = lower_tree.loss(sub_x, sub_y)
        if error_lower < error_best and lower_tree.depth > 0:
            logging.info("Updated by replacing by lower child tree")
            new_sub_tree.a[new_sub_tree.root_node] = np.zeros(p)
            new_sub_tree.b[new_sub_tree.root_node] = 1
            error_best = error_lower
            updated = True
            return new_sub_tree

        error_upper = upper_tree.loss(sub_x, sub_y)
        if error_upper < error_best and upper_tree.depth > 0:
            logging.info("Updated by replacing by upper child tree")
            new_sub_tree.a[new_sub_tree.root_node] = np.zeros(p)
            new_sub_tree.b[new_sub_tree.root_node] = 0
            error_best = error_upper
            updated = True
            return new_sub_tree

        para_tree, error_para = self.best_split(lower_tree, upper_tree, sub_x, sub_y, L)
        error_para = para_tree.loss(sub_x, sub_y)
        if error_para < error_best:
            logging.info("Updated by parallel split")
            new_sub_tree.a[new_sub_tree.root_node] = para_tree.a[para_tree.root_node]
            new_sub_tree.b[new_sub_tree.root_node] = para_tree.b[para_tree.root_node]
            error_best = error_para
            updated = True

        if not updated:
            logging.info("No update, return the original tree")

        return new_sub_tree

    @staticmethod
    def parallel_split_criteria_scan(start: int, end: int, values, parent_tree: Tree, parent_node: int,
                                     x, y, Nmin: int):
        best_error = np.inf
        best_b = None

        for i in range(start, end):
            b = (values[i] + values[i + 1]) / 2
            parent_tree.b[parent_node] = b
            error, min_leaf_size = parent_tree.loss_and_min_leaf_size(x, y)
            if min_leaf_size >= Nmin:
                if error < best_error:
                    best_error = error
                    best_b = b

        return {"error": best_error, "b": best_b}

    def best_split(self, lower_tree: Tree, upper_tree: Tree, x, y, L: list):
        self._check_best_split_input(lower_tree, upper_tree)

        n, p = x.shape
        error_best = np.inf

        parent_node = int(round(lower_tree.root_node / 2))
        parent_node_a = np.zeros(p)
        parent_a = {**{parent_node: parent_node_a}, **lower_tree.a, **upper_tree.a}
        parent_b = {**{parent_node: 0}, **lower_tree.b, **upper_tree.b}
        parent_tree = Tree(parent_node, lower_tree.depth + 1, parent_a, parent_b)
        best_tree = parent_tree.copy()

        logging.debug("Calculating best parallel split for {0} points with dimension {1}".format(n, p))
        sorted_sub_x = self.sorted_x[L, ::]
        cpu_count = multiprocessing.cpu_count()
        num_jobs = cpu_count * 3
        chunk_size = int(n / num_jobs)

        for j in tqdm(self.shuffle([i for i in range(p)])):
            logging.debug("Visiting {0}th dimension. Current best error of the subtree: {1}".format(j, error_best))
            values = sorted_sub_x[::, j]
            parent_tree.a[parent_node] = np.zeros(p)
            parent_tree.a[parent_node][j] = 1

            return_list = []

            for chunk_number in range(num_jobs):
                start = chunk_size * chunk_number
                end = min(chunk_size * (chunk_number + 1), n - 1)
                return_list.append(self.pool.apply_async(OptimalTreeModelOptimizer.parallel_split_criteria_scan,
                                                         args=(start, end, values, parent_tree, parent_node, x, y,
                                                               self.Nmin)))

            return_list = [res.get() for res in return_list]

            for res in return_list:
                error = res["error"]
                b = res["b"]
                if error < error_best:
                    error_best = error
                    best_tree.a[parent_node] = np.zeros(p)
                    best_tree.a[parent_node][j] = 1
                    best_tree.b[parent_node] = b

        logging.debug("Complete calculating best parallel split")

        return best_tree, error_best


class OptimalHyperTreeModelOptimizer(OptimalTreeModelOptimizer):
    def __init__(self, Nmin: int, num_random_tree_restart: int = 4):
        self.H = num_random_tree_restart
        super(OptimalHyperTreeModelOptimizer, self).__init__(Nmin)

    def best_split(self, lower_tree: Tree, upper_tree: Tree, x, y, L: list):
        self.static_best_split(self.Nmin, lower_tree, upper_tree, x, y, L)

    def optimize_node(self, subtree: Tree, x, y, L):
        new_sub_tree = subtree.copy()
        sub_x = x[L, ::]
        sub_y = y[L]

        p = sub_x.shape[1]

        lower_tree, upper_tree = new_sub_tree.children()
        error_best = new_sub_tree.loss(sub_x, sub_y)

        logging.debug("Current best error of the subtree: {0}".format(error_best))

        updated = False

        error_lower = lower_tree.loss(sub_x, sub_y)
        if error_lower < error_best and lower_tree.depth > 0:
            logging.info("Updated by replacing by lower child tree")
            new_sub_tree.a[new_sub_tree.root_node] = np.zeros(p)
            new_sub_tree.b[new_sub_tree.root_node] = 1
            error_best = error_lower
            updated = True
            return new_sub_tree

        error_upper = upper_tree.loss(sub_x, sub_y)
        if error_upper < error_best and upper_tree.depth > 0:
            logging.info("Updated by replacing by upper child tree")
            new_sub_tree.a[new_sub_tree.root_node] = np.zeros(p)
            new_sub_tree.b[new_sub_tree.root_node] = 0
            error_best = error_upper
            updated = True
            return new_sub_tree

        para_tree, error_para = super(OptimalHyperTreeModelOptimizer, self).best_split(lower_tree, upper_tree, sub_x,
                                                                                       sub_y, L)
        error_para = para_tree.loss(sub_x, sub_y)
        if error_para < error_best:
            logging.info("Updated by parallel split")
            new_sub_tree.a[new_sub_tree.root_node] = para_tree.a[para_tree.root_node]
            new_sub_tree.b[new_sub_tree.root_node] = para_tree.b[para_tree.root_node]
            error_best = error_para
            updated = True
            return new_sub_tree

        for h in range(self.H):
            res = self.parallel_random_tree_restart(h, self.Nmin, lower_tree, upper_tree, x, y,
                                                    L, sub_x, sub_y)
            if res["error"] < error_best:
                logging.info("Updated by hyperplane split")
                new_sub_tree.a[new_sub_tree.root_node] = res["a"]
                new_sub_tree.b[new_sub_tree.root_node] = res["b"]
                error_best = res["error"]
                updated = True
                return new_sub_tree

        if not updated:
            logging.info("No update, return the original tree")

        return new_sub_tree

    def parallel_random_tree_restart(self, h: int, Nmin: int, lower_tree: Tree, upper_tree: Tree, x, y, L: list, sub_x,
                                     sub_y):
        logging.info("Randomly restarting tree {0}".format(h))
        hyper_tree, error_hyper = self.static_best_split(Nmin, lower_tree, upper_tree,
                                                         sub_x, sub_y, L)
        error_hyper = hyper_tree.loss(sub_x, sub_y)
        return {"error": error_hyper, "a": hyper_tree.a[hyper_tree.root_node],
                "b": hyper_tree.b[hyper_tree.root_node]}

    @staticmethod
    def parallel_u_scan(start: int, end: int, values, parent_tree: Tree, parent_node: int, j: int, x, y,
                        Nmin: int):
        best_c = None
        best_error = np.inf
        for i in range(start, end):
            c = (values[i] + values[i + 1]) / 2
            # c = max(c, -1)
            # c = min(c, 1)
            parent_tree.a[parent_node][j] = c
            error, min_leaf_size = parent_tree.loss_and_min_leaf_size(x, y)
            if min_leaf_size >= Nmin:
                if error < best_error:
                    best_error = error
                    best_c = c
        return {"error": best_error, "c": best_c}

    @staticmethod
    def parallel_w_scan(start: int, end: int, values, parent_tree: Tree, parent_node: int, j: int, x, y,
                        Nmin: int):
        best_b = None
        best_error = np.inf
        for i in range(start, end):
            b = (values[i] + values[i + 1]) / 2
            parent_tree.a[parent_node][j] = 0
            parent_tree.b[parent_node] = b
            error, min_leaf_size = parent_tree.loss_and_min_leaf_size(x, y)
            if min_leaf_size >= Nmin:
                if error < best_error:
                    best_error = error
                    best_b = b
        return {"error": best_error, "b": best_b}

    def static_best_split(self, Nmin, lower_tree: Tree, upper_tree: Tree, x, y, L: list):
        AbstractOptimalTreeModelOptimizer._check_best_split_input(lower_tree, upper_tree)

        n, p = x.shape

        parent_node = int(round(lower_tree.root_node / 2))
        parent_node_a = np.random.rand(p) - 0.5
        parent_node_b = random.random() - 0.5
        parent_a = {**{parent_node: parent_node_a}, **lower_tree.a, **upper_tree.a}
        parent_b = {**{parent_node: parent_node_b}, **lower_tree.b, **upper_tree.b}
        parent_tree = Tree(parent_node, lower_tree.depth + 1, parent_a, parent_b)
        best_tree = parent_tree.copy()

        error_previous = best_tree.loss(x, y)
        error_best = error_previous

        parameter_updated = True

        logging.debug("Calculating best hyperplane split for {0} points with dimension {1}".format(n, p))

        while True:
            for j in tqdm(AbstractOptimalTreeModelOptimizer.shuffle([i for i in range(p)])):
                logging.debug("Visiting {0}th dimension. Current best error of the subtree: {1}".format(j, error_best))

                # Calculate V and U
                if parameter_updated:
                    vi = x.dot(parent_tree.a[parent_node].T) - parent_tree.b[parent_node]
                    uik = np.zeros([n, p])
                    for ii in range(n):
                        for kk in range(p):
                            if x[ii, kk] > 1e-5:
                                uik[ii, kk] = float(parent_tree.a[parent_node][kk] * x[ii, kk] - vi[ii]) / x[ii, kk]
                            elif vi[ii] >= 0:
                                uik[ii, kk] = -np.inf
                            else:
                                uik[ii, kk] = np.inf
                    parameter_updated = False
                    values = sorted(uik[::, j])

                # Scan U in parallel
                cpu_count = multiprocessing.cpu_count()
                num_jobs = cpu_count * 3
                chunk_size = int(n / num_jobs)

                return_list = []
                for chunk_number in range(num_jobs):
                    start = chunk_size * chunk_number
                    end = min(chunk_size * (chunk_number + 1), n - 1)
                    return_list.append(self.pool.apply_async(OptimalHyperTreeModelOptimizer.parallel_u_scan,
                                                             args=(
                                                                 start, end, values, parent_tree, parent_node, j, x, y,
                                                                 Nmin)))

                return_list = [res.get() for res in return_list]

                for res in return_list:
                    error = res["error"]
                    c = res["c"]
                    if error < error_best:
                        error_best = error
                        best_tree.a[parent_node][j] = c
                        parameter_updated = True

                parent_tree.a[parent_node] = best_tree.a[parent_node].copy()

                # Scan W in parallel
                if best_tree.a[parent_node][j] != 0:
                    wik = vi.reshape([n, 1]) + parent_tree.b[parent_node] - x * parent_tree.a[parent_node]

                    values = sorted(wik[::, j])

                    return_list = []

                    for chunk_number in range(num_jobs):
                        start = chunk_size * chunk_number
                        end = min(chunk_size * (chunk_number + 1), n - 1)
                        return_list.append(self.pool.apply_async(OptimalHyperTreeModelOptimizer.parallel_w_scan,
                                                                 args=(
                                                                     start, end, values, parent_tree, parent_node, j, x,
                                                                     y,
                                                                     Nmin)))

                    return_list = [res.get() for res in return_list]

                    for res in return_list:
                        error = res["error"]
                        b = res["b"]
                        if error < error_best:
                            error_best = error
                            best_tree.a[parent_node][j] = 0
                            best_tree.b[parent_node] = b
                            parameter_updated = True

                    parent_tree.a[parent_node] = best_tree.a[parent_node].copy()
                    parent_tree.b[parent_node] = best_tree.b[parent_node]

            # Update current best error or stop the iteration
            if round(error_previous, 5) == round(error_best, 5):
                logging.debug("Complete calculating best parallel split")
                return best_tree, error_best

            if error_previous > error_best:
                error_previous = error_best
