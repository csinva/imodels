from abc import abstractmethod, ABCMeta
import numpy as np
from collections import Counter
import random


class Tree(metaclass=ABCMeta):
    def __init__(self, root_node: int, depth: int, a: dict, b: dict, alpha: float = 0.0):
        self.root_node = root_node
        self.depth = depth
        self.a = a
        self.b = b
        self.c = None
        self.alpha = alpha

        assert depth >= 0, "Tree's depth should be non-negative! (depth: {0})".format(depth)
        for node in self.get_parent_nodes():

            if node not in a:
                print(root_node, depth)
                raise ValueError("The given `a` doesn't contain node {0}!".format(node))
            if node not in b:
                raise ValueError("The given `b` doesn't contain node {0}!".format(node))

    def subtree(self, root_node: int):
        this_tree_parent_depth = int(np.ceil(np.log2(self.root_node + 1))) - 1
        whole_depth = this_tree_parent_depth + self.depth

        parent_depth = int(np.ceil(np.log2(root_node + 1))) - 1

        if self.depth == 0:
            raise ValueError("The current tree contains only one leaf node. Cannot create subtree for leaf node! ")

        subtree_empty = Tree(root_node, whole_depth - parent_depth, self.a, self.b)
        subtree_a = {}
        subtree_b = {}
        for left_node in subtree_empty.get_parent_nodes():
            subtree_a[left_node] = self.a[left_node].copy()
            subtree_b[left_node] = self.b[left_node]

        return Tree(root_node, whole_depth - parent_depth, subtree_a, subtree_b, alpha=self.alpha)

    def get_nodes(self):
        nodes = [self.root_node]
        previous_depth_nodes = [self.root_node]
        for d in range(1, self.depth + 1):
            new_previous_depth_nodes = []
            for current_node in previous_depth_nodes:
                left_node = current_node * 2
                right_node = current_node * 2 + 1
                nodes.append(left_node)
                nodes.append(right_node)
                new_previous_depth_nodes.append(left_node)
                new_previous_depth_nodes.append(right_node)
            previous_depth_nodes = new_previous_depth_nodes
        return nodes

    def get_leaf_nodes(self):
        parent_depth = int(np.ceil(np.log2(self.root_node + 1))) - 1
        whole_depth = parent_depth + self.depth
        nodes = self.get_nodes()
        lower_node_index = 2 ** whole_depth
        return [node for node in nodes if node >= lower_node_index]

    def get_parent_nodes(self):
        parent_depth = int(np.ceil(np.log2(self.root_node + 1))) - 1
        whole_depth = parent_depth + self.depth
        nodes = self.get_nodes()
        lower_node_index = 2 ** whole_depth
        return [node for node in nodes if node < lower_node_index]

    def children(self):
        if self.depth == 0:
            raise ValueError("The current tree contains only one leaf node. Cannot create children for leaf node! ")

        return self.subtree(self.root_node * 2), self.subtree(self.root_node * 2 + 1)

    def min_leaf_size(self, x):
        fake_y = np.zeros(x.shape[0])
        loss, min_leaf_size = self.loss_and_min_leaf_size(x, fake_y)
        return min_leaf_size

    def evaluate(self, x):
        n = x.shape[0]
        leaf_samples_mapping = {t: [] for t in self.get_leaf_nodes()}
        for i in range(n):
            current_x = x[i, ::]
            t = self.root_node
            d = 1
            while d < self.depth + 1:
                at = self.a[t]
                bt = self.b[t]
                if at.dot(current_x) < bt:
                    t = t * 2
                else:
                    t = t * 2 + 1
                d = d + 1
            leaf_samples_mapping[t].append(i)
        return leaf_samples_mapping

    def loss(self, x, y):
        loss, min_leaf_size = self.loss_and_min_leaf_size(x, y)
        return loss

    def loss_and_min_leaf_size(self, x, y):
        assert x.shape[0] == y.shape[0], "Number of rows of x should be equal to length of y! ({0} != {1})".format(
            x.shape[0], y.shape[0]
        )
        res = self.evaluate(x)
        return self.loss_and_min_leaf_size_helper(res, y)

    def loss_and_min_leaf_size_helper(self, res: dict, y):
        predict_y = np.zeros(y.shape[0])
        predict_leaf_value = {t: 0 for t in self.get_leaf_nodes()}
        for t in res:
            x_indices_this_node = res[t]
            if len(x_indices_this_node) > 0:
                true_y_this_node = [y[i] for i in x_indices_this_node]
                occurrence_count = Counter(true_y_this_node)
                label_this_node = sorted(occurrence_count.items(), key=lambda x: x[1], reverse=True)[0][0]
                predict_leaf_value[t] = label_this_node
                for i in x_indices_this_node:
                    predict_y[i] = label_this_node

        tree_complexity = 0.0 if len(self.get_parent_nodes()) == 0 else sum(
            [sum([1 if a != 0 else 0 for a in self.a[t]]) for t in self.get_parent_nodes()]) / float(
            len(self.get_parent_nodes()))

        loss = sum([1 if y[i] != predict_y[i] else 0 for i in range(y.shape[0])]) / y.shape[
            0] + self.alpha * tree_complexity

        leaf_samples_count = {t: len(res[t]) for t in res}
        min_leaf_size = min([i for i in leaf_samples_count.values() if i > 0])
        return loss, min_leaf_size

    def generate_majority_leaf_class(self, x, y):
        assert x.shape[0] == y.shape[0], "Number of rows of x should be equal to length of y! ({0} != {1})".format(
            x.shape[0], y.shape[0]
        )
        res = self.evaluate(x)
        predict_leaf_value = {t: 0 for t in self.get_leaf_nodes()}
        for t in res:
            x_indices_this_node = res[t]
            if len(x_indices_this_node) > 0:
                true_y_this_node = [y[i] for i in x_indices_this_node]
                occurrence_count = Counter(true_y_this_node)
                label_this_node = sorted(occurrence_count.items(), key=lambda x: x[1], reverse=True)[0][0]
                predict_leaf_value[t] = label_this_node
        self.c = predict_leaf_value

    def copy(self):
        return Tree(self.root_node, self.depth, self.a.copy(), self.b.copy(), self.alpha)


class WholeTree(Tree):
    def __init__(self, depth: int, a: dict, b: dict, alpha: float = 0.1):
        super(WholeTree, self).__init__(1, depth, a, b, alpha)


class TreeModel(Tree):
    def __init__(self, depth: int, p: int, alpha: float = 0.1):
        parent_nodes = [t for t in range(2 ** depth)]
        a = {}
        b = {}
        for t in parent_nodes:
            at = np.zeros(p)
            j = random.randint(0, p - 1)
            at[j] = 1
            bt = random.random()
            a[t] = at
            b[t] = bt
        super(TreeModel, self).__init__(1, depth, a, b, alpha)


if __name__ == "__main__":
    fake_a = {t: np.array([0, 0]) for t in range(100)}
    fake_b = {t: 0 for t in range(100)}
    tree = Tree(3, 2, fake_a, fake_b)
    print(tree.get_nodes())
    l, r = tree.children()
    print(l.a, r.get_nodes())
    # print(tree.min_leaf_size(np.array([[1, 1], [2, 2]])))
