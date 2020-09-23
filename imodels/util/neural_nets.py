"""Bridging random forests and deep neural networks. Code to convert a sklearn decision tree to a pytorch neural network
following "Neural Random Forests" https://arxiv.org/abs/1604.07143


Example
-------

    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    np.random.seed(13)


    num_features = 4
    N = 1000
    max_depth = 100

    # prepare data
    X = np.random.rand(N, num_features)
    y = np.random.rand(N)
    X_t = torch.Tensor(X)

    # train rf
    dt = DecisionTreeRegressor(max_depth=max_depth)
    dt.fit(X, y)


    # pepare net
    net = Net(dt)


    # check if preds are close
    preds_dt = dt.predict(X).flatten()
    preds_net = net(X_t).detach().numpy().flatten()

    assert np.isclose(preds_dt, preds_net).all(), 'preds are not close'

"""
import time
from copy import deepcopy

import numpy as np
from torch import nn


class Net(nn.Module):
    '''
    class which converts estimator (decision tree type) to a dnn
    '''

    def __init__(self, estimator):
        super(Net, self).__init__()

        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left  # left_child, id of the left child of the node
        children_right = estimator.tree_.children_right  # right_child, id of the right child of the node
        feature = estimator.tree_.feature  # feature, feature used for splitting the node
        threshold = estimator.tree_.threshold  # threshold, threshold value at the node
        num_leaves = estimator.tree_.n_leaves
        num_non_leaves = estimator.tree_.node_count - num_leaves
        node_depth, is_leaves = self.calc_depths_and_leaves(n_nodes, children_left, children_right)
        self.values = estimator.tree_.value
        self.all_leaf_paths = {}
        self.calc_all_leaf_paths(0, n_nodes, children_left, children_right, running_list=[])  # set all_leaf_paths

        # initialize layers to zero
        self.layers = nn.Sequential(
            nn.Linear(estimator.n_features_, num_non_leaves),
            nn.Linear(num_non_leaves, num_leaves),
            nn.Linear(num_leaves, 1, bias=False)
        )
        for i in range(2):
            self.layers[i].weight.data *= 0
            self.layers[i].bias.data *= 0

        # set the first layer
        nonleaf_node_to_nonleaf_neuron_num = {}  # np.zeros(num_non_leaves)
        nonleaf_neuron_num = 0
        for i in range(n_nodes):
            if not is_leaves[i]:
                self.layers[0].weight.data[nonleaf_neuron_num, feature[i]] = 1
                self.layers[0].bias.data[nonleaf_neuron_num] = -threshold[i]
                nonleaf_node_to_nonleaf_neuron_num[i] = nonleaf_neuron_num
                nonleaf_neuron_num += 1

                # set the 2nd + 3rd layer
        for leaf_neuron_num, leaf_idx in enumerate(sorted(self.all_leaf_paths.keys())):
            path = self.all_leaf_paths[leaf_idx]

            # 2nd lay
            for (nonleaf_node, sign) in path:
                self.layers[1].weight.data[leaf_neuron_num,
                                           nonleaf_node_to_nonleaf_neuron_num[
                                               nonleaf_node]] = sign  # num_leaves x num_non_leaves
                self.layers[1].bias.data[leaf_neuron_num] = -1 * float(node_depth[leaf_idx])

            # 3rd lay
            self.layers[2].weight.data[0, leaf_neuron_num] = self.values[leaf_idx][
                0, 0]  # note, this will be multivariate for classification!

    # placeholder so class compiles
    def forward(self, x):
        #     t0 = time.perf_counter()
        x = x.reshape(x.shape[0], -1)
        x = self.layers[0](x)
        t1 = time.perf_counter()
        x[x < 0] = -1
        x[x >= 0] = 1

        #     t2 = time.perf_counter()
        x = self.layers[1](x)
        x = (x == 0).float()

        x = self.layers[2](x)
        #     t3 = time.perf_counter()
        #     print(f't1: {t1-t0:0.2e}, t2: {t2-t1:0.2e} t3: {t3-t2:0.2e}')
        return x

    def calc_depths_and_leaves(self, n_nodes, children_left, children_right):
        '''
        calculate numpy arrays representing the depth of each node and whether they are leaves or not
        '''
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)

        # calculate node_depth and is_leaves
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        return node_depth, is_leaves

    def calc_all_leaf_paths(self, node_idx, n_nodes, children_left, children_right, running_list):
        '''
        recursively store all leaf paths into a dictionary as tuples of (node_idxs, weight)
        weight is -1/+1 depending on if it left/right
        running_list is a reference to one list which is shared by all calls!
        '''
        # check if we are at a leaf
        if children_left[node_idx] == children_right[node_idx]:
            self.all_leaf_paths[node_idx] = deepcopy(running_list)
        else:
            running_list.append((node_idx, -1))  # assign weight of -1 to left
            self.calc_all_leaf_paths(children_left[node_idx], n_nodes, children_left, children_right, running_list)
            running_list.pop()
            running_list.append((node_idx, +1))  # assign weight of +1 to right
            self.calc_all_leaf_paths(children_right[node_idx], n_nodes, children_left, children_right, running_list)
            running_list.pop()

    def extract_util_np(self):
        b0 = self.layers[0].bias.data.numpy()
        idxs0 = self.layers[0].weight.data.argmax(dim=1).numpy()
        w1 = self.layers[1].weight.data.numpy().T
        b1 = self.layers[1].bias.data.numpy()
        num_leaves = self.layers[2].weight.shape[1]
        idxs2 = np.zeros(num_leaves)  # leaf_neuron_num_to_val
        # iterate over leaves and map to values
        for leaf_neuron_num, i in enumerate(sorted(self.all_leaf_paths.keys())):
            idxs2[leaf_neuron_num] = self.values[i, 0, 0]
        return b0, idxs0, w1, b1, idxs2
