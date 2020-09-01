import logging

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as load_data
from sklearn.tree import DecisionTreeClassifier

from .optree import OptimalHyperTreeModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', )

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def run_example(depth: int = 2):
    features, label = load_data(return_X_y=True)
    p = features.shape[1]
    column_names = ["x{0}".format(i) for i in range(p)]
    data = pd.DataFrame(data=features, columns=column_names)
    data["label"] = label

    test_indices = np.random.random_integers(0, data.shape[0]-1, size=(int(data.shape[0] * 0.2), ))
    train_indices = [i for i in range(0, data.shape[0]) if i not in test_indices]
    train = data.iloc[train_indices, ].reset_index()
    test = data.iloc[test_indices, ].reset_index()

    print(train.shape)

    # Use sklearn
    train_features_sklearn = features[train_indices, ::]
    train_label_sklearn = label[train_indices]
    test_features_sklearn = features[test_indices, ::]
    test_label_sklearn = label[test_indices]
    cart_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1)
    clf = cart_model.fit(train_features_sklearn, train_label_sklearn)
    predicted_y = clf.predict(test_features_sklearn)

    # Use PyOptree
    model = OptimalHyperTreeModel(column_names, "label", tree_depth=depth, N_min=1)
    model.train(train, train_method="mio")

    test = model.predict(test)

    print(model.a)

    print("PyOptree Library Tree Prediction Accuracy: {}".format(sum(test["prediction"]==test["label"]) / len(test["label"])))

    print("SKLearn Library Tree Prediction Accuracy: {}".format(sum(predicted_y==test_label_sklearn) / len(test_label_sklearn)))


if __name__ == "__main__":
    run_example(depth=3)

