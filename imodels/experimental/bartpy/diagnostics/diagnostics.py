import numpy as np
import pandas as pd
from imodels import FIGSRegressor
from matplotlib import pyplot as plt
from sklearn import datasets, model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from imodels.util.tree_interaction_utils import get_interacting_features
from ..diagnostics.residuals import plot_qq, plot_homoskedasity_diagnostics
from ..diagnostics.sampling import plot_tree_mutation_acceptance_rate, plot_tree_likelihhod, plot_tree_probs
from ..diagnostics.sigma import plot_sigma_convergence
from ..diagnostics.trees import plot_tree_depth
from ..initializers.sklearntreeinitializer import SklearnTreeInitializer
from ..sklearnmodel import SklearnModel, BART



def plot_diagnostics(model: SklearnModel):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, _)) = plt.subplots(2, 4, figsize=(10, 10))
    fig.suptitle("Diagnostics")
    plot_qq(model, ax1)
    plot_tree_depth(model, ax2)
    plot_sigma_convergence(model, ax3)
    plot_homoskedasity_diagnostics(model, ax4)
    plot_tree_mutation_acceptance_rate(model, ax5)
    # plot_tree_likelihhod(model, ax6)
    # plot_tree_probs(model, ax7)

    plt.show()


if __name__ == '__main__':

    n = 100
    n_trees = 1
    # X, y = datasets.make_friedman1(n)
    x_1 = np.random.choice([2, 3, 5, 7, 8], size=n)
    x_2 = np.random.choice([1, 2, 3, 4], size=n)
    X = pd.DataFrame([x_1, x_2]).T


    def _f(x):
        x_1 = x[0]
        x_2 = x[1]

        if np.logical_and(x_1 <= 5 , x_2 in [1,3]):
            return 8
        elif np.logical_and(x_1 > 5 , x_2 in [1,3]):
            return 2
        elif np.logical_and(x_1 <= 3 , x_2 in [2,4]):
            return 1
        elif np.logical_and(3 < x_1 <= 7 , x_2 in [2,4]):
            return 5
        elif np.logical_and(x_1 > 7 , x_2 in [2,4]):
            return 8


    y = np.array([_f(X.iloc[i, :]) for i in range(n)]) + np.random.normal(size=n, scale=0.1)


    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=4)

    bart_zero = BART(classification=False, store_acceptance_trace=True, n_trees=n_trees, n_samples=10000, n_burn=100,
                     n_chains=5,
                     thin=1)
    bart_zero.fit(X_train, y_train)

    # bart_figs = BART(classification=False, store_acceptance_trace=True, n_trees=n_trees, n_samples=1000, n_burn=100,
    #                  n_chains=5,
    #                  thin=1, initializer=SklearnTreeInitializer(tree_=figs))
    #
    # bart_figs.fit(X_train, y_train)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig, ax = plt.subplots(1)
    bart_preds = bart_zero.predict(X_test)
    # bart_figs_preds = bart_figs.predict(X_test)
    # figs_preds = figs.predict(X_test)
    # # print(f"figs: {mean_squared_error(figs_preds, y_test)}")
    # bfigs_preds = bart_figs.predict(X_test)
    # print(f"bfigs: {mean_squared_error(bfigs_preds, y_test)}")

    # plot_tree_depth(bart_sgb, ax1, f"CART initialization {np.round(mean_squared_error(bart_sgb_preds, y_test), 4)}")
    plot_tree_depth(bart_zero, ax, f"BART initialization (MSE: {np.round(mean_squared_error(bart_preds, y_test), 4)})")
    # plot_tree_depth(bart_figs, ax2,
    #                 f"FIGS initialization (MSE: {np.round(mean_squared_error(bart_figs_preds, y_test), 4)}"
    #                 f", FIGS MSE: {np.round(mean_squared_error(figs_preds, y_test), 2)})", x_label=True)
    # plt.title(f"Bayesian tree with different initilization of Friedman 1 dataset n={n}")

    plt.show()

    # y_hat = bart.predict(X)
    # plot_diagnostics(bart)
