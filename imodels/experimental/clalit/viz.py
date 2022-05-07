import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from .clalit import DATASETS_CLASSIFICATION, PTH


def _get_plot_data(ds_name):
    df = pd.read_csv(os.path.join(PTH, f'{ds_name}.csv'), index_col=0)
    plt_data = df.loc[["n", "roc_auc_mean", "roc_auc_std", "n_rules_mean"], :]
    figs_data = list(plt_data.loc[:, "FIGS"])
    figs_data.append("FIGS")
    xgb_data = list(plt_data.loc[:, "XGB"])
    xgb_data.append("XGB")
    plt_data_final = pd.DataFrame([figs_data, xgb_data])
    return plt_data_final


def main():
    plt_data = pd.concat([_get_plot_data(d[0]) for d in DATASETS_CLASSIFICATION])

    figs_data = plt_data.values[plt_data.iloc[:, -1] == "FIGS", :]
    xgb_data = plt_data.values[plt_data.iloc[:, -1] == "XGB", :]

    fig, ax = plt.subplots(2, figsize=(10, 10))
    fig.subplots_adjust(hspace=.5)

    ticks = [f"{d[0].replace('-', ' ').capitalize()} (n={int(figs_data[i, 0])})" for i, d in
             enumerate(DATASETS_CLASSIFICATION)]

    n_samples = np.arange(
        len(DATASETS_CLASSIFICATION))  # [np.log(figs_data[i, 0]) for i in range(len(DATASETS_CLASSIFICATION))]

    ax[0].scatter(n_samples, figs_data[:, 3], c="blue", label="FIGS")

    ax[0].scatter(n_samples, xgb_data[:, 3], c="red", label="XGB")

    ax[0].set_xticks(n_samples, ticks,rotation=20)

    # ax[0].set_xlabel("Number of Samples (log scale)")
    ax[0].set_ylabel("# of Rules")
    ax[0].legend()

    ax[1].errorbar(n_samples, figs_data[:, 1], yerr=figs_data[:, 2], c="blue", fmt='o',
                   label="FIGS")
    ax[1].errorbar(n_samples, xgb_data[:, 1], yerr=xgb_data[:, 2], c="red", fmt='o', label="XGB")
    ax[1].set_xticks(n_samples, ticks, rotation=20)

    # ax[1].set_xlabel("Number of Samples (log scale)")
    ax[1].set_ylabel("ROC AUC")

    # plt.legend()
    plt.savefig(os.path.join(PTH, "xgb_figs.png"))

    # pass


if __name__ == '__main__':
    main()
