from matplotlib import pyplot as plt

from bartpy.sklearnmodel import SklearnModel


def plot_residuals(model: SklearnModel):
    plt.plot(model.data.unnormalized_y - model.predict())
    plt.show()


def plot_modelled_against_actual(model: SklearnModel):
    plt.plot(model.data.unnormalized_y)
    plt.plot(model.predict())
    plt.show()
