import numpy as np


def make_rj(n=300, p=50):
    """Generates data according to the model in Radchenko & James, 2010
    X_i ~ Unif([0,1]^p)
    y = sqrt(0.5)[sum_{i=1}^5 f_i(x) + f_1(x)f_2(x) + f_1(x)f_3(x)] + N(0,1)
    f_1(x) = x1, f_2(x) = (1+x2)^{-1}, f_3(x) = sin(x3), f_4(x) = e^x4, f_5(x) = x5^2
    function withing the sum are normalized

    Params
    ------
        n (int): number of sample
        p (int): number of features

    Returns
    -------
        Tuple[np.array, np.array]: design matrix and label vector

    """

    X = np.random.uniform(0, 1, size=(n, p))
    f_1 = X[:, 0]
    f_2 = (1 + X[:, 1]) ** (-1)
    f_3 = np.sin(X[:, 2])
    f_4 = np.exp(X[:, 3])
    f_5 = X[:, 4] ** (2)

    def _normalize_vec(v):
        return (v - np.mean(v)) / np.std(v)

    effects = _normalize_vec(f_1) + _normalize_vec(f_2) + _normalize_vec(f_3) + _normalize_vec(f_4) + _normalize_vec(
        f_5)
    interactions = f_1 * f_2 + f_1 * f_3

    y = effects + interactions + np.random.normal(size=n)

    return X, y


def make_vp(n=100, p=100):
    """Generates data according to https://arxiv.org/abs/1607.02670 (Sparse additive Gaussian process with soft interactions)
    X_i ~ N(0, I)
    y = x1 + x2^2 + x3 + x4^2 + x5 + x1x2 + x2x3 + x3x4 + N(0, 0.14)

    Args:
        n (int): number of sample
        p (int): number of features

    Returns:
        Tuple[np.array, np.array]: design matrix and label vector

    """
    X = np.random.normal(size=(n, p))
    effects = X[:, 0] + X[:, 1] ** 2 + X[:, 2] + X[:, 3] ** 2 + X[:, 4]
    interactions = X[:, 0] * X[:, 1] + X[:, 1] * X[:, 2] + X[:, 2] * X[:, 3]

    y = effects + interactions + np.random.normal(scale=0.14, size=n)

    return X, y
