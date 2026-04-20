from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import rand_score, adjusted_rand_score
import sklearn.datasets
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
import matplotlib.pyplot as plt


class StableClustering(BaseEstimator, ClusterMixin):
    def __init__(self, k_values, n_repetitions=10, algorithm="k-means", metric="adjusted_rand", random_state=42):
        self.k_values = k_values
        self.n_repetitions = n_repetitions
        self.algorithm = algorithm
        self.metric = metric
        self.random_state = random_state
        self.scores_ = {}

    def fit(self, X):
        best_k = None
        best_score = -1
        self.models_ = []

        for k in tqdm(self.k_values, desc="k"):
            clusters = []
            for i_rep in tqdm(range(self.n_repetitions), desc='Repetitions', leave=False):
                if self.algorithm == "k-means":
                    model = KMeans(
                        n_clusters=k, random_state=self.random_state + i_rep)  # , init='random')
                    labels = model.fit_predict(X)
                # elif self.algorithm == "nmf":
                #     model = NMF(n_components=k, init='random',
                #                 random_state=self.random_state + i_rep)
                #     labels = np.argmax(
                #         model.fit_transform(X - X.min()), axis=1)
                else:
                    raise ValueError(
                        "Invalid algorithm: choose 'k-means'")
                clusters.append(labels)

            scores = []
            for i in range(self.n_repetitions):
                for j in range(i + 1, self.n_repetitions):
                    if self.metric == "rand":
                        score = rand_score(clusters[i], clusters[j])
                    elif self.metric == "adjusted_rand":
                        score = adjusted_rand_score(clusters[i], clusters[j])
                    else:
                        raise ValueError(
                            "Invalid metric: choose 'rand' or 'adjusted_rand'")
                    scores.append(score)

            self.models_.append(deepcopy(model))

            avg_score = np.mean(scores)
            # Store the average score for this k
            self.scores_[k] = float(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_k = k
                best_model = model

        # Fit the final model on the whole data
        self.best_k_ = best_k
        self.best_model_ = best_model
        return self

    def predict(self, X):
        check_is_fitted(self, ["best_model_", "best_k_"])
        if self.algorithm == "k-means":
            return self.best_model_.predict(X)
        # elif self.algorithm == "nmf":
        #     return np.argmax(self.best_model.transform(X), axis=1)

    def predict_closest_points_to_centroids(self, X, n_closest=1):
        '''
        Returns predicted cluster index for each point in X.
        For the n_closest points of each cluster to each centroid, returns the cluster index.
        Returns -1 for all other points.
        '''
        check_is_fitted(self, ["best_model_", "best_k_"])
        distances = self.best_model_.transform(
            X)  # Shape: (n_samples, n_clusters)
        cluster_assignments = self.predict(X)

        preds = np.full(X.shape[0], -1)
        for i in range(self.best_k_):
            distances_ = deepcopy(distances[:, i])
            distances_[cluster_assignments != i] = np.inf
            closest_points = np.argsort(distances_)[:n_closest]
            preds[closest_points] = i
        return preds


if __name__ == '__main__':
    # sample sklearn datraset
    X_simple = sklearn.datasets.load_iris().data

    stable_clustering = StableClustering(
        k_values=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15], n_repetitions=10,
        algorithm="k-means",
        # algorithm="nmf",
        metric="adjusted_rand")
    stable_clustering.fit(X_simple)
    print(stable_clustering.scores_)  # Dictionary of scores for each k

    plt.plot(list(stable_clustering.scores_.keys()), list(
        stable_clustering.scores_.values()), '.-')
