"""Example of using the emmv_scores function with a model from the Alibi-Detect library."""

import alibi
from alibi_detect.od import IForest, Mahalanobis
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from emmv import emmv_scores


def run():
    """Run the example."""
    # Data generation adapted from:
    # https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_vae_adult.html#Dataset

    # pylint: disable=no-member

    rng = np.random.RandomState(42)
    adult: alibi.utils.data.Bunch = alibi.datasets.fetch_adult()
    X, y = adult.data, adult.target
    feature_names = adult.feature_names
    category_map_tmp = adult.category_map

    # Reduce dataset from example to just 100 instances
    X = X[:100, :]
    y = y[:100]

    np.random.seed(1)
    permutations = np.random.permutation(np.c_[X, y])
    X, y = permutations[:, :-1], permutations[:, -1]

    keep_cols = [2, 3, 5, 0, 8, 9, 10]
    feature_names = (
        feature_names[2:4] + feature_names[5:6] + feature_names[0:1] + feature_names[8:11]
    )
    X = X[:, keep_cols]

    category_map = {}
    i = 0
    for key, value in category_map_tmp.items():
        if key in keep_cols:
            category_map[i] = value
            i += 1
    cat_cols = list(category_map.keys())

    x_num = X[:, -4:].astype(np.float32, copy=False)
    xmin, xmax = x_num.min(axis=0), x_num.max(axis=0)
    rng = (-1.0, 1.0)
    x_num_scaled = (x_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

    x_cat = X[:, :-4].copy()
    ohe = OneHotEncoder(categories='auto')
    ohe.fit(x_cat)
    X = np.c_[x_cat, x_num_scaled].astype(np.float32, copy=False)

    n_train = 80
    X_train, _ = X[:n_train, :], y[:n_train]
    X_test, __ = X[n_train:, :], y[n_train:]

    # Alibi-Detect models do not have a "decision_function" method, so we need to make one.
    def scoring_function(model, X_test):
        return model.predict(X_test)['data']['instance_score']

    # Isolation Forest
    model = IForest(threshold=0.1)
    model.fit(X_train)
    scores = emmv_scores(model, X_test, scoring_function)
    print(f'\nIForest\nExcess Mass: {scores[0]}\nMass Volume: {scores[1]}')

    # Second example with Mahalanobis Distance
    cat_vars_ord = {}
    n_categories = len(cat_cols)
    for i in range(n_categories):
        cat_vars_ord[i] = len(np.unique(adult.data[:, i]))
    model = Mahalanobis(threshold=0.1, cat_vars=cat_vars_ord)
    model.fit(X_train)
    scores = emmv_scores(model, X_test, scoring_function)
    print(f'\nMahalanobis\nExcess Mass: {scores[0]}\nMass Volume: {scores[1]}')


if __name__ == "__main__":
    run()
