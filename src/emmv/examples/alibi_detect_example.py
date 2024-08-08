"""Example of using the emmv_scores function with a model from the Alibi-Detect library."""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from emmv import emmv_scores


def run():
    """Run the example."""
    # Data generation adapted from:
    # https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_vae_adult.html#Dataset
    import alibi
    from alibi_detect.od import IForest, Mahalanobis

    rng = np.random.RandomState(42)

    adult = alibi.datasets.fetch_adult()
    X, y = adult.data, adult.target
    feature_names = adult.feature_names
    category_map_tmp = adult.category_map

    # Reduce dataset from example to just 100 instances
    X = X[:100, :]
    y = y[:100]

    np.random.seed(1)
    Xy_perm = np.random.permutation(np.c_[X, y])
    X, y = Xy_perm[:, :-1], Xy_perm[:, -1]

    keep_cols = [2, 3, 5, 0, 8, 9, 10]
    feature_names = (
        feature_names[2:4] + feature_names[5:6] + feature_names[0:1] + feature_names[8:11]
    )
    X = X[:, keep_cols]

    category_map = {}
    i = 0
    for k, v in category_map_tmp.items():
        if k in keep_cols:
            category_map[i] = v
            i += 1
    cat_cols = list(category_map.keys())

    X_num = X[:, -4:].astype(np.float32, copy=False)
    xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
    rng = (-1.0, 1.0)
    X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

    X_cat = X[:, :-4].copy()
    ohe = OneHotEncoder(categories='auto')
    ohe.fit(X_cat)
    X = np.c_[X_cat, X_num_scaled].astype(np.float32, copy=False)

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
