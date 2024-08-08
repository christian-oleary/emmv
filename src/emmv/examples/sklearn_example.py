"""Example of using the emmv_scores function with a model from the scikit-learn library."""

import numpy as np
from sklearn.ensemble import IsolationForest

from emmv import emmv_scores


def run():
    """Run the example."""
    rng = np.random.RandomState(42)
    num_cols = 2

    # Data and model fitting adapted from:
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html

    # Generate train data
    X = 0.3 * rng.randn(100, num_cols)
    X_train = np.r_[X + 2, X - 2]

    # Generate test data
    X_test = 0.3 * rng.randn(20, num_cols)
    X_test = np.r_[X_test + 2, X_test - 2]

    # Add outliers
    outliers = rng.uniform(low=-4, high=4, size=(20, num_cols))
    X_test = np.concatenate((X_test, outliers), axis=0)

    # fit the model
    model = IsolationForest(max_samples=100, random_state=rng)
    model.fit(X_train)

    # Get EM & MV scores
    scores = emmv_scores(model, X_test)
    print(f'Excess Mass: {scores[0]}\nMass Volume: {scores[1]}')


if __name__ == "__main__":
    run()
