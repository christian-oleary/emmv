"""Example of using the emmv_scores function with a model from the PyOD library."""

import numpy as np
from pyod.models.copod import COPOD

from emmv import emmv_scores


def run():
    """Run the example."""
    rng = np.random.RandomState(42)

    NUM_COLS = 2
    # Generate train data
    X = 0.3 * rng.randn(100, NUM_COLS)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * rng.randn(20, NUM_COLS)
    X_regular = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, NUM_COLS))
    # fit the model
    model = COPOD()
    model.fit(X_train)

    # Get EM & MV scores
    X_test = np.concatenate((X_regular, X_outliers), axis=0)
    test_scores = emmv_scores(model, X_test)
    print('Excess Mass score;', test_scores['em'])
    print('Mass Volume score:', test_scores['mv'])


if __name__ == "__main__":
    run()
