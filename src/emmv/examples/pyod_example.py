"""Example of using the emmv_scores function with a model from the PyOD library."""

import numpy as np
from pyod.models.copod import COPOD

from emmv import emmv_scores


def run():
    """Run the example."""
    rng = np.random.RandomState(42)

    num_cols = 2
    # Generate train data
    X = 0.3 * rng.randn(100, num_cols)
    X_train = np.r_[X + 2, X - 2]

    # Generate some regular novel observations
    X = 0.3 * rng.randn(20, num_cols)
    regular = np.r_[X + 2, X - 2]

    # Generate some abnormal novel observations
    outliers = rng.uniform(low=-4, high=4, size=(20, num_cols))

    # fit the model
    model = COPOD()
    model.fit(X_train)

    # Get EM & MV scores
    X_test = np.concatenate((regular, outliers), axis=0)
    scores = emmv_scores(model, X_test)
    print(f'Excess Mass: {scores[0]}\nMass Volume: {scores[1]}')


if __name__ == "__main__":
    run()
