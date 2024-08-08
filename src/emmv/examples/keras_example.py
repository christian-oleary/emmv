"""Example of using the emmv_scores function with a model from the Keras library."""

import os
import random

import numpy as np

from emmv import emmv_scores


def run():
    """Run the example."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, InputLayer
    from tensorflow.keras.models import Sequential

    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    rng = np.random.RandomState(seed)
    tf.random.set_seed(seed)
    NUM_COLS = 2

    # Generate train data
    X = 0.3 * rng.randn(100, NUM_COLS)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * rng.randn(20, NUM_COLS)
    X_regular = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, NUM_COLS))
    X_test = np.concatenate((X_regular, X_outliers), axis=0)

    # fit the model
    model = Sequential(
        [InputLayer(input_shape=NUM_COLS), Dense(32), Dense(NUM_COLS, activation='relu')]
    )
    model.compile(loss='mse', optimizer='adam')
    model.fit(
        X_train,
        X_train,  # i.e. reconstruction model
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        verbose=1,
    )

    # Get EM & MV scores

    # TF models do not have a "decision_function" method, so we need to specify
    # our own custom anomaly scoring function. This one uses MAPE.
    def scoring_function(model, df):
        offset = 0.00000001  # to prevent division by 0
        # 1. model predictions
        preds = model.predict(df)
        # 2. Use a regression metric, e.g. MAPE
        anomaly_scores = np.mean((np.abs(preds - df) / (df + offset)), axis=1)  # i.e. anomaly score
        return anomaly_scores

    scores = emmv_scores(model, X_test, scoring_function)
    print(f'Excess Mass: {scores[0]}\nMass Volume: {scores[1]}')


if __name__ == "__main__":
    run()
