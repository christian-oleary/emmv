"""Example of using the emmv_scores function with a model from the ADTK library."""

from adtk.data import validate_series
from adtk.detector import GeneralizedESDTestAD
import pandas as pd
import numpy as np

from emmv import emmv_scores

# pylint: disable=protected-access


def run():
    """Run the example."""

    rng = np.random.RandomState(42)

    # Generate train data
    timestamps = pd.date_range("2018-01-01", periods=200, freq="H")
    X = 0.3 * rng.randn(100)
    values = np.r_[X + 2, X - 2]
    data = pd.Series(values, index=timestamps)
    X_train = validate_series(data)

    # Generate some regular novel observations
    X = 0.3 * rng.randn(67)
    x_regular = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    x_outliers = rng.uniform(low=-4, high=4, size=66)
    # Create test data
    timestamps = pd.date_range("2018-01-01", periods=200, freq="H")
    values = np.concatenate((x_regular, x_outliers), axis=0)
    data = pd.Series(values, index=timestamps)
    X_test = validate_series(data)

    # Fit model
    model = GeneralizedESDTestAD()
    model.fit_detect(X_train)

    # Get EM & MV scores

    # ADTK models do not have a "decision_function" method, so we need to specify
    # our own custom anomaly scoring function. This one is specific to GeneralizedESDTestAD.
    # It is adapted from: https://github.com/odnura/adtk/blob/73bfb30ba457dd540e8aea82782431254da480ce/src/adtk/detector/_detector_1d.py#L346
    def scoring_function(model, df):
        data = pd.Series(df)  # 1D data expected
        new_sum = data + model._normal_sum  # pylint: disable=protected-access
        new_count = model._normal_count + 1
        new_mean = new_sum / new_count
        new_squared_sum = data**2 + model._normal_squared_sum  # pylint: disable=protected-access
        new_std = np.sqrt(
            (new_squared_sum - 2 * new_mean * new_sum + new_count * new_mean**2) / (new_count - 1)
        )
        anomaly_scores = (data - new_mean).abs() / new_std
        return anomaly_scores

    scores = emmv_scores(model, X_test, scoring_function)
    print(f'\nIForest\nExcess Mass: {scores[0]}\nMass Volume: {scores[1]}')


if __name__ == "__main__":
    run()
