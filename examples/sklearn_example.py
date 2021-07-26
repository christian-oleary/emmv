from emmv import emmv_scores

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Data and model fitting adapted from: https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
# fit the model
model = IsolationForest(max_samples=100, random_state=rng)
model.fit(X_train)

# Get EM & MV scores
test_scores = emmv_scores(model, X)
print('Excess Mass score;', test_scores['em'])
print('Mass Volume score:', test_scores['mv'])
