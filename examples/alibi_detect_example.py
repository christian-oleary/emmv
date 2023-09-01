'''
Author: Christian O'Leary
Email: christian.oleary@mtu.ie
'''

import alibi
from alibi_detect.od import IForest, Mahalanobis, OutlierVAE
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer

from emmv import emmv_scores

# Data generation adapted from:
# https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_vae_adult.html#Dataset
rng = np.random.RandomState(42)
NUM_COLS = 2

adult = alibi.datasets.fetch_adult()
X, y = adult.data, adult.target
feature_names = adult.feature_names
category_map_tmp = adult.category_map

# Reduce dataset from example to just 100 instances
X = X[:100,:]
y = y[:100]

np.random.seed(1)
tf.random.set_seed(1)
Xy_perm = np.random.permutation(np.c_[X, y])
X, y = Xy_perm[:,:-1], Xy_perm[:,-1]

keep_cols = [2, 3, 5, 0, 8, 9, 10]
feature_names = feature_names[2:4] + feature_names[5:6] + feature_names[0:1] + feature_names[8:11]
X = X[:, keep_cols]

category_map = {}
i = 0
for k, v in category_map_tmp.items():
    if k in keep_cols:
        category_map[i] = v
        i += 1
cat_cols = list(category_map.keys())
num_cols = [col for col in range(X.shape[1]) if col not in cat_cols]

X_num = X[:, -4:].astype(np.float32, copy=False)
xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
rng = (-1., 1.)
X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

X_cat = X[:, :-4].copy()
ohe = OneHotEncoder(categories='auto')
ohe.fit(X_cat)
X = np.c_[X_cat, X_num_scaled].astype(np.float32, copy=False)

n_train = 80
n_valid = 10
X_train, y_train = X[:n_train,:], y[:n_train]
X_valid, y_valid = X[n_train:n_train+n_valid,:], y[n_train:n_train+n_valid]
X_test, y_test = X[n_train+n_valid:,:], y[n_train+n_valid:]


# Alibi-Detect models do not have a "decision_function" method, so we need to make one.
def scoring_function(model, X_test):
    return model.predict(X_test)['data']['instance_score']


# Isolation Forest
model = IForest(threshold=0.1)
model.fit(X_train)
test_scores = emmv_scores(model, X_test, scoring_function)
print('IForest')
print('Excess Mass score;', test_scores['em'])
print('Mass Volume score:', test_scores['mv'])


# Second example with Mahalanobis Distance
cat_vars_ord = {}
n_categories = len(cat_cols)
for i in range(n_categories):
    cat_vars_ord[i] = len(np.unique(adult.data[:, i]))
model = Mahalanobis(threshold=0.1, cat_vars=cat_vars_ord)
model.fit(X_train)
test_scores = emmv_scores(model, X_test, scoring_function)
print('\nMahalanobis')
print('Excess Mass score;', test_scores['em'])
print('Mass Volume score:', test_scores['mv'])


# Third example with Variational Autoencoder adapted from:
# https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_vae_adult.html
n_features = X_train.shape[1]
latent_dim = 2

encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(n_features,)),
        Dense(25, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.relu),
        Dense(5, activation=tf.nn.relu)
    ])

decoder_net = tf.keras.Sequential([
        InputLayer(input_shape=(latent_dim,)),
        Dense(5, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.relu),
        Dense(25, activation=tf.nn.relu),
        Dense(n_features, activation=None)
    ])

# initialize outlier detector
model = OutlierVAE(threshold=0.1, # threshold for outlier score
                score_type='mse', # use MSE of reconstruction error for outlier detection
                encoder_net=encoder_net, # can also pass VAE model instead
                decoder_net=decoder_net, # of separate encoder and decoder
                latent_dim=latent_dim,
                samples=5)

model.fit(X_train, loss_fn=tf.keras.losses.mse, epochs=1, verbose=True)

test_scores = emmv_scores(model, X_test, scoring_function)
print('\nOutlierVAE')
print('Excess Mass score;', test_scores['em'])
print('Mass Volume score:', test_scores['mv'])
