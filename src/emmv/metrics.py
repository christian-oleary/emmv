"""Excess-Mass and Mass-Volume scores for unsupervised ML AD models."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import auc

logger = logging.getLogger(__name__)


def default_scoring_func(model, df):
    """Default scoring function for anomaly detection models."""
    return model.decision_function(df)


def emmv_scores(
    trained_model,
    df: pd.DataFrame,
    scoring_func=None,
    n_generated: int = 100000,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    em_min: float = 0.9,
) -> tuple:
    """Get Excess-Mass (EM) and Mass Volume (MV) scores for unsupervised ML AD models.

    :param trained_model: Trained ML model with a 'decision_function' method
    :param df: Pandas dataframe of features (X matrix)
    :param scoring_func: Anomaly scoring function (callable)
    :param n_generated: Number of generated samples, defaults to 100000
    :param alpha_min: Min value for alpha axis, defaults to 0.9
    :param alpha_max: Max value for alpha axis, defaults to 0.999
    :param em_min: Min EM value required, defaults to 0.9
    :return: Tuple of EM and MV scores
    """

    if scoring_func is None:
        scoring_func = default_scoring_func

    # Get limits and volume support
    lim_inf = df.min(axis=0)
    lim_sup = df.max(axis=0)
    offset = 1e-60  # to prevent division by 0

    # Volume of rectangle containing all data in X
    volume_support = float((lim_sup - lim_inf).prod()) + offset

    # An "array of levels, on which we want to evaluate
    # EM_s(t) on samples X from an underlying density f."
    levels = np.arange(0, 100 / volume_support, 0.01 / volume_support)

    # uniform_sample represents "s, evaluated on a uniform sample generated on
    # a rectangle containing all the data X. This is used to estimate Leb(s>t)"
    try:
        uniform_sample = np.random.uniform(lim_inf, lim_sup, size=(n_generated, df.shape[1]))
    except IndexError:  # i.e. 1D data
        uniform_sample = np.random.uniform(lim_inf, lim_sup, size=n_generated)

    # Get anomaly scores
    anomaly_scores = scoring_func(trained_model, df)  # .reshape(1, -1)[0]
    uniform_scores = scoring_func(trained_model, uniform_sample)

    # Get EM and MV scores
    _, em_score = excess_mass(levels, em_min, volume_support, uniform_scores, anomaly_scores)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    _, mv_score = mass_volume(axis_alpha, volume_support, uniform_scores, anomaly_scores)
    return em_score, mv_score


def excess_mass(
    levels: np.ndarray,
    em_min: float,
    volume_support: float,
    uniform_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> tuple:
    """Calculate Excess-Mass scores.

    Variables explained here: https://github.com/ngoix/EMMV_benchmarks/issues/2
    - "leb" refers to Lebesgue measure
    - "s" refers a measurable function
    - "X" refers to dataset

    :param np.ndarray levels: Levels on which to evaluate EM_s(t) on samples X from underlying density f.
    :param float em_min: Beginning of EM curve.
    :param float volume_support: Volume of rectangle containing all data in X.
    :param np.ndarray uniform_scores: s(U), U = uniform samples from rectangle of all data in X, used to estimate Leb(s>t)
    :param np.ndarray anomaly_scores: s(X), i.e. s evaluated on a sample from underlying density f, used to estimate P(s>t)
    :return tuple: AUC and EM scores
    """
    n_samples = anomaly_scores.shape[0]
    unique_anomaly_scores = np.unique(anomaly_scores)
    excess_mass_scores = np.zeros(levels.shape[0])
    excess_mass_scores[0] = 1.0

    for u in unique_anomaly_scores:
        excess_mass_scores = np.maximum(
            excess_mass_scores,
            1.0 / n_samples * (anomaly_scores > u).sum()
            - levels * (uniform_scores > u).sum() / len(uniform_scores) * volume_support,
        )
    index = int(np.argmax(excess_mass_scores <= em_min).flatten()[0]) + 1
    if index == 1:
        logger.warning('Failed to achieve em_min')
        index = -1
    em_auc = auc(levels[:index], excess_mass_scores[:index])
    return em_auc, float(np.mean(excess_mass_scores))


def mass_volume(
    axis_alpha: np.ndarray,
    volume_support: float,
    uniform_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> tuple:
    """Calculate Mass-Volume (MV) scores.

    Variables explained here: https://github.com/ngoix/EMMV_benchmarks/issues/2
    - "leb" refers to Lebesgue measure
    - "s" refers a measurable function
    - "X" refers to dataset

    :param np.ndarray axis_alpha: Alpha axis
    :param float volume_support: Volume of rectangle containing all data in X.
    :param np.ndarray uniform_scores: s(U), U = uniform samples from rectangle of all data in X, used to estimate Leb(s>t)
    :param np.ndarray anomaly_scores: s(X), i.e. s evaluated on a sample from underlying density f, used to estimate P(s>t)
    :return tuple: AUC and MV scores
    """

    n_samples = anomaly_scores.shape[0]
    sorted_indices = anomaly_scores.argsort()
    mass = 0.0
    count = 0
    score = anomaly_scores[sorted_indices[-1]]  # i.e. 'u'

    # Calculate MV scores
    mv_scores = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            count += 1
            score = anomaly_scores[sorted_indices[-count]]
            mass = 1.0 / n_samples * count  # sum(s_X > u)

        score_count = float((uniform_scores >= float(score)).sum())
        mv_scores[i] = (score_count / len(uniform_scores)) * volume_support

    return auc(axis_alpha, mv_scores), float(np.mean(mv_scores))
