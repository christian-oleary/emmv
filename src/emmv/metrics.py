"""Excess-Mass and Mass-Volume scores for unsupervised ML AD models."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def default_scoring_func(model, df):
    """Default scoring function for anomaly detection models."""
    return model.decision_function(df)


def emmv_scores(
    model,
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

    # Specify limits, volume, and levels for uniform sampling
    lim_inf, lim_sup, volume, levels = calculate_limits(df)

    # Perform uniform sampling
    try:
        uniform_sample = np.random.uniform(lim_inf, lim_sup, size=(n_generated, df.shape[1]))
    except IndexError:  # 1D array
        uniform_sample = np.random.uniform(lim_inf, lim_sup, size=n_generated)

    # Get anomaly scores
    uniform_scores = scoring_func(model, uniform_sample)
    anomaly_scores = scoring_func(model, df)  # .reshape(1, -1)[0]

    # Calculate and return EM and MV scores
    return (
        float(np.mean(excess_mass(levels, em_min, volume, uniform_scores, anomaly_scores))),
        float(np.mean(mass_volume(alpha_min, alpha_max, volume, uniform_scores, anomaly_scores))),
    )


def calculate_limits(df: pd.DataFrame, offset: float = 1e-60) -> tuple:
    """Specify a rectangle containing all data in X.

    :param pd.DataFrame df: Input dataframe
    :param float offset: Offset to prevent division by 0, defaults to 1e-60
    :return float: Volume of rectangle containing all data in X
    """
    # Min and max values of each feature
    lim_inf = df.min(axis=0)
    lim_sup = df.max(axis=0)

    # Volume of rectangle containing all data in X
    volume = float((lim_sup - lim_inf).prod()) + offset

    # An "array of levels, on which we want to evaluate
    # EM_s(t) on samples X from an underlying density f."
    levels = np.arange(0, 100 / volume, 0.01 / volume)

    return lim_inf, lim_sup, volume, levels


def excess_mass(
    levels: np.ndarray,
    em_min: float,
    volume: float,
    uniform_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> np.ndarray:
    """Calculate Excess-Mass scores.

    Variables explained here: https://github.com/ngoix/EMMV_benchmarks/issues/2
    - "t" ('levels' in this code): levels to evaluate EM_s(t) on samples X from underlying density f
    - "leb" refers to Lebesgue measure
    - "s" refers a measurable function
    - "X" refers to dataset
    - "uniform_scores" represents uniform samples from rectangle of all data in X

    :param np.ndarray levels: Levels on which to evaluate EM_s(t) on samples X
    :param float em_min: Beginning of EM curve.
    :param float volume: Volume of rectangle containing all data in X.
    :param np.ndarray uniform_scores: s(U), used to estimate Leb(s>t)
    :param np.ndarray anomaly_scores: s(X), s evaluated on a sample, used to estimate P(s>t)
    :return np.ndarray: EM scores
    """
    n_samples = anomaly_scores.shape[0]
    unique_anomaly_scores = np.unique(anomaly_scores)
    excess_mass_scores = np.zeros(levels.shape[0])
    excess_mass_scores[0] = 1.0

    for score in unique_anomaly_scores:
        anomaly_fraction = 1.0 / n_samples * (anomaly_scores > score).sum()
        uniform = levels * (uniform_scores > score).sum() / len(uniform_scores)
        excess_mass_scores = np.maximum(excess_mass_scores, anomaly_fraction - (uniform * volume))

    index = int(np.argmax(excess_mass_scores <= em_min).flatten()[0]) + 1
    if index == 1:
        logger.warning('Failed to achieve em_min')
        index = -1

    # em_auc = auc(levels[:index], excess_mass_scores[:index])
    return excess_mass_scores


def mass_volume(
    alpha_min: float,
    alpha_max: float,
    volume: float,
    uniform_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    alpha_count: int = 1000,
) -> np.ndarray:
    """Calculate Mass-Volume (MV) scores.

    Variables explained here: https://github.com/ngoix/EMMV_benchmarks/issues/2
    - "t" ('levels' in this code): levels to evaluate EM_s(t) on samples X from underlying density f
    - "leb" refers to Lebesgue measure
    - "s" refers a measurable function
    - "X" refers to dataset
    - "uniform_scores" represents uniform samples from rectangle of all data in X

    :param float alpha_min: Minimum alpha axis value
    :param float alpha_max: Maximum alpha axis value
    :param float volume: Volume of rectangle containing all data in X.
    :param np.ndarray uniform_scores: s(U), used to estimate Leb(s>t)
    :param np.ndarray anomaly_scores: s(X), s evaluated on a sample, used to estimate P(s>t)
    :param int alpha_count: Number of levels
    :return np.ndarray: MV scores
    """

    n_samples = anomaly_scores.shape[0]
    sorted_indices = anomaly_scores.argsort()
    mass = 0.0
    count = 0
    score = anomaly_scores[sorted_indices[-1]]  # i.e. 'u'
    axis_alpha = np.linspace(alpha_min, alpha_max, alpha_count)

    # Calculate MV scores
    mv_scores = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            count += 1
            score = anomaly_scores[sorted_indices[-count]]
            mass = 1.0 / n_samples * count  # sum(s_X > u)

        score_count = float((uniform_scores >= float(score)).sum())
        mv_scores[i] = (score_count / len(uniform_scores)) * volume

    # mv_auc = auc(axis_alpha, mv_scores)
    return mv_scores
