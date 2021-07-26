import numpy as np
from sklearn.metrics import auc


def emmv_scores(trained_model, df, n_generated=10000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
	"""Get Excess-Mass (EM) and Mass Volume (MV) scores for unsupervised ML AD models.

	:param trained_model: Trained ML model with a 'decision_function' method 
	:param df: Pandas dataframe of features (X matrix)
	:param n_generated: , defaults to 10000
	:param alpha_min: Min value for alpha axis, defaults to 0.9
	:param alpha_max: Max value for alpha axis, defaults to 0.999
	:param t_max: Min EM value required, defaults to 0.9
	:return: A dictionary of two scores ('em' and 'mv')
	"""

	# Get limits and volume support.
	lim_inf = df.min(axis=0)
	lim_sup = df.max(axis=0)
	offset = 1e-60 # to prevent division by 0
	volume_support = (lim_sup - lim_inf).prod() + offset

	# Determine EM and MV parameters
	t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
	axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
	unif = np.random.uniform(lim_inf, lim_sup, size=(n_generated, df.shape[1]))

	# Get anomaly scores
	anomaly_score = trained_model.decision_function(df).reshape(1, -1)[0]
	s_unif = trained_model.decision_function(unif)

	# Get EM and MV scores
	AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif, anomaly_score, n_generated)
	AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif, anomaly_score, n_generated)

	# Return a dataframe containing EMMV information
	scores = {
		'em': np.mean(em),
		'mv': np.mean(mv),
	}
	return scores

def excess_mass(t, t_max, volume_support, s_unif, s_X, n_generated):
	EM_t = np.zeros(t.shape[0])
	n_samples = s_X.shape[0]
	s_X_unique = np.unique(s_X)
	EM_t[0] = 1.
	
	for u in s_X_unique:
		EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
						t * (s_unif > u).sum() / n_generated
						* volume_support)
	amax = np.argmax(EM_t <= t_max) + 1
	if amax == 1:
		amax = -1 # failed to achieve t_max
	AUC = auc(t[:amax], EM_t[:amax])
	return AUC, EM_t, amax


def mass_volume(axis_alpha, volume_support, s_unif, s_X, n_generated):
	n_samples = s_X.shape[0]
	s_X_argsort = s_X.argsort()
	mass = 0
	cpt = 0
	u = s_X[s_X_argsort[-1]]
	mv = np.zeros(axis_alpha.shape[0])
	for i in range(axis_alpha.shape[0]):
		while mass < axis_alpha[i]:
			cpt += 1
			u = s_X[s_X_argsort[-cpt]]
			mass = 1. / n_samples * cpt  # sum(s_X > u)
		mv[i] = float((s_unif >= float(u)).sum()) / n_generated * volume_support
	return auc(axis_alpha, mv), mv
