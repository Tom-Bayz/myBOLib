# -*- coding: utf-8 -*-
u"""Expected Improvement."""
from __future__ import division
import numpy as np
from scipy.stats import norm


def Thompson_samp(mu, cov, sample_N=100 ,xi=0):
	u""" Expected Improvement !!MAXimization!! """
	sample = np.random.multivariate_normal(mu,cov,sample_N)

	TPS = np.zeros(np.shape(mu)[0])
	for i in range(sample_N):
		inx = np.argmax(sample[i,:])
		TPS[inx] += 1

	return TPS/sample_N
