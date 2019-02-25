# -*- coding: utf-8 -*-
u"""Upper Confidence Bound."""
import numpy as np

def UCB_func(mu, sigma, beta=4.0):
	u"""Upper Confidence Bound."""
	return mu + np.sqrt(beta * sigma)
