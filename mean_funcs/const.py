# -*- coding: utf-8 -*-
u"""定数関数の定義."""
from __future__ import division
import numpy as np
from .base import Mymean
import copy


class Const(Mymean):
	def __init__(self, const=0):
		self.hyp = const
		self.name = "Const"

	def getMean(self, testX):
		size = np.shape(np.atleast_1d(testX))[0]
		return np.array([self.hyp] * size)

	def getDerMean(self):
		pass
