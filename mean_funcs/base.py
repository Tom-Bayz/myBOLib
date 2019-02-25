# -*- coding: utf-8 -*-
u"""平均関数の定義."""
import numpy as np
import copy
from abc import ABCMeta, abstractmethod


class Mymean(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def getMean(self):
		pass

	@abstractmethod
	def getDerMean(self):
		pass
