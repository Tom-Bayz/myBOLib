# -*- coding: utf-8 -*-
u"""定数関数の定義."""
from __future__ import division
import numpy as np
from .base import Mymean
import copy


class Curve(Mymean):
	def __init__(self, curve=None):
		if curve is None:
			print("please input parameter!! @curve.py")
			return False
		self.name = "Const"
		self.curve = curve

	def getMean(self,trainID,const=0):
		# we never use "const"
		
		return np.array(self.curve)[trainID]

	def getDerMean(self):
		pass
