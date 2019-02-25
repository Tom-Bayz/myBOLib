"""
Definition various Kernel_functions

pyGPs
Copyright(c) by
Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014

07/12/2017 ver 1.0.0
Tomohiro Yonezu
"""
import numpy as np
import scipy.linalg as splinalg
import scipy.spatial.distance as spdist
import copy
from abc import ABCMeta, abstractmethod

class kernel_temp(object):
	u""" template of kernel_functions """

	__metaclass__ = ABCMeta

	@abstractmethod
	def getCovMat(self):
		u""" get covariance matrix """
		pass
