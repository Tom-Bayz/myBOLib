"""
Definition of RBF(Gaussian) kernel function

Copyright(c) by  Kenta Kanamori 2017

07/12/2017 ver 1.0.0
Tomohiro Yonezu
"""
from __future__ import division
import numpy as np
import scipy.spatial.distance as dist
from . import kernel_temp as kmp
import copy
import math
import sys
from matplotlib import pyplot as plt
import os
from numba import jit


class RBF(kmp):
	"""
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Gaussian Kernel function
	k(x,x'):=ell * exp(||x-x'||^2 / (2 * var)).

	ell : variance magnitude
	var : length scale
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	"""

	def __init__(self, ell=1, var="median"):

		""" parameter check """
		if (var is "median") or (var is "mean"):
			#print("RBF kernel median heuristics")
			pass

		elif (ell <=0) or (var <= 0):
			print("RBF_kernel Error : parameter must be positive")
			return

		""" set parameters """
		self.hyp = [copy.copy(ell), copy.copy(var)]
		self.name = "RBF"

	@jit
	def getCovMat(self, x_dim, trainX=None, testX=None, mode=None):
		""" make kernel matrix """
		ell = self.hyp[0]
		var = self.hyp[1]

		""" check if dimention arg x_dim is correct """
		if (x_dim <= 0):
			print("["+self.name+"] Dimention of input must be positive.")
			return -1
		if (math.modf(x_dim)[0] != 0):
			print("["+self.name+"] Dimention of input must be integer.")
			return -1

		""" check if dimension of input is correct """
		X = np.atleast_2d(testX)
		n,d = np.shape(X)
		if (d!= x_dim) and (n == x_dim):
			print("transposed")
			X = X.T
			n,d = d,n
		if (d!= x_dim) and (n != x_dim):
			print("["+self.name+"] Error!! dimension setting is not correct")
			return -1

		if var == 0:
			var = 1

		""" make kernel matrix """
		if mode is "test":
			M = dist.cdist(testX, testX, "sqeuclidean")/var
			M = np.atleast_2d(ell * np.exp(-0.5 * M))
		if mode is "cross":
			M = dist.cdist(trainX, testX, "sqeuclidean")/var
			M = np.atleast_2d(ell * np.exp(-0.5 * M))
		if mode is "train":
			M = dist.cdist(trainX, trainX, "sqeuclidean")/var
			M = np.atleast_2d(ell * np.exp(-0.5 * M))
		return M

if __name__ == "__main__":

	x = np.atleast_2d(np.linspace(-3,3,130)).T
	kernel = RBF(1,2)

	M = kernel.getCovMat(1,trainID=np.array(range(60)),testX=x,mode="cross")

	print(np.shape(M))
	plt.imshow(M)
	plt.show()
