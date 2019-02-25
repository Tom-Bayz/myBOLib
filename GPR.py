"""
Definition of Gaussian Process Regression

23/01/2019 ver 1.0.0
Tomohiro Yonezu
"""
import numpy as np
import scipy.linalg as splinalg
import scipy.spatial.distance as distance
import copy
from . import kernel_funcs as kf
from . import mean_funcs as mf
import sys
import matplotlib.pyplot as plt
from numba import jit
import scipy

class GPRegression(object):
	"""
	1. get Covariance matrix by using kernel class
	2. make predictibe distribution from 1
	"""

	def __init__(self, allX, allY, kernel=kf.RBF(), mean=mf.Const(), input_dim=-1,in_norm=True,out_norm=True):
		self.name = "Gaussian Process"
		self.kernel = copy.copy(kernel)
		self.mean = copy.copy(mean)

		if input_dim != np.shape(np.atleast_2d(allX))[1]:
			print("Something wrong with input matrix. please check it.")
			print("Input dim from data : "+str(np.shape(np.atleast_2d(allX))[1]))
			print("Input dim you set : "+str(input_dim))
			sys.exit()

		self.N = np.shape(allX)[0]
		self.allX = np.atleast_2d(allX)
		self.allY = allY
		self.input_dim = input_dim
		self.trainID = []
		self.pred_dist = []
		self.task_descriptor = [] # for MTGPR

		self.cost = []
		self.epsilon = -5

		##### for nomalization #####
		self.in_norm = in_norm
		self.out_norm = out_norm
		self.avr = np.nan
		self.std = np.nan

		##### for fit method #####
		self.nlZ = []
		self.chol = []
		self.alpha = []

		##### for hallucinated predict #####
		self.halcY = []
		self.halc_trainID = []
		self.halc_nlZ = []
		self.halc_chol = []
		self.halc_alpha = []
		self.halc_avr = np.nan
		self.halc_std = np.nan

		self.at = np.array([])


	@jit
	def model_select(self,candidate=None,halc=False):

		if candidate is None:
			base = np.sum((np.max(self.allX,axis=0)-np.min(self.allX,axis=0))**2)
			candidate = base*(10**(np.linspace(-6,1,100,endpoint=True)))

		maxML = -np.inf
		ml_record = []
		best_param = []

		if halc:
			for param in candidate:
				self.kernel.hyp[1] = param

				ml = self.fit(halc=True)
				ml_record.append(ml)

				if maxML < ml:
					maxML = ml
					best_param = param

			self.kernel.hyp[1] = best_param
			self.fit(halc=True)

		else:
			for param in candidate:
				self.kernel.hyp[1] = param

				ml = self.fit()
				ml_record.append(ml)

				if maxML < ml:
					maxML = ml
					best_param = param

			self.kernel.hyp[1] = best_param
			self.fit()

		return np.array(candidate), np.array(ml_record)

	@jit
	def fit(self,halc=False):

		if halc:
			##### load training point #####
			trainID = np.sort(self.halc_trainID)

			##### trining feature value #####
			trainX = self.allX[trainID,:]

			##### training function value #####
			if self.in_norm:
				trainY = self.halcY[np.sort(trainID)]
				self.halc_avr = np.mean(trainY)
				self.halc_std = np.std(trainY)

				trainY = trainY-self.avr

				if self.halc_std != 0 and self.halc_std==self.halc_std:
					trainY = trainY/self.halc_std
			else:
				trainY = self.halcY[np.sort(trainID)]

			##### set prior mean #####
			self.mean.hyp = np.mean(trainY)
			residual = trainY - self.mean.getMean(trainX)

			##### process for median heuristics #####
			if self.kernel.hyp[1] == "median":
				batchsize = 1000
				ns1 = 0
				nact = self.N / batchsize
				unis = []
				while ns1 <= nact:
					act1 = np.array(range(ns1 * batchsize, np.minimum((ns1 + 1) * batchsize, self.N))).astype("int64")

					ns2=np.copy(ns1)
					while ns2 <= nact:
						#print("n1:"+str(ns1)+" n2:"+str(ns2))
						act2 = np.array(range(ns2 * batchsize, np.minimum((ns2 + 1) * batchsize, self.N))).astype("int64")
						unis = np.unique(np.r_[unis,np.unique(distance.cdist(self.allX[act1,:], self.allX[act2,:], "sqeuclidean"))])
						ns2 += 1
					ns1 += 1
				self.kernel.hyp[1] = np.median(np.unique(unis))

			##### calc LogMarginalLikelihood by LU decomposition #####
			self.halc_chol = self.kernel.getCovMat(self.input_dim, trainX, self.allX,"train") + (10**self.epsilon)*np.eye(np.shape(trainX)[0])
			self.halc_chol = self._cholesky_(self.halc_chol)
			self.halc_alpha = scipy.linalg.cho_solve( (self.halc_chol, False), residual )

			# LogMarginalLikelihood
			ML =  -0.5 * residual.dot(self.halc_alpha) \
				- np.sum(np.log(np.diag(self.halc_chol))) - 0.5 * len(self.halc_alpha) * np.log(2 * np.pi)

		else:
			##### load training point #####
			trainID = np.sort(self.trainID)

			##### trining feature value #####
			trainX = self.allX[trainID,:]

			##### training function value #####
			if self.in_norm:
				trainY = self.allY[np.sort(self.trainID)]
				self.avr = np.mean(trainY)
				self.std = np.std(trainY)

				trainY = trainY-self.avr

				if self.std != 0 and self.std==self.std:
					trainY = trainY/self.std
			else:
				trainY = self.allY[np.sort(self.trainID)]

			##### set prior mean #####
			self.mean.hyp = np.mean(trainY)
			residual = trainY - self.mean.getMean(trainX)


			##### process for median heuristics #####
			if self.kernel.hyp[1] == "median":
				batchsize = 1000
				ns1 = 0
				nact = self.N / batchsize
				unis = []
				while ns1 <= nact:
					act1 = np.array(range(ns1 * batchsize, np.minimum((ns1 + 1) * batchsize, self.N))).astype("int64")

					ns2=np.copy(ns1)
					while ns2 <= nact:
						#print("n1:"+str(ns1)+" n2:"+str(ns2))
						act2 = np.array(range(ns2 * batchsize, np.minimum((ns2 + 1) * batchsize, self.N))).astype("int64")
						unis = np.unique(np.r_[unis,np.unique(distance.cdist(self.allX[act1,:], self.allX[act2,:], "sqeuclidean"))])
						ns2 += 1
					ns1 += 1
				self.kernel.hyp[1] = np.median(np.unique(unis))

			##### calc LogMarginalLikelihood by LU decomposition #####
			self.chol = self.kernel.getCovMat(self.input_dim, trainX, self.allX,"train") + (10**self.epsilon)*np.eye(np.shape(trainX)[0])
			self.chol = self._cholesky_(self.chol)
			self.alpha = scipy.linalg.cho_solve( (self.chol, False), residual )

			# LogMarginalLikelihood
			ML =  -0.5 * residual.dot(self.alpha) \
				- np.sum(np.log(np.diag(self.chol))) - 0.5 * len(self.alpha) * np.log(2 * np.pi)

		return ML

	def _cholesky_(self, cov):
		n, _ = cov.shape
		while 1:
			try:
				chol = splinalg.cholesky(cov)
				break
			except np.linalg.LinAlgError:
				print("cov matrix is modified.")
				val, vec = scipy.linalg.eigh(cov)
				cov = np.dot(val * np.maximum(vec, np.ones(len(vec)) * 1e-6)[:, np.newaxis], val.T)
		return chol


	#@jit
	def predict(self,at=None,return_cov=False,return_var=False,halc=False):

		if halc:
			trainX = np.atleast_2d(self.allX[np.sort(self.halc_trainID),:])
			alpha = self.halc_alpha
			chol = self.halc_chol
		else:
			trainX = np.atleast_2d(self.allX[np.sort(self.trainID),:])
			alpha = self.alpha
			chol = self.chol

		if at is None:
			testX = np.atleast_2d(self.allX)
		else:
			testX = np.atleast_2d(self.allX[at,:])

		##### culcuration of predictive distribution #####
		pred_dist = {}

		if return_cov:
			##### posterior mean & covariance #####
			K = self.kernel.getCovMat(self.input_dim, trainX, testX,"test")
			k = self.kernel.getCovMat(self.input_dim, trainX, testX, "cross")

			mean = (k.T).dot(alpha) + self.mean.getMean(testX)
			cov = K - (k.T).dot(scipy.linalg.cho_solve((chol,False), k))

			pred_dist["mean"] = mean
			pred_dist["cov"] = cov

		elif return_var:
			##### posterior mean only#####
			batchsize = 5000
			ns = 0
			nact = self.N / batchsize
			mean = np.zeros(self.N)
			var = np.zeros(self.N)

			while ns <= nact:
				act = np.array(range(ns * batchsize, np.minimum((ns + 1) * batchsize, self.N))).astype("int64")

				K = self.kernel.getCovMat(self.input_dim, trainX, testX[act,:],"test")
				k = self.kernel.getCovMat(self.input_dim, trainX, testX[act,:],"cross")

				mean[act] = (k.T).dot(alpha) + self.mean.getMean(act)
				var[act] = np.diag(K) - np.diag((k.T).dot(scipy.linalg.cho_solve((chol,False), k)))
				ns += 1

			#var[var < epsilon] = epsilon
			pred_dist["mean"] = mean
			pred_dist["var"] = var

		else:
			##### posterior mean only#####
			batchsize = 5000
			ns = 0
			nact = self.N / batchsize
			mean = np.zeros(self.N)

			while ns <= nact:
				act = np.array(range(ns * batchsize, np.minimum((ns + 1) * batchsize, self.N))).astype("int64")

				K = self.kernel.getCovMat(self.input_dim, trainX, testX[act,:],"test")
				k = self.kernel.getCovMat(self.input_dim, trainX, testX[act,:],"cross")

				mean[act] = (k.T).dot(alpha) + self.mean.getMean(act)
				ns += 1

			pred_dist["mean"] = mean

		if self.out_norm and self.in_norm:

			if self.std != 0 and self.std == self.std:
				pred_dist["mean"] = pred_dist["mean"]*self.std
				if return_cov:
					pred_dist["cov"] = pred_dist["cov"]*(self.std**2)

				if return_var:
					pred_dist["var"] = pred_dist["var"]*(self.std**2)

			pred_dist["mean"] = pred_dist["mean"] + self.avr

		if return_cov:
			while True:
				val ,vec = scipy.linalg.eigh(pred_dist["cov"])
				if np.prod(val > 0):
					break
				print(" >> pred cov is not PSD!!")
				#print(pred_dist["cov"])
				pred_dist["cov"] += 1e-9*np.eye(np.shape(pred_dist["cov"])[0])

		if at is None:
			self.pred_dist = pred_dist

		""" return predictive distribution """
		return pred_dist
