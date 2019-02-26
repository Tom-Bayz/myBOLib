"""
Definition of Multi-task Gaussian Process Regression

12/01/2018
Written by Tomohiro Yonezu

ver.3.2.1
	can use multipul task-descriptor (e.g task_des1=> RDF, task_des2=> GB rotate angle
	 	normalize observation as internal process of MTGP class
"""
from __future__ import division
import numpy as np
import scipy.linalg as splinalg
import scipy
import copy
from . import kernel_funcs as kf
from . import mean_funcs as mf
from . import GPR
import sys
from matplotlib import pyplot as plt
from scipy.spatial import distance
import glob
import os
import pickle
import random
from numba import jit

class MTGPRegression(object):
	"""
	1. get Covariance matrix by using kernel class
	2. make predictibe distribution from 1
	"""

	def __init__(self, input_kernel=kf.RBF(), task_kernel=None, mean=mf.Const(), input_dim=-1, task_dim=-1, task_des_Num=1,in_norm=True):

		self.mean = copy.copy(mean) # prior mean

		##### initialization for input space #####
		self.input_kernel = copy.copy(input_kernel) # kernel function for input space
		self.input_dim = input_dim # dimention of input space

		##### initialization for tasks #####
		self.task_num = 0 # initialize the number of tasks by zero
		self.task_dim = np.atleast_1d(task_dim)
		self.task_des_Num = task_des_Num
		if task_kernel is None:
			self.task_kernel = []
			for tN in range(task_des_Num):
				self.task_kernel.append(copy.copy(kf.RBF(5,"median")))
		else:
			self.task_kernel = copy.copy(task_kernel)

		self.in_norm = in_norm

		self.allX = []
		self.allY = []
		self.allT = []

		##### initialization for otherwise #####
		self.gps = [] # the group of GPR instance
		self.pred_dist = [] # the prediction of all task

		self.chol = []
		self.alpha = []

		self.epsilon = -6

		##### instance name #####
		self.name = "Multi-task Gaussian process"

	def add_objFunc(self, name=None, allX=None, allY=None, task_descriptor=None, trainID=None,  cost=1):
		##### add objective function #####
		self.task_num += 1

		##### make new GPR instance #####
		new_gpr = GPR.GPRegression(allX = allX, allY=allY, input_dim=self.input_dim)

		##### set trainID #####
		if trainID is None:
			print("please initialize trainID")
			return 0;
		new_gpr.trainID = trainID

		if name is None:
			name = "func_"+str(self.task_num)
		new_gpr.name = name

		##### set cost #####
		new_gpr.cost = cost

		##### check input size #####
		n,d = np.shape(np.atleast_2d(allX))
		if d != self.input_dim:
			n,d = d,n
			allX = allX.T
		new_gpr.input_dim = self.input_dim


		##### check args #####
		if task_descriptor is None:
			print("Error [add_objFunc]: Task descriptor is necessary")
			return
		if (allX is None) or (allY is None):
			print("Error [add_objFunc]: test point or function value is empty")
			return
		if len(task_descriptor) != self.task_des_Num:
			print("Erro [add_objFunc]: Number of task-descriptors is not match the number you set")
			return


		new_gpr.task_descriptor = (task_descriptor)
		self.gps.append(new_gpr)

		print("Added function below")
		self.print_FuncInfo()

		return True


	def print_FuncInfo(self, k=None):
		if k is None:
			k =self.task_num-1

		if k > self.task_num:
			print("Error!! There is not function No."+str(k)+" yet.")
			return False

		print("=====[function No."+str("%2d"%k)+"]===========")
		print("| - name       : "+(self.gps[k]).name)
		print("| - input size : "+str(self.gps[k].N))
		print("| - #training  : "+str(np.shape(np.atleast_1d(self.gps[k].trainID))[0]))
		print("| - cost       : "+str(self.gps[k].cost))
		print("================================")

		return True
	@jit
	def model_select(self):

		##### prepaer allX #####
		if self.allX == []:
			self.allX = np.copy(self.gps[0].allX)
			for k in range(1,self.task_num):
				self.allX = np.r_[self.allX,self.gps[k].allX]

		##### prepare for kernel matrix of task descriptor #####
		if self.allT == []:
			for tN in range(self.task_des_Num):
				td = np.atleast_2d((self.gps[0].task_descriptor[tN]))
				T = np.copy(np.repeat(td,self.gps[0].N,axis=0))

				for k in range(1,self.task_num):
					td = np.atleast_2d((self.gps[k].task_descriptor[tN]))
					T = np.r_[T,np.repeat(td,self.gps[k].N,axis=0)]

				(self.allT).append(T)

		##### make candidate #####
		grid_num = int(5000**(1/(self.task_des_Num+1)))

		candidate = []
		base = 0.5*np.sum((np.max(self.allX,axis=0)-np.min(self.allX,axis=0))**2)
		c = base*(10**(np.linspace(-3,3,grid_num,endpoint=True)))[:,np.newaxis]
		candidate.append(c)

		for tN in range(self.task_des_Num):
			base = 0.5*np.sum((np.max(self.allT[tN],axis=0)-np.min(self.allT[tN],axis=0))**2)
			c = base*(10**(np.linspace(-3,3,grid_num,endpoint=True)))[:,np.newaxis]
			candidate.append(c)

		candidate = yz.gen_mesh(candidate)
		para_num = np.shape(candidate)[0]

		maxML = -np.inf
		ml_record = []
		best_param = []

		for param in range(para_num):
			self.input_kernel.hyp[1] = candidate[param,0]

			for tN in range(self.task_des_Num):
				self.task_kernel[tN].hyp[1] = candidate[param,tN+1]

			ml = self.fit()
			ml_record.append(ml)

			if maxML < ml:
				maxML = ml
				best_param = param

		self.input_kernel.hyp[1] = candidate[best_param,0]
		for tN in range(self.task_des_Num):
			self.task_kernel[tN].hyp[1] = candidate[best_param,tN+1]

		self.fit()

		return np.array(candidate), np.array(ml_record)
	@jit
	def fit(self):

		##### load training point #####
		all_trainID = np.sort(np.atleast_2d(self.gps[0].trainID))
		sn = self.gps[0].N
		for k in range(1,self.task_num):
			all_trainID = np.c_[all_trainID,np.sort(np.atleast_2d(self.gps[k].trainID))+sn]
			sn += self.gps[k].N

		all_trainID = np.array(all_trainID)[0,:].astype("int64")

		##### prepaer allX #####
		if self.allX == []:
			self.allX = np.copy(self.gps[0].allX)
			for k in range(1,self.task_num):
				self.allX = np.r_[self.allX,self.gps[k].allX]

		##### prepare trainX #####
		trainX = self.allX[all_trainID,:]

		##### prepare for kernel matrix of task descriptor #####
		if self.allT == []:
			for tN in range(self.task_des_Num):
				td = np.atleast_2d((self.gps[0].task_descriptor[tN]))
				T = np.copy(np.repeat(td,self.gps[0].N,axis=0))

				for k in range(1,self.task_num):
					td = np.atleast_2d((self.gps[k].task_descriptor[tN]))
					T = np.r_[T,np.repeat(td,self.gps[k].N,axis=0)]

				self.allT.append(T)

		##### taask des at training point #####
		trainT = []
		for tN in range(self.task_des_Num):
			trainT.append(self.allT[tN][all_trainID,:])

		##### prepare training Y #####
		ty = np.atleast_2d(self.gps[0].allY[np.sort(self.gps[0].trainID)])
		self.gps[0].avr = np.mean(ty)
		self.gps[0].std = np.std(ty)

		if self.in_norm:
			if self.gps[0].avr == self.gps[0].avr:
				ty = ty-self.gps[0].avr
			if (self.gps[0].std != 0) and (self.gps[0].std==self.gps[0].std):
				ty = ty/self.gps[0].std
		trainY = (np.atleast_2d(ty))

		for k in range(1,self.task_num):
			ty = np.atleast_2d(self.gps[k].allY[np.sort(self.gps[k].trainID)])
			self.gps[k].avr = np.mean(ty)
			self.gps[k].std = np.std(ty)

			if self.in_norm:
				if self.gps[k].avr == self.gps[k].avr:
					ty = ty-self.gps[k].avr
				if (self.gps[k].std != 0) and (self.gps[k].std==self.gps[k].std):
					ty = ty/self.gps[k].std
			trainY = np.c_[trainY,(np.atleast_2d(ty))]

		##### set prior mean #####
		self.mean.hyp = np.mean(trainY)
		residual = (trainY - self.mean.getMean(trainX))[0,:]

		##### process for median heuristics #####
		N = np.shape(self.allX)[0]

		if self.input_kernel.hyp[1] == "median": # for input kernel
			batchsize = 500
			ns1 = 0
			nact = N / batchsize
			unis = []
			while ns1 <= nact:
				act1 = np.array(range(ns1 * batchsize, np.minimum((ns1 + 1) * batchsize, N))).astype("int64")

				ns2=np.copy(ns1)
				while ns2 <= nact:
					#print("n1:"+str(ns1)+" n2:"+str(ns2))
					act2 = np.array(range(ns2 * batchsize, np.minimum((ns2 + 1) * batchsize, N))).astype("int64")
					unis = np.unique(np.r_[unis,np.unique(distance.cdist(self.allX[act1,:], self.allX[act2,:], "sqeuclidean"))])
					ns2 += 1
				ns1 += 1
			self.input_kernel.hyp[1] = np.median(np.unique(unis))

		for tN in range(self.task_des_Num):
			if self.task_kernel[tN].hyp[1] == "median": # for task kernel
				batchsize = 500
				ns1 = 0
				nact = N / batchsize
				unis = []
				while ns1 <= nact:
					act1 = np.array(range(ns1 * batchsize, np.minimum((ns1 + 1) * batchsize, N))).astype("int64")

					ns2=np.copy(ns1)
					while ns2 <= nact:
						#print("ns1:"+str(ns1)+" ns2:"+str(ns2))
						act2 = np.array(range(ns2 * batchsize, np.minimum((ns2 + 1) * batchsize, N))).astype("int64")
						unis = np.unique(np.r_[unis,np.unique(distance.cdist(self.allT[tN][act1,:], self.allT[tN][act2,:], "sqeuclidean"))])
						ns2 += 1
					ns1 += 1
				self.task_kernel[tN].hyp[1] = np.median(np.unique(unis))
				print("task-kernel median: "+str(self.task_kernel[tN].hyp[1]))

		##### calc LogMarginalLikelihood by LU decomposition #####
		self.chol = self.input_kernel.getCovMat(self.input_dim, trainX, self.allX,"train") + (10**self.epsilon)*np.eye(np.shape(trainX)[0])
		for tN in range(self.task_des_Num):
			self.chol *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], self.allT[tN],"train")
		self.chol = self._cholesky_(self.chol)
		self.alpha = scipy.linalg.cho_solve( (self.chol, False), residual)

		# LogMarginalLikelihood
		ML =  -0.5 * residual.dot(self.alpha) \
			- np.sum(np.log(np.diag(self.chol))) - 0.5 * len(self.alpha) * np.log(2 * np.pi)

		return ML
	#@jit
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
	@jit
	def predict(self,return_cov=False,return_var=False):

		##### prepaer allX #####
		if self.allX == []:
			self.allX = np.copy(self.gps[0].allX)
			for k in range(1,self.task_num):
				self.allX = np.r_[self.allX,self.gps[k].allX]

		if self.allT == []:
			for tN in range(self.task_des_Num):
				td = np.atleast_2d((self.gps[0].task_descriptor[tN]))
				T = np.copy(np.repeat(td,self.gps[0].N,axis=0))

				for k in range(1,self.task_num):
					td = np.atleast_2d((self.gps[k].task_descriptor[tN]))
					T = np.r_[T,np.repeat(td,self.gps[k].N,axis=0)]

				self.allT.append(T)

		all_trainID = np.sort(np.atleast_2d(self.gps[0].trainID))
		sn = self.gps[0].N
		for k in range(1,self.task_num):
			all_trainID = np.c_[all_trainID,np.sort(np.atleast_2d(self.gps[k].trainID))+sn]
			sn += self.gps[k].N

		all_trainID = np.array(all_trainID)[0,:].astype("int64")

		trainX = np.atleast_2d(self.allX[np.sort(all_trainID),:])

		alpha = self.alpha
		chol = self.chol

		##### taask des at training point #####
		trainT = []
		for tN in range(self.task_des_Num):
			trainT.append(self.allT[tN][all_trainID,:])


		testX = np.atleast_2d(self.allX)
		testT = []
		for tN in range(self.task_des_Num):
			testT.append(np.atleast_2d(self.allT[tN]))


		pred_dist = {}
		if return_cov:
			##### posterior mean & covariance #####
			K = self.input_kernel.getCovMat(self.input_dim, trainX, testX,"test")
			k = self.input_kernel.getCovMat(self.input_dim, trainX, testX, "cross")
			for tN in range(self.task_des_Num):
				K *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], testT[tN],"test")
				k *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], testT[tN],"cross")

			mean = (k.T).dot(alpha) + self.mean.getMean(testX)
			cov = K - (k.T).dot(scipy.linalg.cho_solve((chol,False), k))

			pred_dist["mean"] = mean
			pred_dist["cov"] = cov

		elif return_var:
			##### number of candidate #####
			N = np.shape(testX)[0]

			##### posterior mean only#####
			batchsize = 500
			ns = 0
			nact = N / batchsize
			mean = np.zeros(np.shape(self.allX)[0])
			var = np.zeros(np.shape(self.allX)[0])

			while ns <= nact:
				act = np.array(range(ns * batchsize, np.minimum((ns + 1) * batchsize, N))).astype("int64")

				K = self.input_kernel.getCovMat(self.input_dim, trainX, testX[act,:],"test")
				k = self.input_kernel.getCovMat(self.input_dim, trainX, testX[act,:], "cross")
				for tN in range(self.task_des_Num):
					K *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], (testT[tN])[act,:],"test")
					k *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], (testT[tN])[act,:],"cross")

				mean[act] = (k.T).dot(alpha) + self.mean.getMean(act)
				var[act] = np.diag(K) - np.diag((k.T).dot(scipy.linalg.cho_solve((chol,False), k)))
				ns += 1

			pred_dist["mean"] = mean
			pred_dist["var"] = var

		else:
			##### number of candidate #####
			N = np.shape(testX)[0]

			##### posterior mean only#####
			batchsize = 500
			ns = 0
			nact = N / batchsize
			mean = np.zeros(np.shape(self.allX)[0])


			while ns <= nact:
				act = np.array(range(ns * batchsize, np.minimum((ns + 1) * batchsize, N))).astype("int64")

				K = self.input_kernel.getCovMat(self.input_dim, trainX, testX[act,:],"test")
				k = self.input_kernel.getCovMat(self.input_dim, trainX, testX[act,:], "cross")
				for tN in range(self.task_des_Num):
					K *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], (testT[tN])[act,:],"test")
					k *= self.task_kernel[tN].getCovMat(self.task_dim[tN], trainT[tN], (testT[tN])[act,:],"cross")

				mean[act] = (k.T).dot(alpha) + self.mean.getMean(act)
				ns += 1

			pred_dist["mean"] = mean

		if return_cov:
			while True:
				val ,vec = scipy.linalg.eigh(pred_dist["cov"])
				if np.prod(val > 0):
					break
				print(" >> pred cov is not PSD!!")
				pred_dist["cov"] += 1e-9*np.eye(np.shape(pred_dist["cov"])[0])


		self.pred_dist = pred_dist

		sn = 0
		for k in range(self.task_num):
			each_mean = pred_dist["mean"][sn:(sn+self.gps[k].N)]

			self.gps[k].pred_dist = {"mean":each_mean}

			if return_cov:
				each_cov = pred_dist["cov"][sn:(sn+self.gps[k].N),:]
				each_cov = each_cov[:,sn:(sn+self.gps[k].N)]

				self.gps[k].pred_dist["cov"] = each_cov

			elif return_var:
				each_var = pred_dist["var"][sn:(sn+self.gps[k].N)]

				self.gps[k].pred_dist["var"] = each_var

			sn += self.gps[k].N

		""" return predictive distribution """
		return pred_dist

def marge_grid(x1,x2):

    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    x1size = np.shape(x1)[0]
    x2size = np.shape(x2)[0]

    X = np.c_[np.repeat(x1,x2size,axis=0),np.tile(x2,(x1size,1))]

    return X

def gen_mesh(argX):

    if np.shape(argX)[0] == 0:
        print("You should at least 2 arguments.")
        sys.exit()

    X = argX[0]

    for x in argX[1:]:
        X = marge_grid(X,x)

    return X
