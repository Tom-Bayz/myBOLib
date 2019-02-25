##### import package #####
import yonezu as yz
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from numba import jit
import time
import multiprocessing as mproc
import math

class mRMR(object):

	def __init__(self,sampleN=20,max_dist="gaussian",visualize=False):
		self.visualize = visualize
		self.sampleN = sampleN
		self.max_dist = max_dist
		self.name = "mRMR"
		self.type = "Parallel BayesOpt"

	def subproc(self,args):

		mu = args[0]
		var = args[1]
		fstar = args[2]

		##### Entropy of truncated normal for each fstar #####
		a = mu-5*np.sqrt(var)
		b = fstar

		alpha = (a - mu)/np.sqrt(var)
		beta = (b - mu)/np.sqrt(var)
		Z = norm.cdf(beta) - norm.cdf(alpha)

		H = 0.5*np.log(2*np.pi*np.exp(1)*var*Z) + (alpha*norm.pdf(alpha) - beta*norm.pdf(beta))/Z

		if H != H:
			print("   entropy is NaN!!")
			pass

		return H


	#@jit
	def get_nextID(self,model=None,batch_point=None):

		J = np.shape(batch_point)[0]
		N = np.shape(model.allX)[0]

		current_max = np.max(model.allY[np.sort(model.trainID)])

		##### check model #####
		if model.name != "Gaussian Process":
			print("PMES can be calcurated only from Gaussian Process model.")
			return False

		elapsed_time = time.time()

		##### model prediction #####
		print(" >> gaussian process regression...")
		model.fit()
		g = model.predict(return_var=True)
		print("    complete")
		mu = g["mean"]
		var = g["var"]

		##### sampling global max #####
		count = 0
		while True:

			mrmr_start = time.time()

			if self.max_dist == "gaussian":
				x_plus = np.argmax(mu)
				Fstar_samples = np.random.normal(mu[x_plus],var[x_plus],self.sampleN)
				Fstar_samples = Fstar_samples[Fstar_samples >= current_max]
				count += self.sampleN
			elif self.max_dist == "true max":
				Fstar_samples = np.array(2*[np.max(model.allY)])
			else:
				print("invalid distribution of max!!")
				return False

			if count > 10000:
				Fstar_samples = [current_max]
				break

			if np.shape(Fstar_samples)[0] != 0:
				break

		##### calcurate PMES from here #####
		mrmr = np.ones(N)
		mrmr[np.sort(model.trainID)] = -np.inf
		mrmr[np.sort(batch_point)] = -np.inf

		Relevance = np.copy(mrmr)
		Redundancy = np.copy(mrmr)

		print(" >> start calc mRMR acquisition...")

		for i in np.array(np.where(mrmr == 1)[0]):


			##### compute Relevance #####
			p = mproc.Pool(4)
			trunc_norm_Entropy = p.map(self.subproc,[ [mu[i],var[i],fstar] for fstar in Fstar_samples ])
			trunc_norm_Entropy = np.array(trunc_norm_Entropy)
			trunc_norm_Entropy = trunc_norm_Entropy[trunc_norm_Entropy == trunc_norm_Entropy]
			p.close()

			M = np.shape(trunc_norm_Entropy)[0]
			cnd_entropy = np.sum(trunc_norm_Entropy)

			if M == 0:
				trunc_norm_Entropy = -np.inf
			else:
				trunc_norm_Entropy = np.sum(trunc_norm_Entropy)/M

			MR = 0.5*np.log(2*np.pi*np.exp(1)*var[i]) - trunc_norm_Entropy

			##### compute Redundancy #####
			mR = 0
			for x_q in batch_point:

				joint = np.sort([x_q,i])
				sub_pred = model.predict(at=joint,return_cov=True)
				sub_cov = sub_pred["cov"]

				mR += ( 0.5*np.log(2*np.pi*np.exp(1)*var[i]) + 0.5*np.log(2*np.pi*np.exp(1)*var[x_q])\
				- 0.5*(2 + 2*np.log(2*np.pi) + np.log(np.linalg.det(sub_cov)) ) )

			mR = mR/J

			##### compute mRMR
			Relevance[i] = MR
			Redundancy[i] = mR
			mrmr[i] = MR - mR

		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq = mrmr

		##### maximize acquisition #####
		nextID = np.argmax(acq)

		if np.shape([nextID])[0] > 1:
			nextID = np.random.choice(nextID)

		#"""
		##### plot true function and posterior #####
		if self.visualize and (np.shape(model.allX)[1] == 1):

			Lcb = mu - np.sqrt(var)
			Ucb = mu + np.sqrt(var)

			x = model.allX[:,0]
			y = model.allY

			fig, ax = plt.subplots(3, 1, figsize=(5, 7))

			ax[0].grid(True)
			ax[0].set_xlim(min(x),max(x))
			ax[0].set_xlabel(r"$x$",fontsize=15)
			ax[0].plot(x, y, "--", color="red") # true function
			ylim = np.array(ax[0].set_ylim())*1.1
			ax[0].set_ylim(ylim)
			ax[0].fill_between(x, Ucb, Lcb, color="blue", alpha=0.4,label="confidence: "+r"$\sigma^2(x)$") # post var
			ax[0].plot(x, y, "--", color="red",label="unknown function:" + r"$f(x)$") # true function
			ax[0].plot(x, mu, "-", color="blue",label="predictive mean: "+r"$\mu(x)$") # post var

			##### plot training point and calcurating point #####
			ax[0].plot(x[model.trainID], y[model.trainID], "s", color="black",label="observed")
			ax[0].plot(x[batch_point], y[batch_point], "o",marker=">", color="black",label="processing")
			ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)


			##### plot acquisition function #####
			ax[1].set_xlim(min(x),max(x))
			ax[1].set_xlabel(r"$x$",fontsize=15)
			ax[1].grid(True)
			ax[1].plot(x,acq,color="#228b22",label="mRMR")
			ylim = ax[1].set_ylim()
			ax[1].plot([x[nextID]],[acq[nextID]],"o",marker="*",markersize=10,color="red")
			ax[1].fill_between(x,ylim[0]*np.ones(N),acq,color="#228b22",alpha=0.5)
			ax[1].legend()

			ax[2].set_xlim(min(x),max(x))
			ax[2].set_xlabel(r"$x$",fontsize=15)
			ax[2].grid(True)
			ax[2].plot(x,Relevance,color="red",label="Relevance")
			ax[2].plot(x,Redundancy,color="blue",label="Redundancy")
			ylim = ax[2].set_ylim()
			ax[2].fill_between(x,ylim[0]*np.ones(N),Relevance,color="red",alpha=0.5)
			ax[2].fill_between(x,ylim[0]*np.ones(N),Redundancy,color="blue",alpha=0.5)
			ax[2].legend()

			if not(os.path.exists("./fig_mRMR")):
				os.mkdir("./fig_mRMR")

			t = np.shape(model.trainID)[0]
			plt.savefig("./fig_mRMR/step"+"%04d"%t+".pdf",bbox_inches="tight")
			plt.close()
		#"""

		return nextID, max(acq)
