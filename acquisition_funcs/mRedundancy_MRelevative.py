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

	def __init__(self,sampleN=20,max_dist="gaussian",visualize=False,type="sub"):
		self.visualize = visualize
		self.sampleN = sampleN
		self.max_dist = max_dist
		self.redundancy_type = type
		self.name = "mRMR"
		self.type = "Parallel BayesOpt"

	def subproc(self,args):

		model = args[0]
		Fstar_samples = args[1]
		batch_point = args[2]
		i = args[3]

		J = np.shape(batch_point)[0]

		mu = model.pred_dist["mean"]
		var = model.pred_dist["var"]

		trunc_norm_Entropy = 0
		M = 0

		for fstar in Fstar_samples:

			##### Entropy of truncated normal for each fstar #####
			a = mu[i]-5*np.sqrt(var[i])
			b = fstar

			alpha = (a - mu[i])/np.sqrt(var[i])
			beta = (b - mu[i])/np.sqrt(var[i])
			Z = norm.cdf(beta) - norm.cdf(alpha)

			H = np.log(np.sqrt(2*np.pi*np.exp(1)*var[i])*Z) + (alpha*norm.pdf(alpha) - beta*norm.pdf(beta))/(2*Z)
			#H = np.log(np.sqrt(2*np.pi*np.exp(1)*var[i])*Z) + (0 - beta*norm.pdf(beta))/(2*Z)

			if H != H:
				print("   entropy is NaN!!")
				pass
			else:
				trunc_norm_Entropy += H
				M += 1

		if M == 0:
			trunc_norm_Entropy = -np.inf
		else:
			trunc_norm_Entropy = trunc_norm_Entropy/M

		Relevance = 0.5*np.log(2*np.pi*np.exp(1)*var[i]) - trunc_norm_Entropy

		##### compute Redundancy #####
		Redundancy = 0
		for x_q in batch_point:

			joint = np.sort([x_q,i])
			sub_pred = model.predict(at=joint,return_cov=True)
			sub_cov = sub_pred["cov"]

			Redundancy += ( 0.5*np.log(2*np.pi*np.exp(1)*var[i]) + 0.5*np.log(2*np.pi*np.exp(1)*var[x_q])\
			- 0.5*np.log(((2*np.pi*np.exp(1))**2)*np.linalg.det(sub_cov)) )

		Redundancy = Redundancy/J

		return Relevance,Redundancy
	
	def get_nextID(self,model=None,batch_point=None):

		if model.name == "Gaussian process":
			return self.get_singleID(model=model, batch_point=batch_point)

		elif model.name == "Multi-task Gaussian process":
			return self.get_multiID(model=model, batch_point=batch_point)
		
		else:
			print("BUCB: Error invalid regression model!!")
			sys.exit()

	def get_multiID(self,model,batch_point):
		print("this process is not prepared...")
		sys.exit()
	
	def get_singleID(self,model,batch_point):

		J = np.shape(batch_point)[0]
		N = np.shape(model.allX)[0]

		current_max = np.max(model.allY[np.sort(model.trainID)])

		elapsed_time = time.time()

		##### model prediction #####
		print(" >> gaussian process regression...")
		model.fit()
		g = model.predict(return_var=True)
		print("    complete")
		mu = g["mean"]
		var = g["var"]

		##### sampling global max #####
		#count = 0
		#while True:

		mrmr_start = time.time()

		if self.max_dist == "gaussian":
			x_plus = np.argmax(mu)
			Fstar_samples = np.random.normal(mu[x_plus],var[x_plus],self.sampleN)
			Fstar_samples = Fstar_samples[Fstar_samples >= current_max]
			#count += self.sampleN
		elif self.max_dist == "true max":
			Fstar_samples = np.array(2*[np.max(model.allY)])
		elif self.max_dist == "GP_sample":
			sample_index = np.random.choice(range(N),min(5000,N),replace=False)
			sample_g = model.predict(at=sample_index, return_cov=True)
			Fstar_samples = np.random.multivariate_normal(sample_g["mean"],sample_g["cov"],self.sampleN)
			Fstar_samples = np.max(Fstar_samples,axis=1)
			Fstar_samples = Fstar_samples[Fstar_samples >= current_max]
			print(Fstar_samples)
			#count += self.sampleN
		else:
			print("invalid distribution of max!!")
			return False

		if np.shape(Fstar_samples)[0] == 0:
			Fstar_samples = [current_max]

		##### calcurate PMES from here #####
		mrmr = np.ones(N)
		mrmr[np.sort(model.trainID)] = -np.inf
		mrmr[np.sort(batch_point)] = -np.inf

		Relevance = np.copy(mrmr)
		Redundancy = np.copy(mrmr)

		print(" >> start calc mRMR acquisition...")

		index = np.array(np.where(mrmr == 1)[0])

		p = mproc.Pool(8)
		result = p.map(self.subproc,[[model,Fstar_samples,batch_point,i] for i in index])
		result  = np.array(result)
		p.close()

		Relevance[index] = result[:,0]
		Redundancy[index] = result[:,1]

		if self.redundancy_type == "sub":
			mrmr[index] = result[:,0] - result[:,1]
		elif self.redundancy_type == "div":
			mrmr[index] = result[:,0]/(result[:,1]+1)
		else:
			print("  >> mRMR: please check redundancy_type")
			sys.exit()

		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq = mrmr

		##### maximize acquisition #####
		nextID = np.argmax(acq)

		if np.shape([nextID])[0] > 1:
			nextID = np.random.choice(nextID)

		return nextID, max(acq)
	
	def plot():
		
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
