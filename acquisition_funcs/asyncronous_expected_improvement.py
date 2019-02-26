import numpy as np
import os
import time

##### scipy #####
from scipy.stats import truncnorm
from scipy.stats import norm 

##### others #####
from tqdm import tqdm
from matplotlib import pyplot as plt
from numba import jit
import math

##### multi-core processing #####
import multiprocessing as mproc

class asyEI(object):

	def __init__(self,sampleN=20,visualize=False):
		self.visualize = visualize
		self.sampleN = sampleN
		self.type = "Parallel BayesOpt"

	def ei(self,mean, var, current_max ,xi=0):
		I = mean - current_max
		z = (I - xi)/np.sqrt(var)
		ei = (I - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

		ei[ei!=ei] = 0 
		ei[ei<0] = 0

		return ei
	
	@jit
	def subproc(self,args):

		model = args[0]
		batch_point = args[1]
		halc_obs = args[2]

		model.halcY[batch_point] = halc_obs # set hallucinated observation
		model.fit(halc=True)
		halc_pred = model.predict(return_var=True,halc=True)

		halc_mu = halc_pred["mean"]
		halc_var = halc_pred["var"]

		sample_ei=self.ei(mean=halc_mu,var=halc_var,current_max= np.max(model.halcY[np.sort(model.halc_trainID)]))

		return sample_ei

	def get_nextID(self,model=None,batch_point=None):

		if model.name == "Gaussian process":
			return self.get_singleID(model=model, batch_point=batch_point)

		elif model.name == "Multi-task Gaussian process":
			return self.get_multiID(model=model, batch_point=batch_point)
		
		else:
			print("asyEI: Error invalid regression model!!")
			sys.exit()
	
	@jit
	def get_multiID(self,model,batch_point):
		print("this process is not prepared...")
		sys.exit()
	
	def get_singleID(self,model,batch_point):

		J = np.shape(batch_point)[0]
		N = np.shape(model.allX)[0]

		##### model prediction #####
		print(" >> gaussian process regression...")
		model.fit()
		g = model.predict(return_var=True)
		print("    complete")
		mu = g["mean"]
		var = g["var"]

		##### calcurate asyEI from here #####
		print(" >> start calc asyEI acquisition...")
		elapsed_time = time.time()

		joint = np.sort(batch_point)
		sub_pred = model.predict(at=joint,return_cov=True)

		sub_mu = sub_pred["mean"]
		sub_cov = sub_pred["cov"]

		##### hallucinated observe preparation #####
		hallc_obs = np.random.multivariate_normal(sub_mu,sub_cov,self.sampleN)
		model.halcY = np.copy(model.allY)
		model.halc_trainID = np.sort(np.r_[model.trainID,batch_point])

		##### separate hallc_obs for multi-core process #####
		##### compute 2nd term from here #####
		p = mproc.Pool(4)
		sample_ei = p.map(self.subproc,[ [model, batch_point,hallc_obs[s,:]] for s in range(self.sampleN)])
		sample_ei = np.array(sample_ei)
		p.close()

		asyei = np.mean(sample_ei,axis=0)

		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq = asyei

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

		fig, ax = plt.subplots(3, 1, figsize=(4, 7))

		ax[0].grid(True)
		ax[0].set_xlim(min(x),max(x))
		ax[0].set_xlabel(r"$x$",fontsize=15)
		ax[0].plot(x, y, "--", color="red") # true function
		ylim = np.array(ax[0].set_ylim())*1.1
		ax[0].set_ylim(ylim)
		ax[0].fill_between(x, Ucb, Lcb, color="blue", alpha=0.4,label="Uncertainty: "+r"$\sigma(x)$") # post var
		ax[0].plot(x, y, "--", color="red",label="unknown function:" + r"$f(x)$") # true function
		ax[0].plot(x, mu, "-", color="blue",label="prediction: "+r"$\mu(x)$") # post var

		##### plot training point and calcurating point #####
		ax[0].plot(x[model.trainID], y[model.trainID], "s", color="black",label="observed Data")
		ax[0].plot(x[batch_point], y[batch_point], "o",marker=">", color="black",label="batch points")
		ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

		##### plot acquisition function #####
		ax[1].set_xlim(min(x),max(x))
		ax[1].set_xlabel(r"$x$",fontsize=15)
		ax[1].grid(True)
		ax[1].plot(x,acq,color="#228b22",label="asyEI")
		ylim = ax[1].set_ylim()
		ax[1].plot([x[nextID]],[asyei[nextID]],"o",marker="*",markersize=10,color="red")
		ax[1].fill_between(x,ylim[0]*np.ones(N),acq,color="#228b22",alpha=0.5)
		ax[1].legend()

		##### optional #####
		ax[2].set_xlim(min(x),max(x))
		ax[2].set_xlabel(r"$x$",fontsize=15)
		ax[2].grid(True)
		ax[2].plot(x,sample_ei[0,:],color="blue",alpha=0.2,label="EI samples")
		for s in range(1,self.sampleN):
			ax[2].plot(x,sample_ei[s,:],color="blue",alpha=0.2)

		ax[2].legend()

		if not(os.path.exists("./fig_asyEI")):
			os.mkdir("./fig_asyEI")

		t = np.shape(model.trainID)[0]
		plt.savefig("./fig_asyEI/step"+"%04d"%t+".pdf",bbox_inches="tight")
		plt.close()
		
		
