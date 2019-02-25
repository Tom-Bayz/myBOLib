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


class LPEI(object):

	def __init__(self,sampleN=20,visualize=False):
		self.visualize = visualize
		self.sampleN = sampleN
		self.type = "Parallel BayesOpt"

	def ei(self,mu, sigma, current_max ,xi=0):
		I = mu - current_max
		z = (I - xi)/sigma
		ei = (I - xi) * norm.cdf(z) + sigma * norm.pdf(z)

		ei[ei!=ei] = 0
		ei[ei<0] = 0

		return ei

	#@jit
	def get_nextID(self,model=None,Worker=None):

		J = np.shape(Worker["now_p"])[0]
		d, N = np.shape(model.allX)

		##### check model #####
		if model.name != "Gaussian Process":
			print("BUCB can be calcurated only from Gaussian Process model.")
			return False

		##### model prediction #####
		model.fit()
		g = model.predict(return_var=True)
		mu = g["mean"]
		var = g["var"]

		##### calcurate asyEI from here #####
		print(" >> start calc LP-EI acquisition...")
		elapsed_time = time.time()

		current_max = np.max(model.allY[np.sort(model.trainID)])
		ei = self.ei(mu=mu,sigma=var,current_max=current_max)

		M = np.max(mu)

		for dim in range(d):
			dth_X = model.allX[:,dim]

			distant = dth_X[]




		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq = bucb

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

			fig, ax = plt.subplots(2, 1, figsize=(4, 7))

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
			ax[0].plot(x[Worker["now_p"]], y[Worker["now_p"]], "o",marker=">", color="black",label="processing")
			ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

			##### plot acquisition function #####
			ax[1].set_xlim(min(x),max(x))
			ax[1].set_xlabel(r"$x$",fontsize=15)
			ax[1].grid(True)
			ax[1].plot(x,acq,color="#228b22",label="BUCB")
			ylim = ax[1].set_ylim()
			ax[1].plot([x[nextID]],acq[[nextID]],"o",marker="*",markersize=10,color="red")
			ax[1].fill_between(x,ylim[0]*np.ones(N),acq,color="#228b22",alpha=0.5)
			ax[1].legend()

			if not(os.path.exists("./fig_BUCB")):
				os.mkdir("./fig_BUCB")

			t = np.shape(model.trainID)[0]
			plt.savefig("./fig_BUCB/step"+"%04d"%t+".pdf",bbox_inches="tight")
			plt.close()
			#"""

		return nextID, max(acq)
