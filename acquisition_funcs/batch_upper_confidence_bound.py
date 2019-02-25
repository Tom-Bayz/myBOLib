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


class BUCB(object):

	def __init__(self,visualize=False):
		self.visualize = visualize
		self.type = "Parallel BayesOpt"

	def ucb(self,mu, sigma, beta=4.0):
		u"""Upper Confidence Bound."""
		return mu + np.sqrt(beta * sigma)

	#@jit
	def get_nextID(self,model=None,batch_point=None):

		J = np.shape(batch_point)[0]
		N = np.shape(model.allX)[0]

		##### check model #####
		if model.name != "Gaussian Process":
			print("BUCB can be calcurated only from Gaussian Process model.")
			return False

		##### model prediction #####
		print(" >> gaussian process regression...")
		model.fit()
		g = model.predict(return_var=True)
		print("    complete")
		mu = g["mean"]
		var = g["var"]

		##### calcurate asyEI from here #####
		print(" >> start calc BUCB acquisition...")
		elapsed_time = time.time()

		joint = np.sort(batch_point)

		model.halcY = np.copy(model.allY)
		model.halc_trainID = np.sort(np.r_[model.trainID,batch_point])
		model.fit(halc=True)
		halc_pred = model.predict(return_var=True,halc=True)
		halc_var = halc_pred["var"]

		##### ucb parameter #####
		t = np.shape(model.halc_trainID)[0]
		delta = 0.1 # based on UCB paper
		beta = 2*np.log( ( ((np.pi**2)*(t**2)/6) * N ) / delta )

		##### calc ucb for hallucinated variance and standard mean #####
		bucb = self.ucb(mu,halc_var,beta=beta)

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

			fig, ax = plt.subplots(2, 1, figsize=(5, 4.6))

			ax[0].grid(True)
			ax[0].set_xlim(min(x),max(x))
			ax[0].plot(x, y, "--", color="red") # true function
			ylim = np.array(ax[0].set_ylim())*1.1
			ax[0].set_ylim(ylim)
			ax[0].fill_between(x, Ucb, Lcb, color="blue", alpha=0.4,label="Uncertainty: "+r"$\sigma^2(x)$") # post var
			ax[0].plot(x, y, "--", color="red",label="Unknown function:" + r"$f(x)$") # true function
			ax[0].plot(x, mu, "-", color="blue",label="prediction: "+r"$\mu(x)$") # post var

			##### plot training point and calcurating point #####
			ax[0].plot(x[model.trainID], y[model.trainID], "s", color="black",label="observed Data")
			ax[0].plot(x[batch_point], y[batch_point], "o",marker=">", color="black",label="batch points")
			ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

			##### plot acquisition function #####
			ax[1].set_xlim(min(x),max(x))
			ax[1].grid(True)
			ax[1].set_xlabel(r"$x$",fontsize=15)
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
