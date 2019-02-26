# -*- coding: utf-8 -*-
u"""Expected Improvement."""
from __future__ import division
import numpy as np
from scipy.stats import norm
import time
from matplotlib import pyplot as plt
import os
import sys

class EI(object):

	def __init__(self,visualize=True,xi=0):
		self.visualize = visualize
		self.xi = xi
		self.type = "Parallel BayesOpt"
	
	def get_nextID(self,model=None,batch_point=None):

		if model.name == "Gaussian process":
			return self.get_singleID(model=model, batch_point=batch_point)

		elif model.name == "Multi-task Gaussian process":
			return self.get_multiID(model=model, batch_point=batch_point)
		
		else:
			print("EI: Error invalid regression model!!")
			sys.exit()
	
	def get_multiID(self,model,batch_point):
		print("this process is not prepared...")
		sys.exit()
	
	def get_singleID(self,model,batch_point):

		N = np.shape(model.allX)[0]

		elapsed_time = time.time()

		print(" >> gaussian process regression...")
		model.fit()
		g = model.predict(return_var=True)
		print("    complete")
		mu = g["mean"]
		var = g["var"]

		current_max = np.max(model.allY[model.trainID])

		print(" >> start calc EI acquisition...")
		I = mu - current_max
		z = (I - self.xi)/np.sqrt(var)
		ei = (I - self.xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

		ei[ei!=ei] = 0
		ei[ei<0] = 0

		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq = ei

		acq[model.trainID] = -np.inf
		acq[batch_point] = -np.inf

		nextID = np.argmax(acq)

		return nextID, np.max(ei)

	
		Lcb = mu - np.sqrt(var)
		Ucb = mu + np.sqrt(var)

		x = model.allX[:,0]
		y = model.allY

		fig, ax = plt.subplots(2, 1, figsize=(5, 5))

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
		ax[1].plot(x,acq,color="#228b22",label="EI")
		ylim = ax[1].set_ylim()
		ax[1].plot([x[nextID]],[acq[nextID]],"o",marker="*",markersize=10,color="red")
		ax[1].fill_between(x,ylim[0]*np.ones(N),acq,color="#228b22",alpha=0.5)
		ax[1].legend()

		if not(os.path.exists("./fig_EI")):
			os.mkdir("./fig_EI")

		t = np.shape(model.trainID)[0]
		plt.savefig("./fig_EI/step"+"%04d"%t+".pdf",bbox_inches="tight")
		plt.close()
