# -*- coding: utf-8 -*-
u"""Expected Improvement."""
from __future__ import division
import numpy as np
from scipy.stats import norm


def TS(object):

	def get_nextID(self,model=None,batch_point=None):

		if model.name == "Gaussian process":
			return self.get_singleID(model=model, batch_point=batch_point)

		elif model.name == "Multi-task Gaussian process":
			return self.get_multiID(model=model, batch_point=batch_point)
		
		else:
			print("TS: Error invalid regression model!!")
			sys.exit()

	def get_multiID(self,model,batch_point):
		print("this process is not prepared...")
		sys.exit()
	
	def get_singleID(self,model,batch_point):

		##### model prediction #####
		print(" >> gaussian process regression...")
		model.fit()
		g = model.predict(return_cov=True)
		print("    complete")

		sample = np.random.multivariate_normal(g["mean"],g["cov"],1)

		acq = sample

		acq[model.trainID] = -np.inf
		acq[batch_point] = -np.inf

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
		ax[1].plot(x,acq,color="#228b22",label="TS")
		ylim = ax[1].set_ylim()
		ax[1].plot([x[nextID]],[acq[nextID]],"o",marker="*",markersize=10,color="red")
		ax[1].fill_between(x,ylim[0]*np.ones(N),acq,color="#228b22",alpha=0.5)
		ax[1].legend()

		if not(os.path.exists("./fig_TS")):
			os.mkdir("./fig_TS")

		t = np.shape(model.trainID)[0]
		plt.savefig("./fig_EI/step"+"%04d"%t+".pdf",bbox_inches="tight")
		plt.close()