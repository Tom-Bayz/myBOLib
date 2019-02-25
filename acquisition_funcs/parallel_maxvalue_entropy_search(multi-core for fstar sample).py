import numpy as np

##### original package #####
import yonezu as yz

##### scipy #####
from scipy.stats import truncnorm
from scipy.stats import mvn
from scipy.stats import multivariate_normal

##### others #####
from matplotlib import pyplot as plt
import os
import time
from numba import jit
from tqdm import tqdm
import math

##### multi-core processing #####
import multiprocessing as mproc

class PMES(object):

	def __init__(self,sampleN=50,max_dist="gaussian",visualize=False):
		self.visualize = visualize
		self.sampleN = sampleN
		self.max_dist = max_dist
		self.name = "PMES"
		self.type = "Parallel BayesOpt"

	def subproc(self,args):

		sub_mu = args[0]
		sub_cov = args[1]
		sub_var = np.diag(sub_cov)
		fstar = args[2]

		grid_num = 100
		J = np.shape(sub_var)[0]-1

		##### compute 2nd term from here #####
		upper = np.min(np.array([sub_mu + 5*np.sqrt(sub_var), fstar*np.ones(J+1)]),axis=0)
		lower = np.min(np.array([sub_mu - 5*np.sqrt(sub_var), fstar*np.ones(J+1)]),axis=0)

		Z = mvn.mvnun(lower,upper,sub_mu,sub_cov)[0]

		if Z > 0:
			if J < 3:
				##### compute conditional Entropy by Division quadrature　#####
				grid = []
				for j in range(J+1):
					jth_space = np.unique(np.linspace(lower[j],upper[j],grid_num))[:,np.newaxis]
					grid.append(jth_space)

				grid = yz.gen_mesh(grid)
				H = np.array(multivariate_normal.pdf(grid,mean=sub_mu, cov=sub_cov)) # compute pdf

				H = H[H >= 1e-9]

				hyper_volume = np.prod(upper-lower)
				ce = ((1/Z)*(hyper_volume/(grid_num**(J+1)))*np.sum(-H*np.log(H)) + np.log(Z)) # integrate

			else:
				##### compute conditional Entropy by Monte calro simulation　#####
				sampleN = 0
				samples = np.atleast_2d(sub_mu)

				while 1:
					s = np.random.multivariate_normal(sub_mu,sub_cov,1000)
					sampleN += 1000
					##### check if samples is in the truncated area #####
					for j in range(J+1):
						in_area = (lower[j] <= s[:,j]) & (s[:,j] <= upper[j])
						s = s[in_area,:]
					samples = np.r_[samples,s]

					if np.shape(samples)[0] < 50000:
						break

				H = multivariate_normal.pdf(samples,mean=sub_mu, cov=sub_cov)
				H = H[H >= 1e-9]

				ce = (1/Z)*((1/sampleN)*np.sum(-np.log(H))) + np.log(Z) # integrate
		else:
			ce
			pass

		return ce

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

			if np.shape(Fstar_samples)[0] > 10:
				break

		##### calcurate PMES from here #####
		N = np.shape(model.allX)[0]
		pmes = np.ones(N)
		pmes[np.sort(model.trainID)] = -np.inf
		pmes[np.sort(batch_point)] = -np.inf

		first_term = np.copy(pmes)
		second_term = np.copy(pmes)

		print(" >> start calc PMES acquisition...")
		for i in tqdm(np.array(np.where(pmes == 1)[0])):


			joint = np.sort(np.r_[batch_point,i])

			sub_pred = model.predict(at=joint,return_cov=True)

			sub_cov = sub_pred["cov"]
			sub_var = np.diag(sub_cov)
			sub_mu = sub_pred["mean"]

			##### compute 1st term #####
			joint_entropy = 0.5*np.log(((2*np.pi*np.exp(1))**(J+1))*np.linalg.det(sub_cov))

			##### compute 2nd term from here #####
			p = mproc.Pool(int(mproc.cpu_count()*0.8))
			cnd_entropy = p.map(self.subproc,[ [sub_mu,sub_cov,fstar] for fstar in Fstar_samples ])
			cnd_entropy = np.array(cnd_entropy)
			cnd_entropy = cnd_entropy[cnd_entropy == cnd_entropy]
			p.close()

			M = np.shape(cnd_entropy)[0]
			cnd_entropy = np.sum(cnd_entropy)

			if M == 0:
				cnd_entropy = -np.inf
			else:
				cnd_entropy = cnd_entropy/M

			first_term[i] = joint_entropy
			second_term[i] = -cnd_entropy
			pmes[i] = joint_entropy - cnd_entropy


		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq = pmes

		###### maxmize acquisition #####
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
			ax[0].plot(x, y, "--", color="red") # true function
			ax[0].set_xlabel(r"$x$",fontsize=15)
			ylim = np.array(ax[0].set_ylim())*1.1
			ax[0].set_ylim(ylim)
			ax[0].set_xlim(min(x),max(x))
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
			ax[1].plot(x,acq,color="#228b22",label="PMES")
			ylim = ax[1].set_ylim()
			ax[1].plot([x[nextID]],[acq[nextID]],"o",marker="*",markersize=10,color="red")
			ax[1].fill_between(x,ylim[0]*np.ones(N),acq,color="#228b22",alpha=0.5)
			ax[1].legend()

			ax[2].set_xlim(min(x),max(x))
			ax[2].grid(True)
			ax[2].set_xlabel(r"$x$",fontsize=15)
			ax[2].plot(x,first_term,color="red",label=r"$H[p(y)]$")
			ax[2].plot(x,-second_term,color="blue",label=r"$H[p(y|y^*)]$")
			ylim = ax[2].set_ylim()
			ax[2].fill_between(x,ylim[0]*np.ones(N),first_term,color="red",alpha=0.5)
			ax[2].fill_between(x,ylim[0]*np.ones(N),-second_term,color="blue",alpha=0.5)
			ax[2].legend()

			if not(os.path.exists("./fig_PMES")):
				os.mkdir("./fig_PMES")

			t = np.shape(model.trainID)[0]
			plt.savefig("./fig_PMES/step"+"%04d"%t+".pdf",bbox_inches="tight")
			plt.close()
			#"""

		return nextID, max(acq)
