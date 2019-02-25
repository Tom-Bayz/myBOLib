import numpy as np

##### original package #####
import yonezu as yz

##### scipy #####
import scipy
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
		self.count = 0

	def pmes_at_i(self,args):

		model = args[0]
		batch_point = args[1]
		Fstar_samples = args[2]
		i = args[3]

		J = np.shape(batch_point)[0]
		joint = np.sort(np.r_[batch_point,i])

		sub_pred = model.predict(at=joint,return_cov=True)
		sub_cov = sub_pred["cov"]
		sub_var = np.diag(sub_cov)
		sub_mu = sub_pred["mean"]

		##### compute 1st term #####
		joint_entropy = 0.5*np.log(((2*np.pi*np.exp(1))**(J+1))*np.linalg.det(sub_cov))

		##### compute 2nd term from here #####
		cnd_entropy = 0
		M = 0

		for fstar in Fstar_samples:

			grid_num = 100

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
					H = np.array(multivariate_normal.pdf(grid,mean=sub_mu, cov=sub_cov,allow_singular=True)) # compute pdf

					H = H[H >= 1e-9]

					hyper_volume = np.prod(upper-lower)
					ce = ((1/Z)*(hyper_volume/(grid_num**(J+1)))*np.sum(-H*np.log(H)) + np.log(Z)) # integrate
					M += 1

				else:
					##### compute conditional Entropy by Monte calro simulation　#####
					S = 10000
					#print("sampling...")
					samples = np.random.multivariate_normal(sub_mu,sub_cov,S)

					##### check if samples is in the truncated area #####
					for j in range(J+1):
						in_area = (lower[j] <= samples[:,j]) & (samples[:,j] <= upper[j])
						samples= samples[in_area,:]

					if np.shape(samples)[0] != 0:
						H = multivariate_normal.pdf(samples,mean=sub_mu, cov=sub_cov,allow_singular=True)
						H = H[H >= 1e-9]
						ce = (1/Z)*((1/S)*np.sum(-np.log(H))) + np.log(Z) # integrate
						M += 1
					else:
						ce = 0

				cnd_entropy += ce
		else:
			pass

		if M == 0:
			cnd_entropy = -np.inf
		else:
			cnd_entropy = cnd_entropy/M

		first_term = joint_entropy
		second_term = -cnd_entropy
		pmes = joint_entropy - cnd_entropy

		return [pmes, first_term, second_term]

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

		##### GPR prediction #####
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
		N = np.shape(model.allX)[0]
		pmes = np.ones(N)
		pmes[np.sort(model.trainID)] = -np.inf
		pmes[np.sort(batch_point)] = -np.inf

		first_term = np.copy(pmes)
		second_term = np.copy(pmes)

		print(" >> start calc PMES acquisition...")
		p = mproc.Pool(min(8,mproc.cpu_count()))
		index = np.array(np.where(pmes == 1)[0])
		result = p.map(self.pmes_at_i,[ [model,batch_point,Fstar_samples,i] for i in index ] )
		result = np.array(result)
		p.close()

		pmes[index] = result[:,0]
		first_term[index] = result[:,1]
		second_term[index] = result[:,2]

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
			ax[0].fill_between(x, Ucb, Lcb, color="blue", alpha=0.4,label="Uncertainty: "+r"$\sigma^2(x)$") # post var
			ax[0].plot(x, y, "--", color="red",label="Unknown function:" + r"$f(x)$") # true function
			ax[0].plot(x, mu, "-", color="blue",label="Predictiion: "+r"$\mu(x)$") # post var

			##### plot training point and calcurating point #####
			ax[0].plot(x[model.trainID], y[model.trainID], "s", color="black",label="observed data")
			ax[0].plot(x[batch_point], y[batch_point], "o",marker=">", color="black",label="batch points")
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
