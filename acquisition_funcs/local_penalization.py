import numpy as np
import os
import time

##### scipy #####
from scipy.stats import truncnorm
from scipy.stats import norm
from scipy.special import erfc
from scipy.spatial import distance

##### others #####
from tqdm import tqdm
from matplotlib import pyplot as plt 
from numba import jit
from numpy.linalg import norm

class LP(object):

	def __init__(self,Lipschitz_const=5,type="EI"):
		self.L = Lipschitz_const
		self.type = "Parallel BayesOpt"

	def ei(self,mean, var, current_max ,xi=0):
		I = mean - current_max
		z = (I - xi)/np.sqrt(var)
		ei = (I - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

		ei[ei!=ei] = 0
		ei[ei<0] = 0

		return ei
	
	def ucb(self,mean, var, beta=4.0):

		return mean + np.sqrt(beta * var)
	
	def local_penalizer(self,X,mean,var,batch_point):

		N = np.shape(mean)[0]

		d = distance.cdist(np.atleast_2d(X[batch_point,:]), X, "euclidean")
		v = np.tile(var[batch_point][:,np.newaxis],(1,N))
		m = np.tile(mean[batch_point][:,np.newaxis],(1,N))
		M = np.max(mean)

		z = (1/np.sqrt(2*v)) * (self.L*d - M + m)

		lp = np.prod(erfc(-z),axis=0)

		return lp

	def get_nextID(self,model=None,batch_point=None):

		if model.name == "Gaussian process":
			return self.get_singleID(model=model, batch_point=batch_point)

		elif model.name == "Multi-task Gaussian process":
			return self.get_multiID(model=model, batch_point=batch_point)
		
		else:
			print("LP: Error invalid regression model!!")
			sys.exit()

	def get_multiID(self,model,batch_point):
		print("this process is not prepared...")
		sys.exit()
	
	def get_singleID(self,model,batch_point):

		J = np.shape(Worker["now_p"])[0]
		d, N = np.shape(model.allX)

		elapsed_time = time.time()

		if self.type == "EI":
			print(" >> gaussian process regression...")
			model.fit()
			g = model.predict(return_var=True)
			print("    complete")
			mu = g["mean"]
			var = g["var"]

			current_max = np.max(model.allY[np.sort(model.trainID)])

			print(" >> start calc LP-EI acquisition...")
			acq = self.ei(mean=mu,var=var,current_max=current_max)
		
		elif self.type == "UCB":
			print(" >> gaussian process regression...")
			model.fit()
			g = model.predict(return_var=True)
			print("    complete")
			mu = g["mean"]
			var = g["var"]

			print(" >> start calc LP-UCB acquisition...")
			t = np.shape(model.trainID)[0]
			delta = 0.1 # based on UCB paper
			beta = 2*np.log( ( ((np.pi**2)*(t**2)/6) * N ) / delta )

			acq = self.ucb(mean=mu,var=var,beta=beta)
		
		else:
			print("LP:invalid type!!")
			sys.exit()

		acq = acq * self.local_penalizer(model.allX,mean,var,batch_point)

		elapsed_time = time.time() - elapsed_time
		print("    complete (time "+str(int(elapsed_time/60))+":%02d"%(elapsed_time%60)+")")

		acq[model.trainID] = -np.inf
		acq[batch_point] = -np.inf

		##### maximize acquisition #####
		nextID = np.argmax(acq)

		if np.shape([nextID])[0] > 1:
			nextID = np.random.choice(nextID)

		return nextID, max(acq)

	def plot():
		print("not prepared yet!!")
		sys.exit()
