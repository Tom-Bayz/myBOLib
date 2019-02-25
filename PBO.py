import os
import sys
import glob
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt

##### original package #####
import yonezu as yz

##### scipy #####
from scipy.stats import truncnorm
from scipy.stats import mvn
from scipy.stats import multivariate_normal

class Parallel_BayesOpt(object):

	def __init__(self,GPR=None,acq=None,J=2,cost=None):

		##### initialize GPR #####
		if GPR is None:
			print("please input GPR!!")
			return False
		else:
			self.GPR = GPR

		self.N = np.shape(self.GPR.allX)[0]
		self.J = J
		self.interval = 1

		##### initialize Workers #####
		if self.N < self.J:
			print("too many workers!!")
			return False
		else:
			self.Worker = {"now_p":-np.ones(J),\
					  "wait_time":-np.ones(J),\
					  "cum_cost":np.ones(J)}

		##### set acquisition function #####
		if acq is None:
			print("please input acuisition function!!")
			return False
		elif acq.type != "Parallel BayesOpt":
			print("This acquisition function is not for Parallel BO!!")
			return False
		else:
			self.acq = acq

		if cost is None:
			self.cost = np.ones(self.N)
		else:
			self.cost = cost

	def set_logINFO(self,dir_name="experiment",file_name="log_file.csv",interval=1):
		self.dir_name = dir_name
		self.file_name = file_name
		self.interval = interval

		if not(os.path.exists(self.dir_name)):
			os.mkdir(self.dir_name)

	def set_initial(self,seed=0,dataN=3):

		##### set random seed #####
		np.random.seed(seed)

		init = np.random.choice(np.array(range(self.N)),self.J+dataN)

		self.Worker["now_p"] = init[:self.J]
		self.Worker["wait_time"] = self.cost[np.array(self.Worker["now_p"])]

		self.GPR.trainID = init[self.J:]

	def optimize(self,T=None,model_selection=None):

		global_maximum = np.max(self.GPR.allY)

		if (T < 2) or (T is None):
			print("invalid T value")
			return False

		inference_regret = []
		simple_regret = []
		passed = 0

		##### inference regret #####
		#self.GPR.model_select()
		#self.GPR.fit()
		#self.GPR.predict()
		#Inf_r = np.abs(global_maximum - self.GPR.allY[np.argmax(self.GPR.pred_dist["mean"])])
		#inference_regret.append(Inf_r)

		##### simple regret #####
		#Smp_r = np.abs(global_maximum - np.max(self.GPR.allY[np.sort(self.GPR.trainID)]))
		#simple_regret.append(Smp_r)

		for t in range(T):

			print("**************************************")
			print(" t="+str(t))

			if not(model_selection is None):
				if (t%model_selection == 0):
					print(" >> model selection...")
					cnd, mls = self.GPR.model_select()
					print("    candidate: "+str(np.array(cnd)))
					print("    complete (selected param="+str(round(self.GPR.kernel.hyp[1],3))+")")
					"""
					plt.xscale("log")
					plt.grid(True)
					plt.plot(cnd,mls,color="black",label="LogMarginalLikelihood")
					plt.plot(cnd[np.argmax(mls)],[np.max(mls)],"o",label="best "+str(round(cnd[np.argmax(mls)],3)))
					plt.legend()
					plt.pause(0.01)
					plt.close()
					"""

			acq = self.acq.get_nextID(self.GPR,self.Worker["now_p"])
			if not(acq):
				print(" !! acquisition function return invalid ID !!")
				return False

			nextID = acq[0]
			##### change Worker information #####
			passed = min(self.Worker["wait_time"])
			self.Worker["wait_time"] -= passed # passed the time
			del_w = np.atleast_1d(range(self.J))[self.Worker["wait_time"] == 0]
			del_w = np.atleast_1d(del_w[0])

			for i,p in enumerate(del_w):
				self.GPR.trainID = np.r_[np.atleast_1d(self.GPR.trainID).astype("int64"),self.Worker["now_p"][p]]
				self.Worker["now_p"][p] = nextID
				self.Worker["wait_time"][p] = self.cost[nextID]

			##### inference regret #####
			self.GPR.fit()
			self.GPR.predict()
			Inf_r = np.abs(global_maximum - self.GPR.allY[np.argmax(self.GPR.pred_dist["mean"])])
			inference_regret.append(Inf_r)

			##### simple regret #####
			Smp_r = np.abs(global_maximum - np.max(self.GPR.allY[np.sort(self.GPR.trainID)]))
			simple_regret.append(Smp_r)

			print(" >> simple regret: ")
			print("    [now]"+str(np.round(Smp_r,3))+" [min]"+str(min(simple_regret)))
			print("")

			if t%self.interval == 0 or t==(T-1):
				np.savetxt(self.dir_name+"/"+self.file_name,np.r_[[simple_regret],[inference_regret]],delimiter=",")

		print("**************************************")

		return {"simple_regret":np.array(simple_regret),"inference_regret":np.array(inference_regret)}
