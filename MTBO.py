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

class MultiTask_BayesOpt(object):

	def __init__(self,MTGPR=None,acq=None):

		##### initialize GPR #####
		if MTGPR is None:
			print("please input model!!")
			return False
		else:
			self.MTGPR = MTGPR

		self.J = None
		self.interval = 1

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

		self.dir_name = "experiment"
		self.file_name = "log_file.csv"
		self.interval = 1

		if not(os.path.exists(self.dir_name)):
			os.mkdir(self.dir_name)

	def set_logINFO(self,dir_name="experiment",file_name="log_file.csv",interval=1):
		self.dir_name = dir_name
		self.file_name = file_name
		self.interval = interval

		if not(os.path.exists(self.dir_name)):
			os.mkdir(self.dir_name)

	def set_initial(self,seed=0,dataN=1):

		##### set random seed #####
		np.random.seed(seed)

		init = np.random.choice(np.array(range(self.N)),self.J+dataN)

		self.Worker["now_p"] = init[:self.J]
		self.Worker["wait_time"] = self.cost[np.array(self.Worker["now_p"])]

		self.MTGPR.trainID = init[self.J:]

	def optimize(self,T=None,model_selection=None):

		global_maximum=[]


		global_maximum = np.max(self.MTGPR.allY)

		if (T < 2) or (T is None):
			print("invalid T value")
			return False

		inference_regret = []
		simple_regret = []
		passed = 0

		for t in range(T):

			print("**************************************")
			print(" t="+str(t))


			if not(model_selection is None):
				if (t%model_selection == 0):
					print(" >> model selection...")
					cnd, mls = self.MTGPR.model_select()
					print("    complete (selected param="+str(round(self.MTGPR.kernel.hyp[1],3))+")")
					"""
					plt.xscale("log")
					plt.grid(True)
					plt.plot(cnd,mls,color="black",label="LogMarginalLikelihood")
					plt.plot(cnd[np.argmax(mls)],[np.max(mls)],"o",label="best "+str(round(cnd[np.argmax(mls)],3)))
					plt.legend()
					plt.pause(0.01)
					plt.close()
					"""

			acq = self.acq.get_nextID(self.model,self.Worker["now_p"])
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
				self.MTGPR.trainID = np.r_[np.atleast_1d(self.MTGPR.trainID).astype("int64"),self.Worker["now_p"][p]]
				self.Worker["now_p"][p] = nextID
				self.Worker["wait_time"][p] = self.cost[nextID]

			##### inference regret #####
			Inf_r = np.abs(global_maximum - self.MTGPR.allY[np.argmax(self.MTGPR.pred_dist["mean"])])
			inference_regret.append(Inf_r)

			##### simple regret #####
			Smp_r = np.abs(global_maximum - self.MTGPR.allY[self.MTGPR.trainID[-1]])
			simple_regret.append(Smp_r)

			print(" >> simple regret: ")
			print("    [now]"+str(np.round(Smp_r,3))+" [min]"+str(min(simple_regret)))
			print("")

			if t%self.interval == 0 or t==(T-1):
				np.savetxt(self.dir_name+"/"+self.file_name,np.r_[[simple_regret],[inference_regret]],delimiter=",")

		print("**************************************")

		return {"simple_regret":np.array(simple_regret),"inference_regret":np.array(inference_regret)}
