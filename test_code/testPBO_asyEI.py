import os
import sys
import glob
import pickle
import argparse
import numpy as np

##### original package #####
import myBOLib as bol
kf = bol.kernel_funcs
af = bol.acquisition_funcs

def PBO_asyEI(X,Y,WorkerN=1,T=50,seed=0,init_dataN=1):

    _ ,d = np.shape(X)

    ##### GPR instance #####
    kernel = kf.RBF(50,"median")
    GP = bol.GPR.GPRegression(allX=X, allY=Y,kernel=kernel, input_dim=d)
    acquisition = af.asyEI(sampleN = 500,visualize = True)

    ##### Parallel BO instance #####
    PBO = bol.PBO.Parallel_BayesOpt(GPR=GP,acq=acquisition,J=WorkerN)
    PBO.set_initial(seed=seed,dataN=init_dataN)
    PBO.set_logINFO(dir_name="./",file_name="asyei_test.csv",interval=3)
    regret_result = PBO.optimize(T=T,model_selection=None)

    return regret_result

if __name__ == "__main__":

    data = np.loadtxt("./synthetic_Data/Forrester.csv",dtype="float",delimiter=",")
    X = data[:,:-1]
    Y = data[:,-1]

    PBO_asyEI(X,Y,init_dataN=2,T=2,WorkerN=2,seed=44)
