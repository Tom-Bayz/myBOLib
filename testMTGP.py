from matplotlib import pyplot as plt
import glob
import numpy as np
import myBOLib as bol
import pickle

mtgpr=bol.MTGPR.MTGPRegression(input_dim=1,task_dim=[1],task_des_Num=1,in_norm=False)
np.random.seed(7)

for i,func in enumerate(np.sort(glob.glob("./synthetic_Data/*"))):

	with open(func,"rb") as f:
		data = pickle.load(f,encoding='latin1')

	X = data["X"][:,np.newaxis]
	Y = data["Y"]
	z = data["z"]
	cost = data["cost"]

	init = (np.random.choice(range(100),np.random.choice([1,2])))
	mtgpr.add_objFunc(allX=X,allY=Y,task_descriptor=[np.atleast_2d(z)],cost=cost,trainID=init)

mtgpr.model_select()
mtgpr.fit()
mtgpr.predict(return_var=True)

plt.grid(True)
for gp in mtgpr.gps:

	x = gp.allX[:,0]
	y = gp.allY

	train = gp.trainID

	mean = gp.pred_dist["mean"]
	var = gp.pred_dist["var"]

	plt.plot(x,mean,label="mean")
	plt.fill_between(x,mean-np.sqrt(var),mean+np.sqrt(var),label="var",alpha=0.3)
	plt.plot(x[train],mean[train],"o",marker="s",markersize=8,color="black")

plt.xlim(min(x),max(x))
plt.show()
