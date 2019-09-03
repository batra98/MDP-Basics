import env
import numpy as np
import agent
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
matplotlib.style.use('ggplot')
# %matplotlib inline
# matplotlib.rcParams['figure.figsize'] = (15, 9)

def three_dimentional_plot(V):
	fig = plt.figure()
	ax = fig.gca(projection = '3d')
	X = np.arange(0,V.shape[0],1)
	Y = np.arange(0,V.shape[1],1)
	X,Y = np.meshgrid(X,Y)
	surf = ax.plot_surface(X,Y,V,rstride = 1,cstride = 1,cmap = cm.coolwarm,linewidth = 0,antialiased = False)
	ax.set_xlabel('Location 2')
	ax.set_ylabel('Location 1')
	plt.show()

def plot_policy(all_P):
	# fig, axes = plt.subplots(1,3,figsize(18,9))
	fig = plt.subplots(1,4)
	itr = 0
	for pi in all_P:
		plt.matshow(pi)
		# ax.invert_yaxis()
		# ax.set_title('Policy ($\pi_{0}$)'.format(itr))
		plt.xlabel('Location 2')
		plt.ylabel('Location 1')

	plt.show()

Jack_env = env.Jack_env()
agent_1 = agent.Jack_PolicyIteration(Jack_env)


all_V = []
all_P = []

all_P.append(agent_1.policy.copy())

while True:
	agent_1.evaluate_policy()
	stable = agent_1.update()

	all_V.append(agent_1.V.copy())
	all_P.append(np.flip(agent_1.policy.copy(),0))
	if stable == True:
		break
# print(all_P)
plot_policy(all_P)
three_dimentional_plot(agent_1.V)
