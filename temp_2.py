import env
import agent
import numpy as np
import matplotlib.pyplot as plt

Env = env.Grid_1()
agent_1 = agent.ValueIteration(Env)
agent_2 = agent.PolicyIteration(Env)
agent_3 = agent.ConfusedAgent(Env)


def plot(Env,policy,V):
	pp = np.reshape(np.argmax(policy, axis=1), Env.shape)
	# print(pp)

	plt.matshow(np.reshape(V, Env.shape))
	# plt.arrow(0,0.5,0,-0.7,head_width = 0.1)



	for i in range(Env.shape[0]):
		for j in range(Env.shape[1]):
			if i == (Env.shape[0]-1) and j == (Env.shape[1]-1):
				continue
			if pp[i][j] == 0:
				plt.arrow(j,i+0.5,0,-0.7,head_width = 0.1)
			elif pp[i][j] == 2:
				plt.arrow(j,i-0.5,0,+0.7,head_width = 0.1)
			elif pp[i][j] == 1:
				plt.arrow(j-0.5,i,0.7,0,head_width = 0.1)
			elif pp[i][j] == 3:
				plt.arrow(j+0.5,i,-0.7,0,head_width = 0.1)

	plt.show()



# agent_1.set_gamma(0.5)
# agent_2.set_gamma(0.5)
itr = 0
while True:
	agent_1.reset()
	agent_1.update()

	if agent_1.get_delta() < agent_1.get_threshold():
		break
	itr += 1
# print(np.reshape(agent_1.V,Env.shape))
print(itr)
policy = agent_1.get_policy()
print(np.reshape(agent_1.V,Env.shape))
plot(Env,policy,agent_1.V)

while True:
	V = agent_2.evaluate_policy()
	# print(V)

	stable = agent_2.update()

	if stable == True:
		break

print(np.reshape(agent_2.V,Env.shape))
plot(Env,agent_2.policy,agent_2.V)


agent_3.get_policy()
print(np.reshape(agent_3.V,Env.shape))
plot(Env,agent_3.policy,agent_3.V)

	



