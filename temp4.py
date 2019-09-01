import env
import agent
import numpy as np
import matplotlib.pyplot as plt


Gambler_env = env.Gambler_env()
agent_1 = agent.Gambler_ValueIteration(Gambler_env)



while True:

	agent_1.reset()
	agent_1.update()

	if agent_1.get_delta() < agent_1.get_threshold():
		break
print(agent_1.get_policy())
print(agent_1.V)