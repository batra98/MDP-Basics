import env
import numpy
import agent

Jack_env = env.Jack_env()
agent_1 = agent.Jack_PolicyIteration(Jack_env)

while True:
	agent_1.evaluate_policy()
	stable = agent_1.update()

	if stable == True:
		break