from abc import ABC, abstractmethod
import numpy as np

class DiscreteAgent(ABC):

	@abstractmethod
	def __init__(self,env):
		pass

	@abstractmethod
	def update(self):
		pass

	@abstractmethod
	def get_action(self,state):
		pass

class PolicyIteration(DiscreteAgent):

	def evaluate_policy(self):
		pass

	def update_policy(self):
		pass


class ValueIteration(DiscreteAgent):
	
	def __init__(self,env):
		self.nstates = env.nstates
		self.nactions = env.nactions
		self.size = env.size
		self.V = np.zeros(env.nstates)
		self.delta = 0
		self.gamma = 1
		self.P = env.P

	def look(state,V):

		A = np.zeros(self.nactions)

		for a in range(self.nactions):
			for prob,next_state,reward,terminate in self.P[state][a]:
				A[a] += prob*(reward+self.gamma*V[next_state])
		return A

	def update(self,env):
		state_values = new_state_values
		old_state_values = state_values.copy()

		for s in range(self.nstates):
			A = look(s,V)


