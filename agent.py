from abc import ABC, abstractmethod
import numpy as np

class DiscreteAgent(ABC):

	def __init__(self,env):
		self.nstates = env.nstates
		self.nactions = env.nactions
		self.V = np.zeros(env.nstates)
		self.V2 = np.zeros(env.nstates)
		self.policy = np.zeros([env.nstates,env.nactions])
		self.delta = 0
		self.gamma = 1
		self.P = env.P
		self.threshold = 0.00001

	@abstractmethod
	def update(self):
		pass

	def get_action(self,state):
		A = np.zeros(self.nactions)

		for a in range(self.nactions):
			for prob,next_state,reward,terminate in self.P[state][a]:
				A[a] += prob*(reward+self.gamma*self.V[next_state])
		return A
		

	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def get_policy(self):
		pass

	def set_gamma(self,gamma):
		self.gamma = gamma

	def set_threshold(self,threshold):
		self.threshold = threshold

	def get_threshold(self):
		return self.threshold

	def get_delta(self):
		return self.delta

	def clear(self):
		self.V = np.zeros(self.nstates)
		self.V2 = np.zeros(self.nstates)
		self.policy = np.zeros([self.nstates,self.nactions])
		

class PolicyIteration(DiscreteAgent):

	def __init__(self,env):
		super().__init__(env)

		self.policy = np.ones([self.nstates, self.nactions]) / self.nactions



	def evaluate_policy(self):
		self.V2 = np.zeros(self.nstates)
		self.V = np.zeros(self.nstates)
		while True:
			self.delta = 0
			self.V = np.copy(self.V2)

			for s in range(self.nstates):
			    v = 0

			    for a, action_prob in enumerate(self.policy[s]):
			        for  prob, next_state, reward, done in self.P[s][a]:
			            v += action_prob * prob * (reward + self.gamma * self.V[next_state])
			    self.delta = max(self.delta, np.abs(v - self.V[s]))
			    self.V2[s] = v

			if self.delta < self.threshold:
			    break

		return np.array(self.V)

	def update(self):
	    policy_stable = True
	    
	    for s in range(self.nstates):
	        
	        chosen_a = np.argmax(self.policy[s])
	        
	        action_values = self.get_action(s)
	        best_a = np.argmax(action_values)
	        
	        if chosen_a != best_a:
	            policy_stable = False
	        self.policy[s] = np.eye(self.nactions)[best_a]

	    return policy_stable


	
	def get_policy(self):
		return self.policy

	
	def reset(self):
		pass

	def clear(self):
		super().clear()
		self.policy = np.ones([self.nstates, self.nactions]) / self.nactions



class ValueIteration(DiscreteAgent):

	# def get_action(self,state):

	# 	A = np.zeros(self.nactions)

	# 	for a in range(self.nactions):
	# 		for prob,next_state,reward,terminate in self.P[state][a]:
	# 			A[a] += prob*(reward+self.gamma*self.V[next_state])
	# 	return A

	def update(self):
		self.V = np.copy(self.V2)
		for s in range(self.nstates):
			A = self.get_action(s)
			best_action_value = np.max(A)

			self.delta = max(self.delta,np.abs(best_action_value-self.V[s]))

			self.V2[s] = best_action_value

	def reset(self):
		self.delta = 0

	def get_policy(self):
		self.policy = np.zeros([self.nstates,self.nactions])
		for s in range(self.nstates):
			A = self.get_action(s)
			# print(A)
			best_action = np.argmax(A)
			self.policy[s,best_action] = 1.0

		return self.policy


class ConfusedAgent(DiscreteAgent):

	def get_policy(self):
		self.policy = np.zeros([self.nstates,self.nactions])
		for s in range(self.nstates):
			A = self.get_action(s)
			temp = np.random.randint(low = 0,high = 4)
			self.policy[s,temp] = 1.0
			self.V[s] = A[temp]

		return self.policy

	def reset(self):
		pass

	def update(self):
		pass


class Gambler_ValueIteration(DiscreteAgent):

	def __init__(self,env):
		super().__init__(env)

		self.p_h = env.p_h
		self.rewards = env.rewards

	def get_action(self,state):
		A = np.zeros(self.nstates)
		possible_bet = range(1,min(state,100-state)+1)
		# print(self.p_h)
		for a in possible_bet:
			A[a] = self.p_h * (self.rewards[state+a]+self.V[state+a]*self.gamma) + (1-self.p_h)*(self.rewards[state-a]+self.V[state-a]*self.gamma)
		# print(self.rewards)
		return A

	def update(self):
		y = []
		self.V = np.copy(self.V2)

		for s in range(1,self.nstates-1):
			A = self.get_action(s)
			best_action_value = np.max(A)
			self.delta = max(self.delta,np.abs(best_action_value-self.V[s]))
			self.V2[s] = best_action_value
		y.append(self.V)

		return y

	def reset(self):
		self.delta = 0

	def get_policy(self):
		self.policy = np.zeros(self.nstates-1)
		for s in range(self.nstates-1):
			A = self.get_action(s)
			best_action = np.argmax(A)
			self.policy[s] = best_action

		return self.policy

	def clear(self):
		super().clear()
		self.policy = np.zeros(self.nstates-1)


class Jack_PolicyIteration():

	def __init__(self,env):
		self.limit = 8
		self.max_cars = env.max_cars
		self.V = np.zeros((self.max_cars+1,self.max_cars+1))
		self.V_2 = np.zeros((self.max_cars+1,self.max_cars+1))
		self.policy = np.zeros(self.V.shape,dtype = np.int)
		self.delta = 0
		self.profit = env.profit
		self.loss = env.loss
		self.gamma = 0.9
		self.store_return_1 = env.store_return_1
		self.store_return_2 = env.store_return_2
		self.store_rent_1 = env.store_rent_1
		self.store_rent_2 = env.store_rent_2
		self.threshold = 0.001
		self.actions = env.actions


	def get_next_value(self,state,action):
		value = 0.0
		value -= self.loss*abs(action)

		cars_at_1 = min(state[0]-action,self.max_cars)
		cars_at_2 = min(state[1]+action,self.max_cars)

		for rent_1 in range(self.limit):
			for rent_2 in range(self.limit):
				prob_rent = self.store_rent_1[rent_1]*self.store_rent_2[rent_2]

				present_number_at_1 = cars_at_1
				present_number_at_2 = cars_at_2

				request_at_1 = min(present_number_at_1,rent_1)
				request_at_2 = min(present_number_at_2,rent_2)

				reward = (request_at_1+request_at_2)*(self.profit)

				present_number_at_1 -= request_at_1
				present_number_at_2 -= request_at_2
				# print(present_number_at_1)
				for return_1 in range(self.limit):
					for return_2 in range(self.limit):
						prob_return = self.store_return_1[return_1]*self.store_return_2[return_2]
						present_number_at_1_1 = min(present_number_at_1+return_1,self.max_cars)
						present_number_at_2_1 = min(present_number_at_2+return_2,self.max_cars)

						# print(present_number_at_1)

						value += (prob_return*prob_rent)*(reward+(self.gamma)*self.V[present_number_at_1_1][present_number_at_2_1])

		return value


	def evaluate_policy(self):
		

		while True:
			self.delta = 0
			self.V_2 = self.V.copy()
			for i in range(self.max_cars+1):
				for j in range(self.max_cars+1):
					next_state_value = self.get_next_value([i,j],self.policy[i][j])
					self.V[i][j] = next_state_value

			self.delta = abs(self.V_2 - self.V).max()
			print(self.delta)
			if self.delta < self.threshold:
				break

		print(self.V)

	def update(self):
		policy_stable = True

		for i in range(self.max_cars+1):
			for j in range(self.max_cars+1):

				chosen_a = self.policy[i][j]
				A = []

				for action in self.actions:
					if (0 <= action <= i) or (-j <= action <= 0):
						A.append(self.get_next_value([i,j],action))
					else:
						A.append(-np.inf)

				new_action = self.actions[np.argmax(A)]
				self.policy[i][j] = new_action

				if policy_stable and chosen_a!=new_action:
					policy_stable = False

		print(self.policy)
	
		return policy_stable	
		

	



