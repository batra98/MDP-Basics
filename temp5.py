import env
import numpy as np


# def poisson_probability(n, lam):
#     global poisson_cache
#     key = n * 10 + lam
#     if key not in poisson_cache:
#         poisson_cache[key] = poisson.pmf(n, lam)
#     return poisson_cache[key]

limit = 7

def expected_return(state,action,V,env):


	value = 0.0
	value -= env.loss * abs(action)

	cars_at_1 = min(state[0]-action,env.max_cars)
	cars_at_2 = min(state[1]+action,env.max_cars)

	for rent_1 in range(limit):
		for rent_2 in range(limit):
			prob_rent = env.store_rent_1[rent_1]*env.store_rent_2[rent_2]

			present_number_at_1 = cars_at_1
			present_number_at_2 = cars_at_2

			request_at_1 = min(present_number_at_1,rent_1)
			request_at_2 = min(present_number_at_2,rent_2)

			reward = (request_at_1+request_at_2)*(env.profit)

			present_number_at_1 -= request_at_1
			present_number_at_2 -= request_at_2
			# print(present_number_at_1)
			for return_1 in range(limit):
				for return_2 in range(limit):
					prob_return = env.store_return_1[return_1]*env.store_return_2[return_2]
					present_number_at_1_1 = min(present_number_at_1+return_1,env.max_cars)
					present_number_at_2_1 = min(present_number_at_2+return_2,env.max_cars)

					# print(present_number_at_1)

					value += (prob_return*prob_rent)*(reward+0.9*V[present_number_at_1_1][present_number_at_2_1])

	return value


Jack_env = env.Jack_env()

V = np.zeros((Jack_env.max_cars+1,Jack_env.max_cars+1))
V_2 = np.zeros((Jack_env.max_cars+1,Jack_env.max_cars+1))
# policy = np.zeros((Jack_env.max_cars+1,Jack_env.max_cars+1))
policy = np.zeros(V.shape, dtype=np.int)

# poisson_cache = dict()
# store_rent_1 = []
# store_rent_2 = []
# store_return_1 = []
# store_return_2 = []

# for i in range(11):
# 	store_rent_1.append(poisson.pmf(i,Jack_env.mu_rent_first_location))
# 	store_rent_2.append(poisson.pmf(i,Jack_env.mu_rent_second_location))
# 	store_return_1.append(poisson.pmf(i,Jack_env.mu_return_first_location))
# 	store_return_2.append(poisson.pmf(i,Jack_env.mu_return_second_location))



while True:

	### policy-evaluation ###

	while True:
		delta = 0
		V_2 = V.copy()


		for i in range(Jack_env.max_cars+1):
			for j in range(Jack_env.max_cars+1):
				next_state_value = expected_return([i,j],policy[i][j],V,Jack_env)
				# print(next_state_value)
				# V[i][j] = next_state_value
				# print(next_state_value)
				V[i][j] = next_state_value
		delta = abs(V_2-V).max()
				# delta = max(delta,np.abs(next_state_value-V[i][j]))
				# V_2[i][j] = next_state_value
		print(delta)
		if delta < 0.001:
			break

	stable = True

	for i in range(Jack_env.max_cars+1):
		for j in range(Jack_env.max_cars+1):
			chosen_a = policy[i][j]
			A = []

			for action in Jack_env.actions:
				if (0 <= action <= i) or (-j <= action <= 0):
					A.append(expected_return([i,j],action,V,Jack_env))
				else:
					A.append(-np.inf)

			new_action = Jack_env.actions[np.argmax(A)]
			policy[i][j] = new_action

			if stable and chosen_a!=new_action:
				stable = False

	print(policy)
	if stable:
		break


