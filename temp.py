import env
# import agent
import numpy as np
import matplotlib.pyplot as plt

Env = env.Grid_1()
# agent = agent.DiscreteAgent(Grid_World)

def look(state,V):

	A = np.zeros(Env.nactions)

	for a in range(Env.nactions):
		for prob,next_state,reward,terminate in Env.P[state][a]:
			A[a] += prob*(reward+0.1*V[next_state])
	return A

V = np.zeros(Env.nstates)
V2 = np.zeros(Env.nstates)
while True:
	delta = 0

	policy = np.zeros([Env.nstates,Env.nactions])


	V = np.copy(V2)
	for s in range(Env.nstates):
		A = look(s,V)
		print(A)
		best_action_value = np.max(A)

		delta = max(delta,np.abs(best_action_value-V[s]))

		V2[s] = best_action_value

		best_action = np.argmax(A)
		policy[s,best_action] = 1.0

	# print(delta)
	print(np.reshape(np.argmax(policy, axis=1), Env.shape))
	if delta < 0.0001:
		break

# policy = np.zeros([Env.nstates,Env.nactions])

# for s in range(Env.nstates):
# 	A = look(s,V)
# 	best_action = np.argmax(A)
# 	policy[s,best_action] = 1.0

# def policy_eval(policy, env, discount_factor, theta=0.0001):
# 	V = np.zeros(Env.nstates)
# 	while True:
# 		delta = 0
# 		# For each state, perform a "full backup"
# 		for s in range(Env.nstates):
# 		    v = 0
# 		    # Look at the possible next actions
# 		    for a, action_prob in enumerate(policy[s]):
# 		        # For each action, look at the possible next states...
# 		        for  prob, next_state, reward, done in Env.P[s][a]:
# 		            # Calculate the expected value
# 		            v += action_prob * prob * (reward + discount_factor * V[next_state])
# 		    # How much our value function changed (across any states)
# 		    delta = max(delta, np.abs(v - V[s]))
# 		    V[s] = v
# 		# Stop evaluating once our value function change is below a threshold
# 		print(delta)
# 		if delta < theta:
# 		    break
# 	return np.array(V)

# # Start with a random policy
# policy = np.ones([Env.nstates, Env.nactions]) / Env.nactions

# while True:
#     # Evaluate the current policy
#     V = policy_eval(policy, Env, 0.5)
    
#     # Will be set to false if we make any changes to the policy
#     policy_stable = True
    
#     # For each state...
#     for s in range(Env.nstates):
#         # The best action we would take under the currect policy
#         chosen_a = np.argmax(policy[s])
        
#         # Find the best action by one-step lookahead
#         # Ties are resolved arbitarily
#         action_values = look(s, V)
#         best_a = np.argmax(action_values)
        
#         # Greedily update the policy
#         if chosen_a != best_a:
#             policy_stable = False
#         policy[s] = np.eye(Env.nactions)[best_a]
    
#     # If the policy is stable we've found an optimal policy. Return it
#     if policy_stable:
#         break

pp = np.reshape(np.argmax(policy, axis=1), Env.shape)

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



