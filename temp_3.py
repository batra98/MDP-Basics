import env
import agent
import numpy as np
import matplotlib.pyplot as plt


Gambler_env = env.Gambler_env()


def look(s,V,env):
	A = np.zeros(env.nstates)
	possible_bet = range(1,min(s,100-s)+1)

	for a in possible_bet:
		A[a] = env.p_h * (env.rewards[s+a]+V[s+a]*1)+ (1-env.p_h) * (env.rewards[s-a]+V[s-a]*1)

	return A



V = np.zeros(Gambler_env.nstates)
V2 = np.zeros(Gambler_env.nstates)
# y = []
# itr = 0
while True:
	delta = 0
	V = np.copy(V2)

	for s in range(1,Gambler_env.nstates-1):
		A = look(s,V,Gambler_env)
		best_action_value = np.max(A)
		delta = max(delta,np.abs(best_action_value-V[s]))
		V2[s] = best_action_value

	

	if delta < 0.0001:
		break

	# itr += 1

print(itr)

policy = np.zeros(Gambler_env.nstates-1)
for s in range(1, Gambler_env.nstates-1):
    # One step lookahead to find the best action for this state
    A = look(s, V, Gambler_env)
    best_action = np.argmax(A)
    # Always take the best action
    policy[s] = best_action

print(policy)
print(V)

# x = range(100)

# for i in range(1,len(y)):
# 	plt.plot(x,y[i][:100])

# plt.show()

# Plotting Capital vs Final Policy

# x axis values
x = range(100)
# corresponding y axis values
y = policy
 
# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')
 
# giving a title to the graph
plt.title('Capital vs Final Policy')
 
# function to show the plot
plt.show()
