#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from scipy.stats import poisson
import env

Jack_env = env.Jack_env()



# all possible actions
actions = np.arange(-Jack_env.max_move, Jack_env.max_move + 1)

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UPPER_BOUND = 11

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poisson_cache = dict()


def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


def expected_return(state, action, state_value, env):
    
    value = 0.0

    # cost for moving cars
    value -= env.loss * abs(action)

    # moving cars
    cars_at_1 = min(state[0] - action, env.max_cars)
    cars_at_2 = min(state[1] + action, env.max_cars)

    # go through all possible rental requests
    for rent_1 in range(POISSON_UPPER_BOUND):
        for rent_2 in range(POISSON_UPPER_BOUND):
            # probability for current combination of rental requests
            prob = poisson_probability(rent_1, env.mu_rent_first_location) * poisson_probability(rent_2, env.mu_rent_second_location)
                

            num_of_cars_first_loc = cars_at_1
            num_of_cars_second_loc = cars_at_2

            # valid rental requests should be less than actual # of cars
            valid_rental_first_loc = min(num_of_cars_first_loc, rent_1)
            valid_rental_second_loc = min(num_of_cars_second_loc, rent_2)

            # get credits for renting
            reward = (valid_rental_first_loc + valid_rental_second_loc) * env.profit
            num_of_cars_first_loc -= valid_rental_first_loc
            num_of_cars_second_loc -= valid_rental_second_loc

            
            for returned_cars_first_loc in range(POISSON_UPPER_BOUND):
                for returned_cars_second_loc in range(POISSON_UPPER_BOUND):
                    prob_return = poisson_probability(
                        returned_cars_first_loc, env.mu_return_first_location) * poisson_probability(returned_cars_second_loc, env.mu_return_second_location)
                    num_of_cars_first_loc_ = min(num_of_cars_first_loc + returned_cars_first_loc, env.max_cars)
                    num_of_cars_second_loc_ = min(num_of_cars_second_loc + returned_cars_second_loc, env.max_cars)
                    # print(num_of_cars_first_loc)
                    prob_ = prob_return * prob
                    value += prob_ * (reward + 0.9 *
                                        state_value[num_of_cars_first_loc_][num_of_cars_second_loc_])
    return value


# V = np.zeros((Jack_env.max_cars+1,Jack_env.max_cars+1))
# policy = np.zeros()

V = np.zeros((Jack_env.max_cars + 1, Jack_env.max_cars + 1))
policy = np.zeros(V.shape, dtype=np.int)

while True:
    while True:
        V_2 = V.copy()
        for i in range(Jack_env.max_cars + 1):
            for j in range(Jack_env.max_cars + 1):
                new_state_value = expected_return([i, j], policy[i][j], V, Jack_env)
                print(new_state_value)
                V[i][j] = new_state_value
        max_value_change = abs(V_2 - V).max()
        print('max value change {}'.format(max_value_change))
        if max_value_change < 0.001:
            break

# def figure_4_2(constant_returned_cars=False):
#     value = np.zeros((Jack_env.max_cars + 1, Jack_env.max_cars + 1))
#     policy = np.zeros(value.shape, dtype=np.int)

#     iterations = 0
#     _, axes = plt.subplots(2, 3, figsize=(40, 20))
#     plt.subplots_adjust(wspace=0.1, hspace=0.2)
#     axes = axes.flatten()
#     while True:
#         fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
#         fig.set_ylabel('# cars at first location', fontsize=30)
#         fig.set_yticks(list(reversed(range(Jack_env.max_cars + 1))))
#         fig.set_xlabel('# cars at second location', fontsize=30)
#         fig.set_title('policy {}'.format(iterations), fontsize=30)

#         # policy evaluation (in-place)
#         while True:
#             old_value = value.copy()
#             for i in range(Jack_env.max_cars + 1):
#                 for j in range(Jack_env.max_cars + 1):
#                     new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
#                     print(new_state_value)
#                     value[i, j] = new_state_value
#             max_value_change = abs(old_value - value).max()
#             print('max value change {}'.format(max_value_change))
#             if max_value_change < 0.001:
#                 break

#         # policy improvement
#         policy_stable = True
#         for i in range(Jack_env.max_cars + 1):
#             for j in range(Jack_env.max_cars + 1):
#                 old_action = policy[i, j]
#                 action_returns = []
#                 for action in actions:
#                     if (0 <= action <= i) or (-j <= action <= 0):
#                         action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
#                     else:
#                         action_returns.append(-np.inf)
#                 new_action = actions[np.argmax(action_returns)]
#                 policy[i, j] = new_action
#                 if policy_stable and old_action != new_action:
#                     policy_stable = False
#         #   ('policy stable {}'.format(policy_stable))

#         if policy_stable:
#             fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
#             fig.set_ylabel('# cars at first location', fontsize=30)
#             fig.set_yticks(list(reversed(range(Jack_env.max_cars + 1))))
#             fig.set_xlabel('# cars at second location', fontsize=30)
#             fig.set_title('optimal value', fontsize=30)
#             break

#         iterations += 1

#     plt.savefig('./figure_4_2.png')
#     plt.close()


# if __name__ == '__main__':
#     figure_4_2()
