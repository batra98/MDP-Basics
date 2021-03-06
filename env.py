from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import poisson



UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class DiscreteEnvironment(ABC):

    @abstractmethod
    def __init__(self):
        pass       
    
    @abstractmethod
    def step(self,state,action):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class Gambler_env(DiscreteEnvironment):

    def __init__(self):
        self.nstates = 101
        self.p_h = 0.3
        self.nactions = 0
        self.P = {}

        self.rewards = np.zeros(self.nstates)
        self.rewards[100] = 1

    def step(self,state,action):
        pass

    def reset(self):
        pass

    def set_p_h(self,p_h):
        self.p_h = p_h

        



class Jack_env(DiscreteEnvironment):

    def __init__(self):
        self.max_cars = 20
        self.max_move = 5
        self.mu_rent_first_location = 3
        self.mu_rent_second_location = 4
        self.mu_return_first_location = 3
        self.mu_return_second_location = 2
        self.profit = 10
        self.loss = 2
        self.nstates = self.max_cars*self.max_cars
        self.store_rent_1 = []
        self.store_rent_2 = []
        self.store_return_1 = []
        self.store_return_2 = []

        for i in range(11):
            self.store_rent_1.append(poisson.pmf(i,self.mu_rent_first_location))
            self.store_rent_2.append(poisson.pmf(i,self.mu_rent_second_location))
            self.store_return_1.append(poisson.pmf(i,self.mu_return_first_location))
            self.store_return_2.append(poisson.pmf(i,self.mu_return_second_location))

        self.actions = np.arange(-self.max_move, self.max_move + 1)

    def step(self,state,action):
        pass

    def reset(self):
        pass


    
    
    
class Grid_1(DiscreteEnvironment):

    def __init__(self):
        self.shape = [8,8]
        self.nstates = 64
        self.nactions = 4
        

        P = {}
        itr = 0
        temp = [(0 - 1) * np.random.random() for i in range(0,64)]
        # temp = [-1 for i in range(0,64)]
        # temp[0] = 0
        temp[63] = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                s = itr
                x = j
                y = i

                P[s] = {a:[] for a in range(self.nactions)}

                if s == (self.nstates-1):
                    P[s][UP] = [(1.0,s,0,True)]
                    P[s][RIGHT] = [(1.0,s,0,True)]
                    P[s][DOWN] = [(1.0,s,0,True)]
                    P[s][LEFT] = [(1.0,s,0,True)]

                else:
                    ns_up = s if y == 0 else s - self.shape[1]
                    ns_right = s if x == (self.shape[1] - 1) else s + 1
                    ns_down = s if y == (self.shape[0] - 1) else s + self.shape[1]
                    ns_left = s if x == 0 else s - 1

                    P[s][UP] = [(1.0, ns_up, temp[ns_up], False)]
                    P[s][RIGHT] = [(1.0, ns_right, temp[ns_right], False)]
                    P[s][DOWN] = [(1.0, ns_down, temp[ns_down], False)]
                    P[s][LEFT] = [(1.0, ns_left, temp[ns_left], False)]

                itr += 1

        self.P = P

    
    def step(self,state,action):
        pass

    def reset(self):
        pass
