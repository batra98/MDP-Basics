from abc import ABC, abstractmethod
import numpy as np


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class DiscreteEnvironment(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def is_terminal(state):
        pass        
    
    @abstractmethod
    def step(self,state,action):
        pass
    
    @abstractmethod
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



    def is_terminal(self,state):
        x,y = state
        return (x == 0 and y == 0) or (x == self.size - 1 and y == self.size - 1)
    
    def step(self,state,action):
        if is_terminal(state):
            return state, self.reward_terminal
        
        
        next_state = [(np.array(state)+action)]
        x,y = next_state
        
        if x<0 or x >= self.nstates or y<0 or y >= self.nstates:
            next_state = state
            
        return next_state,self.reward_terminal

    def reset(self):
        pass
