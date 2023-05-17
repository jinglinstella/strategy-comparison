""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: jtao66 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903847861 (replace with your GT ID)  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import random as rand  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
class QLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    This is a Q learner object.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		  		 			  		 			     			  	 
    :type num_states: int  		  	   		  		 			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		  		 			  		 			     			  	 
    :type num_actions: int  		  	   		  		 			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type alpha: float  		  	   		  		 			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type gamma: float  		  	   		  		 			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 			  		 			     			  	 
    :type rar: float  		  	   		  		 			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 			  		 			     			  	 
    :type radr: float  		  	   		  		 			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 			  		 			     			  	 
    :type dyna: int  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_state = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.zeros((num_states, num_actions))

        if dyna > 0:
            self.t_count = np.full((num_states, num_actions, num_states), 0.00001)
            self.r = np.zeros((num_states, num_actions))
            self.history = {}

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        self.s = s

        action = np.argmax(self.q[s])

        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime]))

        if self.dyna > 0:
            history_a = self.history.get(self.s, [])
            # This line retrieves the value associated with the key self.s from the dictionary self.history.
            # If self.s is not in the dictionary, it returns an empty list [].
            # This is essentially getting the history of actions associated with the current state self.s.
            history_set = set().union(history_a, [self.a])
            self.history[self.s] = list(history_set)

            # This line updates the dictionary self.history for the key self.s with the new list of unique actions.
            # The set history_set is converted back to a list before updating the dictionary.

            # update transition matrix and reward matrix
            # T(t_count): prob that we in state s, take action a, end up in s'. T'[s,a,s']
            # R: expected reward if in state s, take action a
            # everytime we see a transition from s to s_prime with action a, we add 1

            self.t_count[self.s, self.a, s_prime] += 1
            self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + self.alpha * r

            for _ in range(self.dyna):
                self.halucinate()

        random = rand.random()

        action = np.argmax(self.q[s_prime]) if random > self.rar else rand.randint(0, self.num_actions - 1)

        self.rar = self.rar * self.radr

        self.s = s_prime
        self.a = action

        return action

    def halucinate(self):

        s = rand.choice(list(self.history.keys()))
        a = rand.choice(self.history[s])

        s_prime = np.argmax(self.t_count[s, a])

        r = self.r[s, a]

        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime]))


    def author(self):
        return 'jtao66'

if __name__ == "__main__":
    pass