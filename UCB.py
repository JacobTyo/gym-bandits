import numpy as np

class UCB():

    def __init__(self, k, delta):
        # number of arms
        self.k = k
        # ucb probability
        self.delta = delta
        # current timestep; completed steps
        self.t = 0
        # empirical mean estimates
        self.means = np.zeros(k)
        # count of plays per arm
        self.T = np.zeros(k)

        self.last_action = None



    def act(self, reward):
        # acumulate reward
        if self.last_action is not None:
            self.means[self.last_action] = self.T[self.last_action] * self.means[self.last_action] \
                                           / (self.T[self.last_action] + 1) + (reward / (self.T[self.last_action] + 1))

        # choose action
        action = None
        if self.t < self.k:
            action = self.t
        else:
            ucbs = self.means + np.sqrt((2 * np.log(1 / self.delta)) / self.T)
            action = np.argmax(ucbs)

        self.last_action = action
        T[action] += 1
        self.t += 1
        return action