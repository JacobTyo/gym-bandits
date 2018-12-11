import numpy as np


class PhasedUcb():

    def __init__(self, k, args):
        # number of arms
        self.k = k
        # ucb probability
        # self.delta = args.ucb_delta
        # current timestep; completed steps
        self.t = 0
        # empirical mean estimates
        self.means = np.zeros(k)
        # count of plays per arm
        self.T = np.zeros(k)

        self.counter = 0
        self.round_size = 200

        self.last_action = None

    def play(self, reward, **kwargs):

        # acumulate reward
        if self.last_action is not None:
            self.means[self.last_action] = self.T[self.last_action] * self.means[self.last_action] \
                                           / (self.T[self.last_action] + 1) + (reward / (self.T[self.last_action] + 1))
            self.T[self.last_action] += 1

        # choose action
        if self.t < self.round_size * self.k:
            action = int(self.t/self.round_size)
            self.counter = -1

        else:

            if self.counter == 0:
                ucbs = self.means + np.sqrt((4 * np.log(self.t)) / self.T)
                action = np.argmax(ucbs)
            elif self.counter < self.round_size:
                action = self.last_action
            else:
                action = self.last_action
                self.counter = -1

        self.last_action = action
        self.t += 1

        self.counter += 1
        return action
