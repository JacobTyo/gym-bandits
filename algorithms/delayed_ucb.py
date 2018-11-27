import numpy as np

class Delayed_Ucb():
    def __init__(self, k, args):
        # number of arms
        self.k = k
        # ucb probability
        # self.delta = args.ucb_delta
        # current timestep; completed steps
        self.t = 0
        # empirical mean estimates
        self.means = np.zeros(k)
        # count of received rewards per arm
        self.S = np.zeros(k)

        self.last_action = None



    def play(self, reward, non_anon_reward, **kwargs):
        # acumulate reward
        for reward in non_anon_reward:
            self.means[reward[0]] = self.S[reward[0]] * self.means[reward[0]] \
                                           / (self.S[reward[0]] + 1) + (reward[1] / (self.S[reward[0]] + 1))
            self.S[reward[0]] += 1

        # choose action
        action = None
        if self.t < self.k:
            action = self.t

        elif not non_anon_reward:
            action = self.last_action
        else:
            if 0 in self.S:
                ucbs = self.means + np.sqrt((4 * np.log(self.t)) / (self.S + 0.00001))
            else:
                ucbs = self.means + np.sqrt((4 * np.log(self.t)) / self.S)
            action = np.argmax(ucbs)

        self.last_action = action
        self.t += 1
        return action