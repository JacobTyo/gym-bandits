import numpy as np


class randomActions():

    def __init__(self, arms):
        self.num_arms = arms

    def play(self, reward, **kwargs):
        return np.random.randint(low=0, high=self.num_arms)
