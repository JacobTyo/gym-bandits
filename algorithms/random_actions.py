import numpy as np


class randomActions():

    def __init__(self, arms):
        self.num_arms = arms

    def play(self, **kwargs):
        return np.random.randint(0, self.num_arms, 1)
