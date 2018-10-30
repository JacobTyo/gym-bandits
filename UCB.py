import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt

# Environment Initializations
env = gym.make("AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward-v0")

env.reset()


# TODO: Everything 