import gym
import gym_bandits
import numpy as np


env = gym.make("AnonymousDelayedBanditTwoArmedStochasticDelayStochasticReward-v0")

env.reset()

total_reward = 0
for i in range(1000):
    action = np.random.choice(np.asarray([0, 1]))
    observation, reward, done, returned_hist = env.step(action)
    observation -= 1

print(returned_hist['received'])
