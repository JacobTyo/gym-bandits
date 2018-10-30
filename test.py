import gym
import gym_bandits
import numpy as np


env = gym.make("AnonymousDelayedBanditTwoArm-v0")

env.reset()

total_reward = 0
for i in range(10000):
    action = np.random.choice(np.asarray([0,1]))
    observation, reward, done, _ = env.step(action)
    if reward > 1:
        print(action)
        print(observation)
        print(reward)
        print('---------')
    # if done:
    #     break

