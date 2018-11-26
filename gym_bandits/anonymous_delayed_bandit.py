import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from types import FunctionType
import functools
import random


class AnonymousDelayedBanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations - This environment returns accumulated rewards at stochastic
    timesteps with respect to the d_dist.  The source of the rewards are not known, only
    the sum of the rewards to be received at each time step.

    p_dist:
        A list of distributions for the arms to pay out at (a list of numbers corresponding to the percentage chance
        of payout per arm)
    r_dist:
        A list of distributions for the rewards that the arms distribute (a list of numbers corresponds to a set
        return value, whereas a list of funcions corresponds to distributions to be sampled from to get reward)
    d_dist:
        A list of distributions to which the delays of each arm are sampled (a list of numbers corresponds to
        deterministic delays, whereas a list of functions corresponds to distributions for the delays to be sampled
        from.  Note that this should numbers or a distribution bounded by [0, horizon]
    """
    def __init__(self, p_dist, r_dist, d_dist, horizon=5000):
        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        self.types = { "p_dist" : float,
                       "r_dist" : float,
                       "d_dist" : float}

        self.p_dist = p_dist
        self.r_dist = r_dist
        self.d_dist = d_dist

        self.time_step = 0

        self.done = False

        self.horizon = horizon
        self.reward = [0 for i in range(self.horizon)]
        self.non_anon_reward = [[] for i in range(self.horizon + 1)]
        self.history = {'arm': [],
                        'reward': [],
                        'delay': [],
                        'received': [],
                        'expected_reward': [],
                        'non_anon_reward': [],
                        'optimal_mean': np.amax(self.means)}

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        self.seed()

        # determine if the p_dist, r_dist, or d_dist are lists of numbers or lists of functions: (assume no mixing)
        if callable(p_dist[0]):
            # then we are dealing with functions
            self.types["p_dist"] = FunctionType
        if callable(r_dist[0]):
            self.types["r_dist"] = FunctionType
        if callable(d_dist[0]):
            self.types["d_dist"] = FunctionType

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, arm):
        assert self.action_space.contains(arm)

        reward_from_this_pull = 0
        delay_this_pull = None

        arm_did_payout = True if np.random.uniform(0, 1, 1)[0] <= self.p_dist[arm] else False

        if arm_did_payout:
            # If the arm payed out, what was the reward?
            if self.types["r_dist"] == FunctionType:  # TODO: verify but I think this one is good
                reward_from_this_pull = self.r_dist[arm]()
            else:
                reward_from_this_pull = self.r_dist[arm]

            if type(reward_from_this_pull) is not int or type(reward_from_this_pull) is not float:
                reward_from_this_pull = reward_from_this_pull[0]

            # and what delay is associated with that reward?
            if self.types["d_dist"] == FunctionType:
                delay_this_pull = self.d_dist[arm]()
            else:
                delay_this_pull = self.d_dist[arm]

        # ensure that delay is non-negative int (it is an index)
        delay_this_pull = int(delay_this_pull) if int(delay_this_pull) >= 0 else 0
        delay_this_pull = delay_this_pull if delay_this_pull < len(self.reward) else len(self.reward)

        # add the reward to the corresponding location in self.reward
        self.reward[delay_this_pull] += reward_from_this_pull
        self.non_anon_reward[delay_this_pull].append((arm, reward_from_this_pull))

        # TODO: THis is probably inefficient for large horizons - should rethink
        reward_this_timestep = self.reward.pop(0)
        self.reward.append(0)
        self.non_anon_reward.append([])

        self.history['arm'].append(arm)
        self.history['reward'].append(reward_from_this_pull)
        self.history['delay'].append(delay_this_pull)
        self.history['received'].append(reward_this_timestep)
        self.history['expected_reward'].append(self.means[arm])
        self.history['non_anon_reward'] = self.non_anon_reward.pop(0)
        self.time_step += 1

        # Technically, no reason to ever be "done"
        # if self.time_step > self.horizon:
        #     self.done = True

        return self.time_step, reward_this_timestep, self.done, self.history

    def reset(self):
        self.time_step = 0
        self.history = {'arm': [],
                        'reward': [],
                        'delay': [],
                        'received': [],
                        'expected_reward': [],
                        'non_anon_reward': [],
                        'optimal_mean': np.amax(self.means)}
        self.reward = [0 for i in range(self.horizon)]
        self.non_anon_reward = [[] for i in range(self.horizon)]
        return 0

    def render(self, mode='human', close=False):
        pass


class AnonymousDelayedBanditTwoArmedDeterministic(AnonymousDelayedBanditEnv):
    """Simplest case where one bandit always pays, and the other always doesn't"""
    def __init__(self):
        AnonymousDelayedBanditEnv.__init__(self, p_dist=[1, 1], r_dist=[3, 2], d_dist=[2, 0])


class AnonymousDelayedBanditTwoArmStochasticReward(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""
    def __init__(self, bandits=2):
        p_dist = [1, 1]
        r_dist = [functools.partial(np.random.uniform, 1, 3, 1), functools.partial(np.random.uniform, 2, 4, 1)]
        d_dist = [2, 5]

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AnonymousDelayedBanditTwoArmedStochasticDelay(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""

    def __init__(self, bandits=2):
        p_dist = [1, 1]
        r_dist = [2, 5]
        d_dist = [functools.partial(np.random.uniform, 1, 3, 1), functools.partial(np.random.uniform, 3, 5, 1)]

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AnonymousDelayedBanditTwoArmedStochasticDelayStochasticReward(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""

    def __init__(self, bandits=2):
        p_dist = [1, 1]
        r_dist = [functools.partial(np.random.uniform, 1, 3, 1), functools.partial(np.random.uniform, 3, 5, 1)]
        d_dist = [functools.partial(np.random.uniform, 1, 3, 1), functools.partial(np.random.uniform, 3, 5, 1)]

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward1(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""

    def __init__(self, bandits=10):
        p_dist = [1 for i in range(bandits)]

        r_dist = [functools.partial(np.random.uniform, 1, 3, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, -2, -1, 1),
                  functools.partial(np.random.uniform, -5, -3, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 1, 10, 1),
                  functools.partial(np.random.uniform, -10, -1, 1),
                  functools.partial(np.random.uniform, -1, 0, 1)]

        d_dist = [functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 8, 1),
                  functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 1, 1),
                  functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 10, 1),
                  functools.partial(np.random.poisson, 10, 1)]

        self.means = [2.0, 4.0, 0.5, -1.5, -4, 0.5, 1.5, 5.5, -5.5, -0.5]
        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""

    def __init__(self, bandits=10):
        p_dist = [1 for i in range(bandits)]

        r_dist = [functools.partial(np.random.uniform, 1, 3, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, -2, -1, 1),
                  functools.partial(np.random.uniform, -5, -3, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 1, 10, 1),
                  functools.partial(np.random.uniform, -10, -1, 1),
                  functools.partial(np.random.uniform, -1, 0, 1)]

        self.means = [2.0, 4.0, 0.5, -1.5, -4, 0.5, 1.5, 5.5, -5.5, -0.5]

        d_dist = [functools.partial(np.random.uniform, 1, 3, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, 5, 12, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 7, 12, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 5, 10, 1)]

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class Chirag(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""

    def __init__(self, bandits=10):
        p_dist = [1 for i in range(bandits)]

        r_dist = [functools.partial(np.random.uniform, 1, 3, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, -2, -1, 1),
                  functools.partial(np.random.uniform, -5, -3, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 1, 10, 1),
                  functools.partial(np.random.uniform, -10, -1, 1),
                  functools.partial(np.random.uniform, 100, 120, 1)]

        self.means = [2.0, 4.0, 0.5, -1.5, -4, 0.5, 1.5, 5.5, -5.5, 0]

        d_dist = [functools.partial(np.random.uniform, 1, 3, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 0, 1, 1),
                  functools.partial(np.random.uniform, 5, 12, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 3, 5, 1),
                  functools.partial(np.random.uniform, 7, 12, 1),
                  functools.partial(np.random.uniform, 1, 2, 1),
                  functools.partial(np.random.uniform, 200, 500, 1)]

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AdaBanditsBaseline(AnonymousDelayedBanditEnv):
    def __init__(self, bandits=10):
        p_dist = [1 for i in range(bandits)]

        r_dist = [functools.partial(np.random.normal, 0.1, 2, 1),
                  functools.partial(np.random.normal, .99, 2, 1),
                  functools.partial(np.random.normal, 0.3, 2, 1),
                  functools.partial(np.random.normal, 0.8, 2, 1),
                  functools.partial(np.random.normal, 0.5, 2, 1),
                  functools.partial(np.random.normal, 0.6, 2, 1),
                  functools.partial(np.random.normal, 0.2, 2, 1),
                  functools.partial(np.random.normal, 0.4, 2, 1),
                  functools.partial(np.random.normal, 0.9, 2, 1),
                  functools.partial(np.random.normal, 0.7, 2, 1)]

        self.means = [0.1, 0.99, 0.3, 0.8, 0.5, 0.6, 0.2, 0.4, 0.9, 0.7]

        d_dist = [functools.partial(np.random.poisson, 800, 1),
                  functools.partial(np.random.poisson, 300, 1),
                  functools.partial(np.random.poisson, 400, 1),
                  functools.partial(np.random.poisson, 700, 1),
                  functools.partial(np.random.poisson, 200, 1),
                  functools.partial(np.random.poisson, 100, 1),
                  functools.partial(np.random.poisson, 700, 1),
                  functools.partial(np.random.poisson, 900, 1),
                  functools.partial(np.random.poisson, 200, 1),
                  functools.partial(np.random.poisson, 500, 1)]

        c = list(zip(r_dist, self.means, d_dist))

        random.shuffle(c)

        r_dist, self.means, d_dist = zip(*c)

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AdaBanditsBaseline_Optimistic(AnonymousDelayedBanditEnv):
    def __init__(self, bandits=10):
        p_dist = [1 for i in range(bandits)]

        r_dist = [functools.partial(np.random.normal, 0.1, 0.01, 1),
                  functools.partial(np.random.normal, .99, 0.01, 1),
                  functools.partial(np.random.normal, 0.3, 0.01, 1),
                  functools.partial(np.random.normal, 0.8, 0.01, 1),
                  functools.partial(np.random.normal, 0.5, 0.01, 1),
                  functools.partial(np.random.normal, 0.6, 0.01, 1),
                  functools.partial(np.random.normal, 0.2, 0.01, 1),
                  functools.partial(np.random.normal, 0.4, 0.01, 1),
                  functools.partial(np.random.normal, 0.9, 0.01, 1),
                  functools.partial(np.random.normal, 0.7, 0.01, 1)]

        self.means = [0.1, 0.99, 0.3, 0.8, 0.5, 0.6, 0.2, 0.4, 0.9, 0.7]

        d_dist = [functools.partial(np.random.poisson, 800, 1),
                  functools.partial(np.random.poisson, 300, 1),
                  functools.partial(np.random.poisson, 400, 1),
                  functools.partial(np.random.poisson, 700, 1),
                  functools.partial(np.random.poisson, 200, 1),
                  functools.partial(np.random.poisson, 100, 1),
                  functools.partial(np.random.poisson, 700, 1),
                  functools.partial(np.random.poisson, 900, 1),
                  functools.partial(np.random.poisson, 200, 1),
                  functools.partial(np.random.poisson, 500, 1)]

        # c = list(zip(r_dist, self.means, d_dist))
        #
        # random.shuffle(c)
        #
        # r_dist, self.means, d_dist = zip(*c)

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)


class AdaBanditsOutliers(AnonymousDelayedBanditEnv):
    def __init__(self, bandits=10):
        p_dist = [1 for i in range(bandits)]

        r_dist = [functools.partial(np.random.normal, 0.01, 1, 1),
                  functools.partial(np.random.normal, 0.1, 1, 1),
                  functools.partial(np.random.normal, 0.2, 1, 1),
                  functools.partial(np.random.normal, 0.3, 1, 1),
                  functools.partial(np.random.normal, 0.4, 1, 1),
                  functools.partial(np.random.normal, 0.5, 1, 1),
                  functools.partial(np.random.normal, 0.7, 1, 1),
                  functools.partial(np.random.normal, 0.8, 1, 1),
                  functools.partial(np.random.normal, 0.9, 1, 1),
                  functools.partial(np.random.normal, 0.99, 1, 1)]

        self.means = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.99]

        d_dist = [functools.partial(np.random.poisson, 8, 1),
                  functools.partial(np.random.poisson, 3, 1),
                  functools.partial(np.random.poisson, 4, 1),
                  functools.partial(np.random.poisson, 7, 1),
                  functools.partial(np.random.poisson, 2, 1),
                  functools.partial(np.random.poisson, 1, 1),
                  functools.partial(np.random.poisson, 7, 1),
                  functools.partial(np.random.poisson, 9, 1),
                  functools.partial(np.random.poisson, 25, 1),  # This is the outlier
                  functools.partial(np.random.poisson, 5, 1)]

        c = list(zip(r_dist, self.means, d_dist))

        random.shuffle(c)

        r_dist, self.means, d_dist = zip(*c)

        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)