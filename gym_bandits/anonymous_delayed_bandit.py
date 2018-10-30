import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class AnonymousDelayedBanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations - This environment returns accumulated rewards at stochastic
    timesteps with respect to the d_dist.  The source of the rewards are not known, only
    the sum of the rewards.  However, everytime a reward is received, it is a complete
    payout (i.e. no rewards will be left in the system immediately after a payout)

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """
    def __init__(self, p_dist, r_dist, d_dist=None, horizon=1000):
        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist
        self.d_dist = d_dist

        self.time_step = 0

        self.done = False

        self.horizon = horizon
        self.reward = [ 0 for i in range(horizon)]
        self.history = {'arm': [],
                        'reward': [],
                        'delay': [],
                        'received': []}

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        reward_from_this_pull = 0
        delay_this_pull = None

        # This looks like it is either a uniform or a normal ??
        if np.random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward_from_this_pull = self.r_dist[action]
                delay_this_pull = int(self.d_dist[action]) - 1
                self.reward[delay_this_pull] += reward_from_this_pull
            else:
                reward_from_this_pull = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])
                delay_this_pull = int(self.d_dist[action]) - 1
                self.reward[delay_this_pull] += reward_from_this_pull

        reward_this_timestep = self.reward[0]
        del self.reward[0]
        self.reward.insert(len(self.reward), 0)

        self.history['arm'].append(action)
        self.history['reward'].append(reward_from_this_pull)
        # print("reward this pull: ", reward_from_this_pull)
        self.history['delay'].append(delay_this_pull)
        # print("Delay this pull", delay_this_pull)
        self.history['received'].append(reward_this_timestep)

        self.time_step += 1

        if self.time_step > 9999:
            self.done = True

        return self.time_step, reward_this_timestep, self.done, {}

    def _reset(self):
        self.time_step = 0
        return 0

    def _render(self, mode='human', close=False):
        pass

    def get_history(self):
        return self.history.copy()

    def normalize_delay(self, delay):
        # not sure for now, so just mult by 10 and round?
        return np.random.choice(delay)


class AnonymousDelayedBanditTwoArmedDeterministicFixed(AnonymousDelayedBanditEnv):
    """Simplest case where one bandit always pays, and the other always doesn't"""
    def __init__(self):
        AnonymousDelayedBanditEnv.__init__(self, p_dist=[1, 0], r_dist=[1, 1], d_dist=[1, 1])


class AnonymousDelayedBanditTenArmedRandomFixed(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""
    def __init__(self, bandits=10):
        p_dist = np.random.uniform(size=bandits)
        r_dist = np.full(bandits, 1)
        d_dist = np.random.uniform(size=bandits)
        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)

class AnonymousDelayedBanditTwoArm(AnonymousDelayedBanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""
    def __init__(self, bandits=10):
        p_dist = [1, 1]
        r_dist = [1, 1]
        d_dist = [2, 1]

        d_dist = np.random.uniform(size=bandits)
        AnonymousDelayedBanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist, d_dist=d_dist)
