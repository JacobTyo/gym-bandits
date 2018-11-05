import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class OdaafExpectedDelay():
    def __init__(self, env, horizon, num_arms, tolerance, expected_delay, bridge_period=25):
        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.step = 0
        self.phase_count = 0

        self.env = env

        # Hyperparameters
        self.tolerance = tolerance
        self.num_arms = num_arms
        self.expected_delay = expected_delay
        self.bridge_period = bridge_period

        self.eliminated_arms = [0 for _ in range(num_arms)]
        self.phase1_pull_results = [[] for _ in range(num_arms)]
        self.total_rewards = [0 for _ in range(horizon)]
        self.cumulative_reward = 0
        self.cumulative_reward_list = [0 for _ in range(horizon)]
        self.post_phase1_arm_averages = [0 for _ in range(num_arms)]

        self.num_required_pulls_phase1 = self.setnm()

    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * self.phase_count * self.expected_delay)) ** 2
        return nm

    def play(self):
        # be lazy for now
        while self.iteration < self.horizon:

            if self.step == 0:
                hist = self.phase1()

            elif self.step == 1:
                self.phase2()

            elif self.step == 2:
                hist = self.phase3()
        return hist

    def phase1(self):
        # This is the phase to play the arms
        for j in range(self.num_arms):
            if self.eliminated_arms[j] == 1:
                # arm has been eliminated, don't play it
                continue
            starting_i = self.iteration
            while self.iteration - starting_i <= self.num_required_pulls_phase1 and self.iteration < self.horizon:
                observation, reward, done, returned_hist = self.env.step(j)
                self.phase1_pull_results[j].append(reward)
                self.total_rewards[j] = reward
                self.cumulative_reward += reward
                self.cumulative_reward_list[self.iteration] = self.cumulative_reward
                self.iteration += 1
        self.step = 1
        return returned_hist

    def phase2(self):
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                self.post_phase1_arm_averages[j] = sum(self.phase1_pull_results[j]) / len(self.phase1_pull_results[j])

        # Eliminate sub-optimal arms
        max_arm_avg = max(self.post_phase1_arm_averages)

        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                if self.post_phase1_arm_averages[j] + self.tolerance < max_arm_avg:
                    self.eliminated_arms[j] = 1
        self.step = 2

    def phase3(self):
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                # play the arm, otherwise it has been eliminated
                for k in range(self.bridge_period):
                    if self.iteration < self.horizon:
                        _, reward, _, returned_hist = self.env.step(j)
                        self.total_rewards[self.iteration] = reward
                        self.cumulative_reward += reward
                        self.cumulative_reward_list[self.iteration] = self.cumulative_reward
                        self.iteration += 1
                    else:
                        self.step = 0
                        self.phase_count += 1
                        break
        self.step = 0
        self.phase_count += 1
        self.post_phase1_arm_averages = [0 for _ in range(self.num_arms)]
        self.phase1_pull_results = [[] for _ in range(self.num_arms)]
        self.tolerance = self.tolerance / 2.0
        self.num_required_pulls_phase1 = self.setnm()
        return returned_hist

class OdaafExpectedBoundedDelay():
    def __init__(self, env, horizon, num_arms, tolerance, expected_delay, delay_upper_bound, bridge_period=25):
        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.step = 0
        self.phase_count = 0

        self.env = env

        # Hyperparameters
        self.tolerance = tolerance
        self.num_arms = num_arms
        self.expected_delay = expected_delay
        self.bridge_period = bridge_period
        self.delay_upper_bound = delay_upper_bound

        self.eliminated_arms = [0 for _ in range(num_arms)]
        self.phase1_pull_results = [[] for _ in range(num_arms)]
        self.total_rewards = [0 for _ in range(horizon)]
        self.cumulative_reward = 0
        self.cumulative_reward_list = [0 for _ in range(horizon)]
        self.post_phase1_arm_averages = [0 for _ in range(num_arms)]

        self.num_required_pulls_phase1 = self.setnm()

    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      4 * self.tolerance * self.expected_delay)) ** 2
        d_min_comparison = ((1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * self.phase_count * self.expected_delay)) ** 2
                            ) / self.phase_count

        d_twidel = self.delay_upper_bound if self.delay_upper_bound <= d_min_comparison else d_min_comparison

        return nm if nm >= self.phase_count * d_twidel else self.phase_count * d_twidel

    def play(self):
        # determine what phase we are in and play
        if self.step == 0:
            self.phase1()

        elif self.step == 1:
            self.phase2()

        elif self.step == 2:
            self.phase3()

    def phase1(self):
        # This is the phase to play the arms
        for j in range(self.num_arms):
            if self.eliminated_arms[j] == 1:
                # arm has been eliminated, don't play it
                continue
            starting_i = self.iteration
            while self.iteration - starting_i <= self.num_required_pulls_phase1 and self.iteration < self.horizon:
                observation, reward, done, returned_hist = self.env.step(j)
                self.phase1_pull_results[j].append(reward)
                self.total_rewards[j] = reward
                self.cumulative_reward += reward
                self.cumulative_reward_list[self.iteration] = self.cumulative_reward
                self.iteration += 1
        self.step = 1

    def phase2(self):
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                self.post_phase1_arm_averages[j] = sum(self.phase1_pull_results[j]) / len(self.phase1_pull_results[j])

        # Eliminate sub-optimal arms
        max_arm_avg = max(self.post_phase1_arm_averages)

        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                if self.post_phase1_arm_averages[j] + self.tolerance < max_arm_avg:
                    self.eliminated_arms[j] = 1
        self.step = 2

    def phase3(self):
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                # play the arm, otherwise it has been eliminated
                for k in range(self.bridge_period):
                    if self.iteration < self.horizon:
                        _, reward, _, _ = self.env.step(j)
                        self.total_rewards[self.iteration] = reward
                        self.cumulative_reward += reward
                        self.cumulative_reward_list[self.iteration] = self.cumulative_reward
                        self.iteration += 1
                    else:
                        self.step = 0
                        self.phase_count += 1
                        break
        self.step = 0
        self.phase_count += 1
        self.post_phase1_arm_averages = [0 for _ in range(self.num_arms)]
        self.phase1_pull_results = [[] for _ in range(self.num_arms)]
        self.tolerance = self.tolerance / 2.0
        self.num_required_pulls_phase1 = self.setnm()


class OdaafBoundedDelayExpectationVariance():
    def __init__(self, env, horizon, num_arms, tolerance, expected_delay, delay_upper_bound, delay_variance,
                 bridge_period=25):
        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.step = 0
        self.phase_count = 0

        self.env = env

        # Hyperparameters
        self.tolerance = tolerance
        self.num_arms = num_arms
        self.expected_delay = expected_delay
        self.bridge_period = bridge_period
        self.delay_upper_bound = delay_upper_bound
        self.delay_variance = delay_variance

        self.eliminated_arms = [0 for _ in range(num_arms)]
        self.phase1_pull_results = [[] for _ in range(num_arms)]
        self.total_rewards = [0 for _ in range(horizon)]
        self.cumulative_reward = 0
        self.cumulative_reward_list = [0 for _ in range(horizon)]
        self.post_phase1_arm_averages = [0 for _ in range(num_arms)]

        self.num_required_pulls_phase1 = self.setnm()

    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      4 * self.tolerance * (self.expected_delay +
                                                                            2 * self.delay_variance))) ** 2

        return nm

    def play(self):
        # determine what phase we are in and play
        if self.step == 0:
            self.phase1()

        elif self.step == 1:
            self.phase2()

        elif self.step == 2:
            self.phase3()

    def phase1(self):
        # This is the phase to play the arms
        for j in range(self.num_arms):
            if self.eliminated_arms[j] == 1:
                # arm has been eliminated, don't play it
                continue
            starting_i = self.iteration
            while self.iteration - starting_i <= self.num_required_pulls_phase1 and self.iteration < self.horizon:
                observation, reward, done, returned_hist = self.env.step(j)
                self.phase1_pull_results[j].append(reward)
                self.total_rewards[j] = reward
                self.cumulative_reward += reward
                self.cumulative_reward_list[self.iteration] = self.cumulative_reward
                self.iteration += 1
        self.step = 1

    def phase2(self):
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                self.post_phase1_arm_averages[j] = sum(self.phase1_pull_results[j]) / len(self.phase1_pull_results[j])

        # Eliminate sub-optimal arms
        max_arm_avg = max(self.post_phase1_arm_averages)

        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                if self.post_phase1_arm_averages[j] + self.tolerance < max_arm_avg:
                    self.eliminated_arms[j] = 1
        self.step = 2

    def phase3(self):
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                # play the arm, otherwise it has been eliminated
                for k in range(self.bridge_period):
                    if self.iteration < self.horizon:
                        _, reward, _, _ = self.env.step(j)
                        self.total_rewards[self.iteration] = reward
                        self.cumulative_reward += reward
                        self.cumulative_reward_list[self.iteration] = self.cumulative_reward
                        self.iteration += 1
                    else:
                        self.step = 0
                        self.phase_count += 1
                        break
        self.step = 0
        self.phase_count += 1
        self.post_phase1_arm_averages = [0 for _ in range(self.num_arms)]
        self.phase1_pull_results = [[] for _ in range(self.num_arms)]
        self.tolerance = self.tolerance / 2.0
        self.num_required_pulls_phase1 = self.setnm()
