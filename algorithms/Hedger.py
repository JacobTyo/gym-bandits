import numpy as np


class Hedger:
    def __init__(self, horizon, num_arms, tolerance, expected_delay, bridge_period=25):

        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.step = 0

        self.received_rewards = [[] for _ in range(num_arms)]
        self.selected_arm = 0
        self.previously_selected_arm = 0
        self.phase_count = 1
        self.num_arms = num_arms

        self.estimated_arm_averages = [0 for _ in range(num_arms)]
        self.is_arm_estimated = [False for _ in range(num_arms)]
        self.is_arm_eliminated = [False for _ in range(num_arms)]

        self.hedging_iterations_this_arm = 0

        # Hyperparameters
        self.tolerance = tolerance
        self.regret_upper_bound = []
        self.num_arms = num_arms
        self.expected_delay = expected_delay
        self.bridge_period = bridge_period

        self.nm = self.setnm()
        self.little_nm = self.set_little_nm()

    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * (
                                                                  self.phase_count) * self.expected_delay)) ** 2
        self.phase_count += 1
        return int(nm)

    def set_little_nm(self):
        term2 = np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * (self.phase_count + 1) * self.expected_delay)
        term5 = (1.0 / (self.tolerance ** 2)) * term2 ** 2

        return self.nm - int(term5)

    def play(self, reward, **kwargs):

        test = 2 * int(self.nm)
        # record reward from previous action
        if self.iteration > 0:
            self.received_rewards[self.selected_arm].append(reward)

        # for the first two arms, just pull them as odaaf
        if self.iteration < 2 * self.nm:
            # just pull the first two arms nm times
            action = self.step0()

        # bridge period
        elif self.iteration == 2 * self.nm:
            self.calculate_arm_averages_normal()
            action = self.bridge_step()

        elif self.iteration < 2 * self.nm + self.bridge_period:
            action = self.bridge_step()

        # now we need to systematically estimate the averages for the unpulled arms
        elif False in self.is_arm_estimated:
            # there still exists some arms that are not estimated
            action = self.estimate_with_hedging()

        # else just pull the best arm
        else:
            action = self.get_best_arm()

        self.iteration += 1
        return action

    def step0(self):

        # pull the first arm nm times
        if self.iteration < self.nm:
            self.selected_arm = 0
            return self.selected_arm

        # pull the second arm nm times
        elif self.iteration < 2 * self.nm:
            self.selected_arm = 1
            return self.selected_arm

    def bridge_step(self):
        # just pull max arm so far
        return self.get_best_arm()

    def estimate_with_hedging(self):
        arm_to_estimate = -1
        for i in range(len(self.is_arm_estimated)):
            if not self.is_arm_estimated:
                arm_to_estimate = i

        # should never be here
        if arm_to_estimate == -1:
            self.selected_arm = self.get_best_arm()
            return self.selected_arm

        self.selected_arm = self.get_best_arm()

        # estimate arm i - pull i little nm times
        if self.hedging_iterations_this_arm < self.little_nm:
            self.previously_selected_arm = arm_to_estimate
            self.selected_arm = arm_to_estimate

        # if we have pulled enough times, calculate new estimate
        elif self.hedging_iterations_this_arm >= self.nm:
            self.calculated_arm_averages_mixed()
            self.hedging_iterations_this_arm = 0
            return self.selected_arm

        self.hedging_iterations_this_arm += 1

        return self.selected_arm

    def calculate_arm_averages_normal(self):
        # for each arm, calculate the average reward
        for i in range(self.num_arms):
            num_rewards_received = len(self.received_rewards[i])
            if num_rewards_received > 0:
                self.estimated_arm_averages[i] = sum(self.received_rewards[i]) / len(self.received_rewards[i])
                self.is_arm_estimated[i] = True

        # reset rewards buffers
        self.received_rewards = [[] for _ in range(self.num_arms)]

    def calculated_arm_averages_mixed(self):
        # for the recently played arms, calculate the new average
        self.estimated_arm_averages[self.previously_selected_arm] = (
            sum(self.received_rewards[self.selected_arm] + sum(self.received_rewards[self.previously_selected_arm])) / (
                    len(self.received_rewards[self.selected_arm]) +
                    len(self.received_rewards[self.previously_selected_arm])) -
            self.estimated_arm_averages[self.selected_arm])

        # Make sure to update the list of estimated arms
        self.is_arm_estimated[self.previously_selected_arm] = True

    def get_best_arm(self):
            return np.argmax(np.asarray(self.estimated_arm_averages))
