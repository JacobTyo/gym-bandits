import numpy as np


class PhasedHedger:
    def __init__(self, horizon, num_arms, tolerance, expected_delay, bridge_period=25):

        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.iterations_this_phase = 0
        self.step = 0

        self.received_rewards = [[] for _ in range(num_arms)]
        self.selected_arm = 0
        self.previously_selected_arm = 0
        self.phase_count = 1
        self.num_arms = num_arms
        self.first_time_1a = True
        self.estimate_this_arm = 0

        self.estimated_arm_averages = [0 for _ in range(num_arms)]
        self.is_arm_estimated = [False for _ in range(num_arms)]
        self.is_arm_eliminated = [False for _ in range(num_arms)]

        self.previous_arm_averages = [0 for _ in range(num_arms)]

        self.hedging_iterations_this_arm = 0

        self.bridge_iterations = 0

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

        # record reward from previous action
        if self.iteration > 0:
            self.received_rewards[self.selected_arm].append(reward)

        if sum(self.is_arm_eliminated) >= 7:
            self.selected_arm = np.argmax(np.asarray(self.previous_arm_averages))
            return self.selected_arm

        if self.step == 0:
            action = self.step0()
        elif self.step == 1:
            action = self.step1()
        elif self.step == 2:
            action = self.step2()

        self.iterations_this_phase += 1
        self.iteration += 1

        return action

    def step0(self):

        if self.iterations_this_phase == 0:
            self.selected_arm = self.get_next_arm(0)
        elif self.iterations_this_phase == self.nm:
            # should calculate the arm average so far
            self.calculate_arm_averages_normal(self.selected_arm)
            # now get next arm and continue playing
            self.selected_arm = self.get_next_arm(self.selected_arm + 1)
        elif self.iterations_this_phase == 2 * self.nm + 1:
            self.calculate_arm_averages_normal(self.selected_arm)
            self.selected_arm = self.step1()
            self.step += 1

        return self.selected_arm

    def step1(self):

        if self.iterations_this_phase == self.nm + 1:
            self.selected_arm = self.get_next_arm(self.selected_arm + 1)
            self.estimate_this_arm = self.selected_arm

        elif self.iterations_this_phase % self.nm + 1 == 0:
            self.selected_arm = self.get_next_arm(self.selected_arm + 1)
            self.estimate_this_arm = self.selected_arm

        elif self.iterations_this_phase % self.nm + self.little_nm == 0:
            # now calculate the arm average for the first arm that has been "mixed"
            self.selected_arm = self.get_best_arm()

        elif self.iterations_this_phase % self.num_arms * self.nm == 0:
            # end of this step, estimate, bridge period, then restart
            self.calculated_arm_averages_mixed(self.estimate_this_arm, self.selected_arm)
            self.selected_arm = self.get_best_arm()
            self.step += 1

        return self.selected_arm

    def step2(self):
        self.selected_arm = self.get_best_arm()
        # just pull max arm so far
        self.bridge_iterations += 1

        if self.bridge_iterations > self.bridge_period:
            # reset everything
            self.bridge_iterations = 0
            self.step = 0
            self.iterations_this_phase = 0

            self.elimination_phase()

            self.received_rewards = [[] for _ in range(self.num_arms)]
            self.previous_arm_averages = self.estimated_arm_averages
            self.estimated_arm_averages = [0 for _ in range(self.num_arms)]

        return self.selected_arm

    def elimination_phase(self):
        for i in range(self.num_arms):
            if self.estimated_arm_averages[i] + self.tolerance < max(self.estimated_arm_averages):
                self.is_arm_eliminated[i] = True

    def calculate_arm_averages_normal(self, arm):
        # for each arm, calculate the average reward
        num_rewards_received = len(self.received_rewards[arm])
        if num_rewards_received > 0:
            self.estimated_arm_averages[arm] = sum(self.received_rewards[arm]) / len(self.received_rewards[arm])
            self.is_arm_estimated[arm] = True

        # reset rewards buffers
        self.received_rewards[arm] = []

    def calculated_arm_averages_mixed(self, arm, bestarm):
        # for the recently played arms, calculate the new average
        self.estimated_arm_averages[arm] = (sum(self.received_rewards[arm]) + sum(self.received_rewards[bestarm])) / \
                                           (len(self.received_rewards[arm]) + len(self.received_rewards[bestarm])) - \
                                           self.estimated_arm_averages[bestarm]

    def get_best_arm(self):
            return np.argmax(np.asarray(self.estimated_arm_averages))

    def get_next_arm(self, arm):
        if arm >= self.num_arms:
            return -1

        if self.is_arm_eliminated[arm]:
            # The arm is eliminated, get next arm
            self.get_next_arm(arm + 1)
        else:
            return arm
        return arm
