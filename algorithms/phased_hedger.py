import numpy as np


class PhasedHedger:
    def __init__(self, horizon, num_arms, tolerance, expected_delay, bridge_period=25):

        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.iterations_this_phase = 0
        self.step = 0

        self.best_arm = 0

        self.received_rewards = [[] for _ in range(num_arms)]
        self.selected_arm = 0
        self.previously_selected_arm = 0
        self.phase_count = 1
        self.num_arms = num_arms
        self.first_time_1a = True
        self.estimate_this_arm = 0

        self.best_arm_found = True

        self.estimated_arm_averages = [0 for _ in range(num_arms)]
        self.is_arm_estimated = [False for _ in range(num_arms)]
        self.is_arm_eliminated = [False for _ in range(num_arms)]
        self.rewards_from_best_this_round = []
        self.hedging = False

        self.best_arm_average_so_far = 0

        self.previous_arm_averages = [0 for _ in range(num_arms)]

        self.hedging_iterations_this_arm = 0

        self.bridge_iterations = 0

        self.step1_iterations = -1

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
        # TODO: This can be negative, need to re-evaluate how it is calculated
        term2 = np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                        self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                        6 * self.tolerance * (self.phase_count + 1) * self.expected_delay)
        term5 = (1.0 / (self.tolerance ** 2)) * term2 ** 2

        if self.nm - int(term5) <= 0:
            return int(self.nm / 2)
        return self.nm - int(term5)

    def play(self, reward, **kwargs):

        # record reward from previous action
        if self.iteration > 0:
            self.received_rewards[self.selected_arm].append(reward)

        if self.hedging:
            self.rewards_from_best_this_round.append(reward)

        if sum(self.is_arm_eliminated) >= 7:
            if self.best_arm_found:
                self.best_arm_found = False
                print("Best arm found")
                print(self.is_arm_eliminated)
                print(self.previous_arm_averages)
            self.selected_arm = [i for i in range(len(self.is_arm_eliminated)) if not self.is_arm_eliminated[i]][0]
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

        # what to do this step:
        # if this is the first time in this step, get the next arm, set it as the arm to be estimated, and then return
        # now run for little_nm times for this arm, and then pull the best arm nm-little_nm times
        # estimate that arm, repeat

        self.step1_iterations += 1

        if self.step1_iterations == 0:
            self.hedging = False
            self.selected_arm = self.get_next_arm(self.selected_arm + 1)
            self.estimate_this_arm = self.selected_arm

        elif self.step1_iterations == self.little_nm:
            self.hedging = True
            self.selected_arm = self.get_best_arm()

        elif self.step1_iterations == self.nm:
            self.hedging = False
            # we have completed an arm, move to the next one
            self.calculated_arm_averages_mixed(self.estimate_this_arm)
            self.selected_arm = self.get_next_arm(self.estimate_this_arm + 1)
            self.estimate_this_arm = self.selected_arm

        elif self.step1_iterations - (self.estimate_this_arm - 2) * self.nm == self.little_nm:
            # just pull the best arm for the rest of the phase
            self.hedging = True
            self.selected_arm = self.get_best_arm()

        elif self.step1_iterations - (self.num_arms - 2) * self.nm == 0:
            self.hedging = False
            # end of this step, estimate, bridge period, then restart
            self.calculated_arm_averages_mixed(self.estimate_this_arm)
            self.selected_arm = self.get_best_arm()
            self.step += 1
            self.step1_iterations = -1

        elif self.step1_iterations - (self.estimate_this_arm - 1) * self.nm == 0:
            self.hedging = False
            self.calculated_arm_averages_mixed(self.estimate_this_arm)
            self.selected_arm = self.get_next_arm(self.estimate_this_arm + 1)
            self.estimate_this_arm = self.selected_arm

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

            # self.received_rewards = [[] for _ in range(self.num_arms)]
            self.previous_arm_averages = self.estimated_arm_averages
            self.estimated_arm_averages = [0 for _ in range(self.num_arms)]

        return self.selected_arm

    def elimination_phase(self):
        for i in range(self.num_arms):
            if self.estimated_arm_averages[i] + self.tolerance < max(self.estimated_arm_averages):
                self.is_arm_eliminated[i] = True
            # decrease tolerance
            self.tolerance = self.tolerance / 2

    def calculate_arm_averages_normal(self, arm):
        # for each arm, calculate the average reward
        num_rewards_received = len(self.received_rewards[arm])
        if num_rewards_received > 0:
            self.estimated_arm_averages[arm] = sum(self.received_rewards[arm]) / len(self.received_rewards[arm])
            self.is_arm_estimated[arm] = True

        # reset rewards buffers
        self.received_rewards[arm] = []

    def calculated_arm_averages_mixed(self, arm):
        # for the recently played arms, calculate the new average
        self.estimated_arm_averages[arm] = (sum(self.received_rewards[arm]) + sum(self.rewards_from_best_this_round)) \
                                           / (len(self.received_rewards[arm]) +
                                              len(self.rewards_from_best_this_round)) - \
                                           self.best_arm_average_so_far

    def get_best_arm(self):
        self.best_arm = np.argmax(np.asarray(self.estimated_arm_averages))
        self.best_arm_average_so_far = np.max(np.asarray(self.estimated_arm_averages))
        return self.best_arm

    def get_next_arm(self, arm):
        if arm >= self.num_arms:
            return -1

        if self.is_arm_eliminated[arm]:
            # The arm is eliminated, get next arm
            self.get_next_arm(arm + 1)
        else:
            return arm
        return arm
