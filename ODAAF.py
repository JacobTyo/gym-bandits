import numpy as np


class Odaaf:
    def __init__(self, horizon, num_arms, tolerance, expected_delay, delay_upper_bound=None, delay_variance=None,
                 bridge_period=25):
        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.step = 0
        self.phase_count = 0
        self.phase_iteration_num = 0

        self.starting = True
        self.best_arm = 0
        self.restarting = True

        self.last_arm_pulled = 0
        self.last_arm_pulls = 0
        self.startingStep2NewPhase = True
        self.step2_remaining_iterations = 0
        self.step2_arm_iterations = 0

        # Hyperparameters
        self.tolerance = tolerance
        self.regret_upper_bound = []
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
        raise NotImplementedError('subclasses must override setnm()!')

    def play(self, reward, **kwargs):

        action = -1

        # for each iteration, determine which step to do
        if self.step == 0:

            # attribute reward
            if self.starting:
                self.starting = False
            else:
                self.phase1_pull_results[self.last_arm_pulled].append(reward)

            if self.restarting:
                self.restarting = False
                self.last_arm_pulled = 0

            # do phase 1
            action = self.step0()

        elif self.step == 1:  # Pretty sure this never happens

            # attribute the last action from phase 1, then eliminate arms
            self.phase1_pull_results[self.last_arm_pulled].append(reward)

            # do phase 2
            action = self.step1()

        elif self.step == 2:

            # do phase 3
            action = self.step2()

        # return the selected action
        return action

    def step0(self):
        # make sure we aren't starting on an arm that is already eliminated
        if self.eliminated_arms[self.last_arm_pulled] == 1:
            self.last_arm_pulled = self.get_next_arm(self.last_arm_pulled + 1)

        # This is the phase to play the arms
        required_pulls = (len(self.eliminated_arms) - sum(self.eliminated_arms)) * self.num_required_pulls_phase1

        # if we are still in this phase
        if self.phase_iteration_num < required_pulls:

            # if we still need to pull the previous arm
            if self.last_arm_pulls <= self.num_required_pulls_phase1:
                self.last_arm_pulls += 1
                self.phase_iteration_num += 1
                return self.last_arm_pulled

            # we need to pull the next arm
            else:

                # if we are out of arms, move to step 2
                if self.get_next_arm(self.last_arm_pulled + 1) < 0:
                    self.step = 1
                    self.last_arm_pulled = self.get_next_arm(0)
                    self.last_arm_pulls = 0
                    self.phase_iteration_num = 0
                    return self.step1()

                # if we are not out of arms, pull the next one
                else:
                    # we have more arms to pull for phase 1
                    self.last_arm_pulled = self.get_next_arm(self.last_arm_pulled + 1)
                    self.last_arm_pulls = 0
                    self.phase_iteration_num += 1
                    return self.last_arm_pulled
        # if we are no longer in step 1, then increment step counter and move to step 2
        else:
            self.step = 1
            return self.step1()

    def step1(self):
        # need to determine what arms to eliminate
        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                self.post_phase1_arm_averages[j] = sum(self.phase1_pull_results[j]) / len(self.phase1_pull_results[j])

        # Eliminate sub-optimal arms
        max_arm_avg = max(self.post_phase1_arm_averages)
        self.best_arm = np.argmax(np.asarray(self.post_phase1_arm_averages))

        for j in range(self.num_arms):
            if self.eliminated_arms[j] != 1:
                if self.post_phase1_arm_averages[j] + self.tolerance < max_arm_avg:
                    self.eliminated_arms[j] = 1

        # increment step
        self.step = 2
        return self.step2()

    def step2(self):

        # Determine how many iterations we need in the bridge period
        if self.startingStep2NewPhase:
            self.step2_remaining_iterations = self.bridge_period
            self.startingStep2NewPhase = False

        # Determine if we are still in the bridge period
        if self.step2_remaining_iterations <= 0:
            # start over
            self.last_arm_pulled = self.best_arm
            self.last_arm_pulls = 0
            self.phase_iteration_num = 0
            self.phase1_pull_results = [[] for _ in range(self.num_arms)]
            self.tolerance = self.tolerance / 2
            self.num_required_pulls_phase1 = self.setnm()
            self.startingStep2NewPhase = True
            self.step = 0
            self.restarting = True
            return self.best_arm

        # pull the best arm
        self.step2_remaining_iterations -= 1
        self.last_arm_pulled = self.best_arm
        return self.last_arm_pulled

    def get_next_arm(self, arm):
        if arm >= self.num_arms:
            return -1

        if self.eliminated_arms[arm] == 1:
            # The arm is eliminated, get next arm
            self.get_next_arm(arm + 1)
        else:
            return arm
        return arm


class OdaafExpectedDelay(Odaaf):
    def setnm(self):
        term1 = np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)))
        term2 = np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * (self.phase_count + 1) * self.expected_delay)
        term3 = (1.0 / (self.tolerance ** 2)) * (term1 + term2) ** 2
        term4 = (1.0 / (self.tolerance ** 2)) * (term1) ** 2
        term5 = (1.0 / (self.tolerance ** 2)) * (term2) ** 2
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * (self.phase_count + 1) * self.expected_delay)) ** 2
        return nm


class OdaafExpectedBoundedDelay(Odaaf):
    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      4 * self.tolerance * self.expected_delay)) ** 2
        d_min_comparison = ((1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      6 * self.tolerance * (self.phase_count + 1) * self.expected_delay)) ** 2
                            ) / (self.phase_count + 1)

        d_twidel = self.delay_upper_bound if self.delay_upper_bound <= d_min_comparison else d_min_comparison

        return nm if nm >= self.phase_count * d_twidel else self.phase_count * d_twidel


class OdaafBoundedDelayExpectationVariance(Odaaf):
    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      4 * self.tolerance * (self.expected_delay +
                                                                            2 * self.delay_variance))) ** 2

        return nm
