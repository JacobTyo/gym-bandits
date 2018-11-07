import numpy as np


class Hedger:
    def __init__(self, horizon, num_arms, tolerance, expected_delay, delay_upper_bound, delay_variance,
                 bridge_period=25):

        self.k = 100
        self.n = 20

        # test with high expected delay 0 use that as k then set n to be k/10 or something 

        # Class variables
        self.horizon = horizon
        self.iteration = 0
        self.step = 0
        self.phase_count = 0
        self.phase_iteration_num = 0

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

        self.eliminated_arms = [0 for _ in range(num_arms)]
        self.phase1_pull_results = [[] for _ in range(num_arms)]
        self.total_rewards = [0 for _ in range(horizon)]
        self.cumulative_reward = 0
        self.cumulative_reward_list = [0 for _ in range(horizon)]
        self.post_phase1_arm_averages = [0 for _ in range(num_arms)]

        self.best_arm_rewards_so_far = [0 for _ in range(horizon)]

        self.num_required_pulls_phase1 = self.setnm()

    def setnm(self):
        nm = (1.0 / (self.tolerance ** 2)) * (np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2))) +
                                              np.sqrt(2 * np.log(self.horizon * (self.tolerance ** 2)) + (8.0 / 3.0) *
                                                      self.tolerance * np.log(self.horizon * (self.tolerance ** 2)) +
                                                      4 * self.tolerance * (self.expected_delay +
                                                                            2 * self.delay_variance))) ** 2

        return nm

    def play(self, reward):

        action = -1

        # for each iteration, determine which step to do
        if self.step == 0:

            # attribute reward
            self.phase1_pull_results[self.last_arm_pulled].append(reward)

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
        if self.phase_iteration_num <= required_pulls:

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
                    self.last_arm_pulled = 0
                    self.last_arm_pulls = 0
                    self.phase_iteration_num += 1
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
            self.step2_remaining_iterations = self.bridge_period * \
                                              (len(self.eliminated_arms) - sum(self.eliminated_arms))
            self.startingStep2NewPhase = False
            self.step2_iterations = 0

        # Determine if we are still in the bridge period
        if self.step2_remaining_iterations <= 0:
            # start over
            self.last_arm_pulled = 0
            self.last_arm_pulls = 0
            self.phase_iteration_num = 0
            self.phase1_pull_results = [[] for _ in range(self.num_arms)]
            return self.step0()

        # continue pulling the last arm
        if self.step2_arm_iterations < self.bridge_period:
            self.last_arm_pulls += 1
            self.phase_iteration_num += 1
            return self.last_arm_pulled

        # get the next arm
        else:
            self.last_arm_pulls = 0
            self.last_arm_pulled = self.get_next_arm(self.last_arm_pulled + 1)
            self.phase_iteration_num += 1
            return self.last_arm_pulled

    def get_next_arm(self, arm):
        if arm >= self.num_arms:
            return -1

        if self.eliminated_arms[arm] == 1:
            # The arm is eliminated, get next arm
            self.get_next_arm(arm + 1)
        else:
            return arm