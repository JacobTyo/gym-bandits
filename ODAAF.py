import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Environment Initializations
env = gym.make("AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2-v0")

num_runs = 50

horizon = 250000  # total number of rounds

regret = [0 for _ in range(horizon)]

all_regrets = np.zeros((50, horizon))

for z in tqdm(range(num_runs)):

    env.reset()

    # Algorithm Initializations
    tolarance = 5
    num_arms = 10
    active_arms_number = num_arms  # ten arm bandit for this example
    arm_pulls = [0 for _ in range(num_arms)]
    phase1_pull_results = [[] for _ in range(num_arms)]

    eliminated_arms = [0 for _ in range(num_arms)]

    post_phase1_arm_averages = [0 for _ in range(num_arms)]

    total_rewards = [0 for _ in range(horizon)]

    cumulative_reward = 0
    cumulative_reward_list = [0 for _ in range(horizon)]

    optimal_reward = 5.5
    optimal_reward_list = [optimal_reward*a for a in range(horizon)]
    optimal_reward_list_each_step = [optimal_reward for a in range(horizon)]

    bridge_period = 25

    overall_iterations = 0
    upper_bound_on_delay = 15
    num_required_pulls_phase1 = 100
    # num_required_pulls_phase1 = np.log(horizon * tolarance ** 2) / (tolarance ** 2) + \
    #                             overall_iterations * upper_bound_on_delay / tolarance

    previous_action = 0

    current_step = 1
    i = 0
    # TODO: this will break down if all rewards are negative (because of how arms are eliminated)
    while i < horizon:
        # Step 1: Play arms

        # print("Starting Phase 1")
        for j in range(num_arms):
            if eliminated_arms[j] == 1:
                continue  # arm has been eliminated, don't play it
            starting_i = i
            while i - starting_i <= num_required_pulls_phase1 and i < horizon:
                observation, reward, done, returned_hist = env.step(j)
                phase1_pull_results[j].append(reward)
                total_rewards[j] = reward
                cumulative_reward += reward
                cumulative_reward_list[i] = cumulative_reward
                i += 1

        # print("Advancing to Phase 2")
        current_step = 2

        # print("Starting Phase 2")
        # compute the averages
        for j in range(num_arms):
            if eliminated_arms[j] != 1:
                post_phase1_arm_averages[j] = sum(phase1_pull_results[j]) / len(phase1_pull_results[j])

        # Eliminate sub-optimal arms  TODO Verify this (tolerance seems bad)
        max_arm_avg = max(post_phase1_arm_averages)
        # print("Max arm avg: ", max_arm_avg)
        # print("Arm Averages: ", post_phase1_arm_averages)
        # print("++++++++++++")
        for j in range(num_arms):
            if eliminated_arms[j] != 1:
                if post_phase1_arm_averages[j] + tolarance < max_arm_avg:
                    eliminated_arms[j] = 1

        # decrease tolarance
        tolarance = tolarance / 2.0
        # num_required_pulls_phase1 = np.log(horizon * tolarance ** 2) / (tolarance ** 2) + \
        #                             overall_iterations * upper_bound_on_delay / np.clip(tolarance, 0.00005, 99999)

        # print("Entering Bridge Period")
        # Just run through some steps to prevent rewards from seeping between phases

        for j in range(num_arms):
            if eliminated_arms[j] != 1:
                # play the arm, otherwise it has been eliminated
                for k in range(bridge_period):
                    if i < horizon:
                        _, reward, _, _ = env.step(j)
                        total_rewards[i] = reward
                        cumulative_reward += reward
                        cumulative_reward_list[i] = cumulative_reward
                        i += 1
                    else:
                        break


        # print("Finished Bridge Period")

        # reset averages and rewards
        post_phase1_arm_averages = [0 for _ in range(num_arms)]
        phase1_pull_results = [[] for _ in range(num_arms)]
        arm_pulls = [0 for _ in range(num_arms)]

        # print(eliminated_arms)
        # print("--------------")

        # TODO: Is the reward supposed to be bounded [0, 1]?

    # plt.plot(total_rewards)
    # plt.show()
    # plt.figure()
    # plt.plot(np.arange(10000), cumulative_reward_list[:10000], np.arange(10000), optimal_reward_list[:10000])
    # plt.show()
    # print(eliminated_arms)
    # print(cumulative_reward)
    regret = (np.asarray(total_rewards) - np.asarray(optimal_reward_list_each_step)).tolist()
    all_regrets[z, :] = np.asarray(regret)


# plt.show()

plt.plot(np.arange(10000), np.mean(all_regrets, axis=0)[:10000])
plt.show()