import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ODAAF

# Environment Initializations
# env = gym.make("AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2-v0")
env = gym.make("AdaBanditsBaseline-v0")

num_runs = 1

horizon = 250000

regret = [0 for _ in range(horizon)]
all_regrets = np.zeros((50, horizon))
rewards = [0 for _ in range(horizon)]

results1 = {}

env.reset()

Odaaf1 = ODAAF.OdaafExpectedDelay(env=env,
                                  horizon=horizon,
                                  num_arms=env.action_space.n,
                                  tolerance=5,
                                  expected_delay=7,
                                  bridge_period=25)

reward = 0
for z in tqdm(range(horizon)):

    action = Odaaf1.play(reward)
    _, reward, _, results1 = env.step(action)


for results in [results1]:
    sampled_run = results
    single_run_reward = sampled_run['reward']
    single_run_cumulated_reward_counter = 0
    single_run_cumulated_reward = []

    all_rewards = [results['reward'] for a in range(num_runs)]
    averaged_reward = np.sum(np.asarray(all_rewards), axis=0) / num_runs
    cumulated_average_reward = []
    cumulated_reward = 0

    all_expected_rewards = [results['expected_reward'] for a in range(num_runs)]
    averaged_all_expected_rewards = np.sum(np.asarray(all_expected_rewards), axis=0) / num_runs
    cumulated_expected_rewards_counter = 0
    cumulated_expected_rewards = []

    all_optimal = [results['optimal_mean'] for a in range(num_runs)]
    averaged_optimal = np.sum(np.asarray(all_optimal), axis=0) / num_runs
    cumulated_optimal = [averaged_optimal*n for n in range(horizon)]


    regret = averaged_optimal - averaged_reward
    cumulated_regret = 0
    regret_plot = []
    for i in range(len(regret)):
        cumulated_regret += regret[i]
        regret_plot.append(cumulated_regret)
        cumulated_reward += averaged_reward[i]
        cumulated_average_reward.append(cumulated_reward)
        single_run_cumulated_reward_counter += single_run_reward[i]
        single_run_cumulated_reward.append(single_run_cumulated_reward_counter)

        #expected rewards
        cumulated_expected_rewards_counter += averaged_all_expected_rewards[i]
        cumulated_expected_rewards.append(cumulated_expected_rewards_counter)

    # book def regret
    regret_final = []
    for i in range(horizon):
        regret_final.append(averaged_optimal * i - cumulated_expected_rewards[i])

    plot_steps = horizon

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.arange(plot_steps), cumulated_average_reward[:plot_steps], np.arange(plot_steps), cumulated_optimal[:plot_steps])
    ax1.set_title('Average Reward and Optimal Reward vs Time Step')
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(np.arange(plot_steps), regret_plot[:plot_steps])
    ax2.set_title('Average Regret vs Time Step')
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(np.arange(plot_steps), single_run_cumulated_reward[:plot_steps], np.arange(plot_steps), cumulated_optimal[:plot_steps])
    ax3.set_title('Single Run Reward and Optimal Reward vs Time Step')
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(np.arange(plot_steps), regret_final[:plot_steps])
    ax4.set_title("Final Reward vs Time Step")

    plt.show()