import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from ucb import Ucb
from delayed_ucb import Delayed_Ucb


def main():

    parser = argparse.ArgumentParser(description='DAAF bandits experiment')
    parser.add_argument('--horizon', type=int, help='length of experiment')
    parser.add_argument('--repetitions', type=int, help='Number of times to run experiment')
    parser.add_argument('--ucb_delta', type=float, help='ucb error probability')
    parser.add_argument('--alg', type=str, choices=["ucb", "delayed_ucb"], help='bandit algorithm to run')
    parser.add_argument('--gym', type=str, choices=['BanditTenArmedRandomFixed',
                'BanditTenArmedRandomRandom',
                'BanditTenArmedGaussian',
                'BanditTenArmedUniformDistributedReward',
                'BanditTwoArmedDeterministicFixed',
                'BanditTwoArmedHighHighFixed',
                'BanditTwoArmedHighLowFixed',
                'BanditTwoArmedLowLowFixed',
                'AnonymousDelayedBanditTwoArmedDeterministic',
                'AnonymousDelayedBanditTwoArmStochasticReward',
                'AnonymousDelayedBanditTwoArmedStochasticDelay',
                'AnonymousDelayedBanditTwoArmedStochasticDelayStochasticReward',
                'AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward',
                'AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward1',
                'AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2'], help='bandit environment')
    args = parser.parse_args()
    horizon = args.horizon
    results = []

    for run in range(args.repetitions):
        # Setup
        env = gym.make(args.gym + "-v0")
        env.reset()

        # New algorithms go here
        agent = None
        if args.alg == "ucb":
            agent = Ucb(env.action_space.n, args)
        elif args.alg == "delayed_ucb":
            agent = Delayed_Ucb(env.action_space.n, args)

        # Experiment
        action = agent.play(None, non_anon_reward=[])
        info = None
        for step in range(horizon):
            observation, reward, done, info = env.step(action)
            action = agent.play(reward, non_anon_reward=info["non_anon_reward"])
        results.append(info)

    max_mean = results[0]["optimal_mean"]
    expected_rewards = np.asarray([info["expected_reward"] for info in results])
    mean_expected_rewards = np.mean(expected_rewards, axis=0)
    std_expected_rewards = np.std(expected_rewards, axis=0)
    # Results
    # print("mean reward: {}".format(np.mean(info["reward"])))
    total_regret = (max_mean * horizon) - np.sum(mean_expected_rewards)
    print("mean regret per step: {}".format(total_regret / horizon))
    cumregret = np.cumsum(max_mean - mean_expected_rewards)
    plt.plot(cumregret)
    plt.fill_between(np.arange(horizon), cumregret - std_expected_rewards, cumregret + std_expected_rewards, alpha=0.5)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.show()
    # plot rewards
    # plt.plot(rewards, label=args.alg)
    # plt.plot(np.arange(max_mean, max_mean * (horizon + 1), step=max_mean), label="optimal")
    # plt.xlabel("step")
    # plt.ylabel("reward")
    # plt.show()
    # plt.figure()

if __name__ == "__main__":
    main()
