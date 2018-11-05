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

    # Results
    # print("mean reward: {}".format(np.mean(info["reward"])))
    max_mean = info["optimal_mean"]
    total_regret = (max_mean * horizon) - np.sum(info["expected_reward"])
    print("mean regret: {}".format(total_regret / horizon))
    plt.plot(np.cumsum(max_mean - np.asarray(info["expected_reward"])))
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
# plt.plot(total_rewards)
# plt.show()
# plt.figure()
# plt.plot(np.arange(10000), cumulative_reward_list[:10000], np.arange(10000), optimal_reward_list[:10000])
# plt.show()
# print(eliminated_arms)
# print(cumulative_reward)

if __name__ == "__main__":
    main()
