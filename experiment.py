import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from ucb import Ucb
from delayed_ucb import Delayed_Ucb
import ODAAF


def run(args, alg):

    horizon = args.horizon
    results = []
    env = gym.make(args.gym + "-v0")

    for run in range(args.repetitions):
        env.reset()

        # New algorithms go here
        agent = None
        if alg == "ucb":
            agent = Ucb(env.action_space.n, args)
        elif alg == "delayed_ucb":
            agent = Delayed_Ucb(env.action_space.n, args)

        # Experiment
        action = agent.play(None, non_anon_reward=[])
        info = None
        for step in range(horizon):
            observation, reward, done, info = env.step(action)
            action = agent.play(reward, non_anon_reward=info["non_anon_reward"])
        results.append(info)

    expected_rewards = np.asarray([info["expected_reward"] for info in results])
    mean_expected_rewards = np.mean(expected_rewards, axis=0)
    std_expected_rewards = np.std(expected_rewards, axis=0)
    return {"mean": mean_expected_rewards, "std": std_expected_rewards}

def main():
    parser = argparse.ArgumentParser(description='DAAF bandits experiment')
    parser.add_argument('--horizon', type=int, help='length of experiment')
    parser.add_argument('--repetitions', type=int, help='Number of times to run experiment')
    parser.add_argument('--ucb_delta', type=float, help='ucb error probability')
    #parser.add_argument('--alg', type=str, choices=["ucb", "delayed_ucb"], help='bandit algorithm to run')
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
                'AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2'
                'AdaBanditsBaseline'], help='bandit environment')
    args = parser.parse_args()
    horizon = args.horizon

    algs = ["ucb", "delayed_ucb"]
    results = {}
    for alg in algs:
        results[alg] = run(args, alg)

    tempenv = gym.make(args.gym + "-v0")
    tempenv.reset()
    max_mean = tempenv.step(0)[-1]["optimal_mean"]
    # Results

    ################### TEMP hack for Jake's odaaf ####################
    env = gym.make(args.gym + "-v0")
    temp = []
    for _ in range(args.repetitions):
        env.reset()
        odaaf = ODAAF.OdaafExpectedDelay(env=env,
                                          horizon=horizon,
                                          num_arms=env.action_space.n,
                                          tolerance=.5,
                                          expected_delay=5,
                                          bridge_period=25)
        temp.append(odaaf.play())

    temp = np.asarray([info["expected_reward"] for info in temp])
    results["odaaf_ed"] = {"mean": np.mean(temp, axis=0), "std": np.std(temp, axis=0)}

    temp = []
    for _ in range(args.repetitions):
        env.reset()

        odaaf = ODAAF.OdaafExpectedBoundedDelay(env=env,
                                                horizon=horizon,
                                                num_arms=env.action_space.n,
                                                tolerance=.5,
                                                expected_delay=5,
                                                delay_upper_bound=30,
                                                bridge_period=25)
        temp.append(odaaf.play())

    temp = np.asarray([info["expected_reward"] for info in temp])
    results["odaaf_ebd"] = {"mean": np.mean(temp, axis=0), "std": np.std(temp, axis=0)}
    ######### End hack ##################

    temp = []
    for _ in range(args.repetitions):
        env.reset()
        odaaf = ODAAF.OdaafBoundedDelayExpectationVariance(env=env,
                                                            horizon=horizon,
                                                            num_arms=env.action_space.n,
                                                            tolerance=.5,
                                                            expected_delay=5,
                                                            delay_upper_bound=30,
                                                            delay_variance=5,
                                                            bridge_period=25)
        temp.append(odaaf.play())

    temp = np.asarray([info["expected_reward"] for info in temp])
    results["odaaf_bdev"] = {"mean": np.mean(temp, axis=0), "std": np.std(temp, axis=0)}

    plt.figure(figsize=(15, 7.5))
    for alg in algs + ["odaaf_ed",
                       "odaaf_ebd",
                       "odaaf_bdev"]:
        mean = results[alg]["mean"]
        std = results[alg]["std"]
        total_regret = (max_mean * horizon) - np.sum(mean)
        print("{} mean regret per step: {}".format(alg, total_regret / horizon))
        cumregret = np.cumsum(max_mean - mean)
        plt.semilogy(cumregret, label=alg)
        plt.fill_between(np.arange(horizon), cumregret - std, cumregret + std, alpha=0.5)

    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.legend(loc='upper left')
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
