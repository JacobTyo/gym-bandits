import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
import argparse
# import seaborn as sns
from multiprocessing import Pool
from algorithms.ucb import Ucb
from algorithms.delayed_ucb import Delayed_Ucb
from algorithms import ODAAF, Hedger, phased_hedger, random_actions, phased_ucb
import functools
import random
import os
import pickle


def run(args, alg):

    horizon = args.horizon
    results = []
    env = gym.make(args.gym + "-v0")

    print("Launching " + alg)
    for run in range(args.repetitions):  # tqdm(range(args.repetitions)):
        env.reset()

        # New algorithms go here
        agent = None
        if alg == "ucb":
            agent = Ucb(env.action_space.n, args)
        elif alg == "phased_ucb":
            agent = phased_ucb.PhasedUcb(env.action_space.n, args)
        elif alg == "delayed_ucb":
            agent = Delayed_Ucb(env.action_space.n, args)
        elif alg == "odaaf_ed":
            agent = ODAAF.OdaafExpectedDelay(horizon=horizon,
                                             num_arms=env.action_space.n,
                                             tolerance=args.tolerance,
                                             expected_delay=args.expected_delay,
                                             bridge_period=args.bridge_period)
        elif alg == "odaaf_ebd":
            agent = ODAAF.OdaafExpectedBoundedDelay(horizon=horizon,
                                                    num_arms=env.action_space.n,
                                                    tolerance=args.tolerance,
                                                    expected_delay=args.expected_delay,
                                                    delay_upper_bound=args.delay_upper_bound,
                                                    bridge_period=args.bridge_period)
        elif alg == "odaaf_bdev":
            agent = ODAAF.OdaafBoundedDelayExpectationVariance(horizon=horizon,
                                                               num_arms=env.action_space.n,
                                                               tolerance=args.tolerance,
                                                               expected_delay=args.expected_delay,
                                                               delay_upper_bound=args.delay_upper_bound,
                                                               delay_variance=args.expected_variance,
                                                               bridge_period=args.bridge_period)
        elif alg == "hedger":
            agent = Hedger.Hedger(horizon=horizon,
                                  num_arms=env.action_space.n,
                                  tolerance=args.tolerance,
                                  expected_delay=args.expected_delay,
                                  bridge_period=args.bridge_period)
        elif alg == "hedger_phased":
            agent = phased_hedger.PhasedHedger(horizon=horizon,
                                               num_arms=env.action_space.n,
                                               tolerance=args.tolerance,
                                               expected_delay=args.expected_delay,
                                               bridge_period=args.bridge_period)
        elif alg == "random_actions":
            agent = random_actions.randomActions(env.action_space.n)

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
    # parser.add_argument('--ucb_delta', type=float, help='ucb error probability', default=0.01)
    parser.add_argument('--bridge_period', type=int, help='ucb error probability', default=1500)
    parser.add_argument('--expected_delay', type=int, help='ucb error probability', default=1000)
    parser.add_argument('--delay_upper_bound', type=int, help='ucb error probability', default=2000)
    parser.add_argument('--expected_variance', type=int, help='ucb error probability', default=1000)
    parser.add_argument('--tolerance', type=float, help='ucb error probability', default=0.5)
    parser.add_argument('--save_name', type=str, help='file name to save results', default="")

    # parser.add_argument('--alg', type=str, choices=["ucb", "delayed_ucb"], help='bandit algorithm to run')
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
                                                    'AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2',
                                                    'AdaBanditsBaseline',
                                                    'AdaBanditsOutliers',
                                                    'AdaBanditsBaseline_Optimistic',
                                                    'AdaBanditsBaselineTrunc',
                                                    'AdaBanditsLong'], help='bandit environment')
    args = parser.parse_args()
    horizon = args.horizon

    pool = Pool(4)

    algs = ["phased_ucb", "delayed_ucb", "odaaf_ed", "odaaf_ebd", "odaaf_bdev", "hedger_phased", "ucb"]
    output = pool.map(functools.partial(run, (args)), algs)

    # output = {}
    # for alg in algs:
    #     output[alg] = run(args, alg)

    i = 0
    results = {}
    for alg in algs:
        results[alg] = output[i]
        i += 1

    tempenv = gym.make(args.gym + "-v0")
    tempenv.reset()
    max_mean = tempenv.step(0)[-1]["optimal_mean"]
    # Results

    cumregret = {}
    plt.figure(figsize=(15, 7.5))
    for alg in algs:
        mean = results[alg]["mean"]
        std = results[alg]["std"]
        total_regret = (max_mean * horizon) - np.sum(mean)
        print("{} mean regret per step: {}".format(alg, total_regret / horizon))
        cumregret[alg] = np.cumsum(max_mean - mean)
        plt.semilogy(cumregret[alg], label=alg)
        # plt.fill_between(np.arange(horizon), cumregret - std, cumregret + std, alpha=0.5)

    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.yscale("linear")
    plt.legend(loc='upper left')

    if args.save_name:
        if not os.path.isdir("../results"):
            os.mkdir("../results")
        with open("../results/" + args.save_name + '.pickle', 'wb') as handle:
            pickle.dump(cumregret, handle)
        plt.savefig("../results/" + args.save_name + '.png')

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
