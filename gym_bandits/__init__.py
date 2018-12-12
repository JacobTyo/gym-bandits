from gym.envs.registration import register

from .bandit import BanditTenArmedRandomFixed
from .bandit import BanditTenArmedRandomRandom
from .bandit import BanditTenArmedGaussian
from .bandit import BanditTenArmedUniformDistributedReward
from .bandit import BanditTwoArmedDeterministicFixed
from .bandit import BanditTwoArmedHighHighFixed
from .bandit import BanditTwoArmedHighLowFixed
from .bandit import BanditTwoArmedLowLowFixed
from .anonymous_delayed_bandit import AnonymousDelayedBanditTwoArmedDeterministic
from .anonymous_delayed_bandit import AnonymousDelayedBanditTwoArmedStochasticDelay
from .anonymous_delayed_bandit import AnonymousDelayedBanditTwoArmStochasticReward
from .anonymous_delayed_bandit import AnonymousDelayedBanditTwoArmedStochasticDelayStochasticReward
from .anonymous_delayed_bandit import AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward1
from .anonymous_delayed_bandit import AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2
from .anonymous_delayed_bandit import Chirag
from .anonymous_delayed_bandit import AdaBanditsBaseline
from .anonymous_delayed_bandit import AdaBanditsBaseline_Optimistic
from .anonymous_delayed_bandit import AdaBanditsOutliers
from .anonymous_delayed_bandit import AdaBanditsOptimalOutlier
from .anonymous_delayed_bandit import AdaBanditsBaselineTrunc
from .anonymous_delayed_bandit import AdaBanditsLong


environments = [['BanditTenArmedRandomFixed', 'v0'],
                ['BanditTenArmedRandomRandom', 'v0'],
                ['BanditTenArmedGaussian', 'v0'],
                ['BanditTenArmedUniformDistributedReward', 'v0'],
                ['BanditTwoArmedDeterministicFixed', 'v0'],
                ['BanditTwoArmedHighHighFixed', 'v0'],
                ['BanditTwoArmedHighLowFixed', 'v0'],
                ['BanditTwoArmedLowLowFixed', 'v0'],
                ['AnonymousDelayedBanditTwoArmedDeterministic', 'v0'],
                ['AnonymousDelayedBanditTwoArmStochasticReward', 'v0'],
                ['AnonymousDelayedBanditTwoArmedStochasticDelay', 'v0'],
                ['AnonymousDelayedBanditTwoArmedStochasticDelayStochasticReward', 'v0'],
                ['AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward1', 'v0'],
                ['AnonymousDelayedBanditTenArmedStochasticDelayStochasticReward2', 'v0'],
                ['Chirag', 'v0'],
                ['AdaBanditsBaseline', 'v0'],
                ['AdaBanditsBaseline_Optimistic', 'v0'],
                ['AdaBanditsOutliers', 'v0'],
                ['AdaBanditsOptimalOutlier', 'v0'],
                ['AdaBanditsLong', 'v0'],
                ['AdaBanditsBaselineTrunc', 'v0']]


for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='gym_bandits:{}'.format(environment[0]),
        timestep_limit=1,
        nondeterministic=True,
    )
