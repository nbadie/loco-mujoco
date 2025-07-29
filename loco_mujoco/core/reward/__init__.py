from .base import Reward
from .default import NoReward, TargetVelocityGoalReward, TargetXVelocityReward, LocomotionReward
from .trajectory_based import TargetVelocityTrajReward, MimicReward
from .utils import *
from .default_cross import TargetVelocityGoalRewardCross
from .trajectory_based_cross import MimicRewardCross
from .trajectory_based_vel import MimicRewardVel


# register all rewards
NoReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
TargetVelocityTrajReward.register()
MimicReward.register()
MimicRewardCross.register()
LocomotionReward.register()
TargetVelocityGoalRewardCross.register()
MimicRewardVel.register()