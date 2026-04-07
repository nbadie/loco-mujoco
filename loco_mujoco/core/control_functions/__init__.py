from .base import ControlFunction
from .default import DefaultControl
from .pd import PDControl
from .skeleton_muscle import SkeletonMuscleControl

# register all control functions
DefaultControl.register()
PDControl.register()
SkeletonMuscleControl.register()
