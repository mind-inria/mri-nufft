"""
Generate trajectories for the benchmarks.
"""

import os

import numpy as np
from mrinufft.trajectories import (
    initialize_3D_from_2D_expansion,
    initialize_2D_radial,
    initialize_2D_spiral,
)
from mrinufft.trajectories.io import write_trajectory

SHAPE2D = (192, 192)
FOV2D = (0.192, 0.192)

SHAPE3D = (64, 64, 64)
FOV3D = (0.192, 0.192, 0.192)

##############
# 2D Radial  #
##############

radial_2d = initialize_2D_radial(16, 256)
write_trajectory(
    radial_2d,
    FOV=FOV2D,
    img_size=SHAPE2D,
    grad_filename="radial2d",
)

#####################
# 3D Stacked Radial #
#####################

stacked_radial_3d = initialize_3D_from_2D_expansion("radial", "stacks", 16, 256, 32)

write_trajectory(
    stacked_radial_3d,
    FOV=FOV3D,
    img_size=SHAPE3D,
    grad_filename="stacked_radial3d",
)
