"""
======================
Trajectory constraints
======================

A collection of methods to make trajectories fit hardware constraints.

"""

# %%
# Hereafter we illustrate different methods to reduce the gradient
# strengths and slew rates required for the trajectory to match the
# hardware constraints of MRI machines. A summary table is available
# below.
#

# %%
# .. list-table:: Constraint fitting methods
#    :header-rows: 1
#
#    * -
#      - Gradient strength
#      - Slew rate
#      - Path preserved
#      - Density preserved
#    * - Arc-length parameterization
#      - Yes
#      - No
#      - Yes
#      - No
#

# Internal
import mrinufft as mn
from mrinufft.trajectories.utils import Acquisition, compute_gradients_and_slew_rates
from utils import show_trajectory_full

# External
import numpy as np


# %%
# Script options
# ==============
# These options are used in the examples below as default values for all trajectories.

# Acquisition parameters
acq = Acquisition.default

# %%

# Trajectory parameters
Nc = 16  # Number of shots
Ns = 3000  # Number of samples per shot
in_out = False  # Choose between in-out or center-out trajectories
nb_zigzags = 5  # Number of zigzags for base trajectories

# %%

# Display parameters
figure_size = 10  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = 2 * Nc // 3  # Highlight one shot in particular
sample_freq = 60  # Frequency of samples to display in the trajectory plots

# %%
# We will be using a cone trajectory to showcase the different methods as
# it switches several times between high gradients and slew rates.

original_trajectory = mn.initialize_2D_cones(
    Nc, Ns, in_out=in_out, nb_zigzags=nb_zigzags
)

# %%
# Arc-length parameterization
# ===========================
# Arc-length parameterization is the simplest method to reduce the gradient
# strength as it resamples the trajectory to have a constant distance between
# samples. This is technically the lowest gradient strength achievable while
# preserving the path of the trajectory, but it does not preserve the k-space
# density and can lead to high slew rates as shown below.

show_trajectory_full(original_trajectory, one_shot, subfigure_size, sample_freq)

grads, slews = compute_gradients_and_slew_rates(original_trajectory, acq)
grad_max, slew_max = np.max(grads), np.max(slews)
print(f"Max gradient: {grad_max:.3f} T/m, Max slew rate: {slew_max:.3f} T/m/ms")

# %%
#

from mrinufft.trajectories.projection import parameterize_by_arc_length

projected_trajectory = parameterize_by_arc_length(original_trajectory)

# %%

show_trajectory_full(projected_trajectory, one_shot, subfigure_size, sample_freq)

grads, slews = compute_gradients_and_slew_rates(projected_trajectory, acq)
grad_max, slew_max = np.max(grads), np.max(slews)
print(f"Max gradient: {grad_max:.3f} T/m, Max slew rate: {slew_max:.3f} T/m/ms")
