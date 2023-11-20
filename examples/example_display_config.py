"""
================================
Trajectory display configuration
================================

The look of the display trajectories can be tweaked by using :py:class:`displayConfig`

You can tune these parameters to your own taste and needs.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mrinufft import displayConfig, display_2D_trajectory, display_3D_trajectory
from mrinufft.trajectories import initialize_2D_radial, initialize_3D_cones

# Trajectory parameters
Nc = 120  # Number of shots
Ns = 500  # Number of samples per shot

# Display parameters
figure_size = 10  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = -5  # Highlight one shot in particular


# %%


def show_traj(traj, name, values):
    fig, axs = plt.subplots(
        1,
        len(values),
        figsize=(subfigure_size * len(values), subfigure_size),
        subplot_kw={"projection": "3d"},
    )
    for ax, val in zip(axs, values):
        with displayConfig(**{name: val}):
            display_3D_trajectory(traj, subfigure=ax)
            ax.set_title(f"{name}={val}", fontsize=4 * subfigure_size)
    plt.show()


# %%
#
# Trajectory displays
# ====================
# To show case the display parameters of trajectories, we will use the following trajectory
# The effect of trajectory parameter are explained in the :ref:`sphinx_glr_example_3D_trajectories.py`

traj = initialize_3D_cones(Nc, Ns)

# %%
# ``linewidth``
# -------------
# The linewidth of the shot can be updated to have more or less empty space in the plot.
show_traj(traj, "linewidth", [0.5, 2, 4])

# %%
# ``palette``
# -----------
# The ``palette`` parameter allows to change the color of the shots.
show_traj(traj, "palette", ["tab10", "magma", "jet"])

# %%
# Labels, titles and legends
# ==========================


# %%
# Gradients profiles
# ==================
