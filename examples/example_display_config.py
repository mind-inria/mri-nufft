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
from mrinufft.trajectories import (
    initialize_2D_radial,
    initialize_3D_cones,
    initialize_3D_floret,
)

# Trajectory parameters
Nc = 120  # Number of shots
Ns = 500  # Number of samples per shot

# Display parameters
figure_size = 10  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = -5  # Highlight one shot in particular


# %%


def show_traj(traj, name, values, **kwargs):
    fig, axs = plt.subplots(
        1,
        len(values),
        figsize=(subfigure_size * len(values), subfigure_size),
        subplot_kw={"projection": "3d"},
    )
    for ax, val in zip(axs, values):
        with displayConfig(**{name: val}):
            display_3D_trajectory(traj, subfigure=ax, **kwargs)
            ax.set_title(f"{name}={val}", fontsize=2 * subfigure_size)
    plt.show()


# %%
#
# Trajectory displays
# ====================
# To show case the display parameters of trajectories, we will use the following trajectory
# The effect of trajectory parameter are explained in the :ref:`sphx_glr_generated_autoexamples_example_3D_trajectories.py` Example.

traj = initialize_3D_floret(Nc, Ns, nb_cones=6)[::-1]

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
# ``one_shot_color``
# ------------------
# The ``one_shot_color`` parameter allows to highlight one shot in particular.
with displayConfig(palette="viridis"):
    show_traj(
        traj, "one_shot_color", ["tab:blue", "tab:orange", "tab:green"], one_shot=-5
    )

# %%
# ``nb_colors``
# -------------
# The ``nb_colors`` parameter allows to change the number of colors used to display the shots.

show_traj(traj, "nb_colors", [1, 4, 10])

# %%
# Labels, titles and legends
# ==========================

# %%
# ``fontsize``
# ------------
# The ``fontsize`` parameter allows to change the fontsize of the labels /title

show_traj(traj, "fontsize", [12, 18, 24])

# %%
# ``pointsize``
# -------------
# To show the gradient constraint violation we can use the ``pointsize`` parameter
show_traj(traj, "pointsize", [0.5, 2, 4], show_constraints=True)

# %%
# ``gradient_point_color`` and ``slewrate_point_color``
# -----------------------------------------------------
# The ``gradient_point_color`` and ``slewrate_point_color`` parameters allows to change the color of the points
# that are violating the gradient or slewrate constraints.

show_traj(
    traj,
    "slewrate_point_color",
    ["tab:blue", "tab:orange", "tab:red"],
    show_constraints=True,
)


# %%
# Gradients profiles
# ==================

# %%

# %%

# %%

# %%
