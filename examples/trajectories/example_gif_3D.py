"""
========================
Animated 3D trajectories
========================

An animation to show 3D trajectory customization.

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import numpy as np

import mrinufft.trajectories.display as mtd
import mrinufft.trajectories.trajectory3D as mtt
from mrinufft.trajectories.display import displayConfig


# %%
# Script options
# ==============

Nc = 8 * 8
Ns = 200
nb_repetitions = 8

one_shot = 0
figsize = 5

nb_frames = 3
duration = 150  # seconds


# %%
# Trajectory generation
# =====================

# Initialize trajectory function
functions = [
    # 3D Cones
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, width=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, width=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, nb_zigzags=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, nb_zigzags=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns)[::-1]),
    # FLORET
    ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns, nb_revolutions=x)),
    ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns, nb_revolutions=x)),
    ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns, max_angle=x)),
    ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns, max_angle=x)),
    ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns)),
    # Seiffert spirals
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(Nc, Ns, curve_index=x),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(
            Nc, Ns, curve_index=0.7, nb_revolutions=x
        ),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(
            Nc, Ns, curve_index=0.7, nb_revolutions=x
        ),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(
            Nc, Ns, curve_index=0.7, nb_revolutions=1
        ),
    ),
    # Helical shells
    (
        "Concentric shells",
        lambda x: mtt.initialize_3D_helical_shells(
            x * Nc // nb_repetitions, Ns, nb_shells=x
        )[::-1],
    ),
    (
        "Concentric shells",
        lambda x: mtt.initialize_3D_helical_shells(
            Nc, Ns, nb_shells=nb_repetitions, spiral_reduction=x
        )[::-1],
    ),
    (
        "Concentric shells",
        lambda x: mtt.initialize_3D_helical_shells(
            Nc, Ns, nb_shells=nb_repetitions, spiral_reduction=3
        )[::-1],
    ),
    # Wave-CAIPI
    (
        "Wave-CAIPI",
        lambda x: mtt.initialize_3D_wave_caipi(
            2 * Nc, Ns, nb_revolutions=5 * x, width=x
        ),
    ),
    (
        "Wave-CAIPI",
        lambda x: mtt.initialize_3D_wave_caipi(
            2 * Nc, Ns, nb_revolutions=5 * x, width=x
        ),
    ),
    ("Wave-CAIPI", lambda x: mtt.initialize_3D_wave_caipi(2 * Nc, Ns)),
]

# Initialize trajectory arguments
arguments = [
    # 3D Cones
    np.linspace(0, 2, 4 * nb_frames),  # width
    np.linspace(2, 1, 2 * nb_frames),  # width
    np.linspace(np.sqrt(5), 1, 4 * nb_frames) ** 2,  # nb_zigzags
    np.linspace(1, np.sqrt(5), 2 * nb_frames) ** 2,  # nb_zigzags
    [None] * nb_frames,  # None
    # FLORET
    np.linspace(1, 3, 4 * nb_frames),  # nb_revolutions
    np.linspace(3, 1, 2 * nb_frames),  # nb_revolutions
    np.linspace(np.pi / 2, np.pi / 4, 4 * nb_frames),  # max_angle
    np.linspace(np.pi / 4, np.pi / 2, 2 * nb_frames),  # max_angle
    [None] * nb_frames,  # None
    # Seiffert spiral
    np.linspace(0, 0.7, 4 * nb_frames),  # curve_index
    np.linspace(1, 2, 4 * nb_frames),  # nb_revolutions
    np.linspace(2, 1, 2 * nb_frames),  # nb_revolutions
    [None] * nb_frames,  # None
    # Helical shells
    np.around(np.linspace(1, nb_repetitions, 4 * nb_frames)).astype(int),  # nb_cones
    np.linspace(1, 3, 4 * nb_frames),  # spiral_reduction
    [None] * nb_frames,  # None
    # Wave-CAIPI
    np.linspace(0, 2, 4 * nb_frames),  # nb_revolutions & width
    np.linspace(2, 1, 2 * nb_frames),  # nb_revolutions & width
    [None] * nb_frames,  # None
]


# %%
# Animation rendering
# ===================

frame_setup = [
    (f, name, arg)
    for (name, f), args in list(zip(functions, arguments))
    for arg in args
]


fig = plt.figure(figsize=(2 * figsize, figsize))
gs = GridSpec(4, 2)
ksp_ax = fig.add_subplot(gs[:, 0], projection="3d")
axs_grad = [fig.add_subplot(gs[i, 1]) for i in range(4)]


def plot_frame(frame_data):
    func, name, arg = frame_data
    ksp_ax.clear()
    [ax.clear() for ax in axs_grad]
    trajectory = func(arg)
    ksp_ax.set_title(name, fontsize=displayConfig.fontsize)
    mtd.display_3D_trajectory(trajectory, one_shot=one_shot, subfigure=ksp_ax)
    ksp_ax.set_aspect("equal")
    mtd.display_gradients_simply(
        trajectory,
        shot_ids=[one_shot],
        subfigure=axs_grad,
        uni_gradient="k",
        uni_signal="gray",
    )


ani = animation.FuncAnimation(fig, plot_frame, frame_setup, interval=50, repeat=False)

plt.show()

# sphinx_gallery_thumbnail_number = 1
