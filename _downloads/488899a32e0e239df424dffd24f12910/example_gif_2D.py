"""
========================
Animated 2D trajectories
========================

An animation to show 2D trajectory customization.

"""

import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

import mrinufft.trajectories.display as mtd
import mrinufft.trajectories.trajectory2D as mtt
from mrinufft.trajectories.display import displayConfig

# %%
# Script options
# ==============

Nc = 16
Ns = 200
nb_repetitions = 8

one_shot = 0
figsize = 4

nb_frames = 3
duration = 150  # seconds


# %%
# Trajectory generation
# =====================

# Initialize trajectory function
functions = [
    # Radial
    ("Radial", lambda x: mtt.initialize_2D_radial(x, Ns)),
    ("Radial", lambda x: mtt.initialize_2D_radial(Nc, Ns, tilt=x)),
    (
        "Radial",
        lambda x: mtt.initialize_2D_radial(Nc, Ns, tilt=(1 + x) * np.pi / Nc, in_out=x),
    ),
    ("Radial", lambda x: mtt.initialize_2D_radial(Nc, Ns, tilt=x)),
    ("Radial", lambda x: mtt.initialize_2D_radial(Nc, Ns, tilt="uniform")),
    # Spiral
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, spiral=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, spiral=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=1e-5)),
    # Cones
    ("Cones", lambda x: mtt.initialize_2D_cones(Nc, Ns, nb_zigzags=x)),
    ("Cones", lambda x: mtt.initialize_2D_cones(Nc, Ns, width=x)),
    ("Cones", lambda x: mtt.initialize_2D_cones(Nc, Ns, width=x)),
    ("Cones", lambda x: mtt.initialize_2D_cones(Nc, Ns, width=0)),
    # Sinusoids
    (
        "Sinusoids",
        lambda x: mtt.initialize_2D_sinusoide(Nc, Ns, nb_zigzags=3 * x, width=x),
    ),
    (
        "Sinusoids",
        lambda x: mtt.initialize_2D_sinusoide(Nc, Ns, nb_zigzags=3 * x, width=x),
    ),
    ("Sinusoids", lambda x: mtt.initialize_2D_sinusoide(Nc, Ns, nb_zigzags=0, width=0)),
    # Rings
    ("Rings", lambda x: mtt.initialize_2D_rings(x, Ns, nb_rings=x)[::-1]),
    ("Rings", lambda x: mtt.initialize_2D_rings(x, Ns, nb_rings=nb_repetitions)[::-1]),
    ("Rings", lambda x: mtt.initialize_2D_rings(Nc, Ns, nb_rings=nb_repetitions)[::-1]),
    # Rosette
    ("Rosette", lambda x: mtt.initialize_2D_rosette(Nc, Ns, coprime_index=x)),
    ("Rosette", lambda x: mtt.initialize_2D_rosette(Nc, Ns, coprime_index=30)),
    # Waves
    ("Waves", lambda x: mtt.initialize_2D_waves(Nc, Ns, nb_zigzags=6 * x, width=x)),
    ("Waves", lambda x: mtt.initialize_2D_waves(Nc, Ns, nb_zigzags=6 * x, width=x)),
    ("Waves", lambda x: mtt.initialize_2D_waves(Nc, Ns, nb_zigzags=6, width=1)),
    # Lissajous
    ("Lissajous", lambda x: mtt.initialize_2D_lissajous(Nc, Ns, density=x)),
    ("Lissajous", lambda x: mtt.initialize_2D_lissajous(Nc, Ns, density=10)),
]

# Initialize trajectory arguments
arguments = [
    # Radial
    np.around(np.linspace(1, Nc, 4 * nb_frames)).astype(int),  # Nc
    np.linspace(2 * np.pi / Nc, np.pi / Nc, 2 * nb_frames),  # tilt
    np.around(np.sin(np.linspace(0, 2 * np.pi, 2 * nb_frames))).astype(bool),  # in_out
    np.linspace(np.pi / Nc, 2 * np.pi / Nc, 2 * nb_frames),  # tilt
    [None] * nb_frames,  # None
    # Spiral
    np.linspace(1e-5, 1, 2 * nb_frames),  # nb_revolutions
    np.linspace(1, np.sqrt(1 / 3), 2 * nb_frames) ** 2,  # spiral
    np.linspace(1 / 3, 1, 2 * nb_frames),  # spiral
    np.linspace(1, 3, 2 * nb_frames),  # nb_revolutions
    np.linspace(3, 1e-5, 4 * nb_frames),  # nb_revolutions
    [None] * nb_frames,  # None
    # Cones
    np.linspace(0, 5, 2 * nb_frames),  # nb_zigzags
    np.linspace(1, 2, nb_frames),  # width
    np.linspace(2, 0, 2 * nb_frames),  # width
    [None] * nb_frames,  # None
    # Sinusoids
    np.linspace(0, 1, 2 * nb_frames),  # width & nb_zigzags
    np.linspace(1, 0, 2 * nb_frames),  # width & nb_zigzags
    [None] * nb_frames,  # None
    # Rings
    np.around(np.linspace(1, nb_repetitions, 4 * nb_frames)).astype(
        int
    ),  # Nc & nb_rings
    np.around(np.linspace(nb_repetitions, Nc, 2 * nb_frames)).astype(int),  # Nc
    [None] * nb_frames,  # None
    # Rosette
    np.around(np.linspace(0, np.sqrt(30), 6 * nb_frames) ** 2).astype(
        int
    ),  # coprime_index
    [None] * nb_frames,  # None
    # Waves
    np.linspace(0, 2, 4 * nb_frames),  # width & nb_zigzags
    np.linspace(2, 1, 2 * nb_frames),  # width & nb_zigzags
    [None] * nb_frames,  # None
    # Lissajous
    np.linspace(1, 10, 6 * nb_frames),  # density
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
gs = GridSpec(3, 2)
ksp_ax = fig.add_subplot(gs[:, 0])
axs_grad = [fig.add_subplot(gs[i, 1]) for i in range(3)]


def plot_frame(frame_data):
    func, name, arg = frame_data
    ksp_ax.clear()
    [ax.clear() for ax in axs_grad]
    trajectory = func(arg)
    ksp_ax.set_title(name, fontsize=displayConfig.fontsize)
    mtd.display_2D_trajectory(trajectory, one_shot=one_shot, subfigure=ksp_ax)
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
# sphinx_gallery_thumbnail_number = -1
