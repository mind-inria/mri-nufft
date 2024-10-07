"""
========================
Animated 3D trajectories
========================

An animation to show 3D trajectory customization.

"""

import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tempfile as tmp
from PIL import Image, ImageSequence

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
    (f, i, name, arg)
    for (name, f), args in list(zip(functions, arguments))
    for i, arg in enumerate(args)
]


def draw_frame(func, index, name, arg):
    """Draw a single frame of the gif and save it to a tmp file."""
    trajectory = func(arg)
    # General configuration
    fig = plt.figure(figsize=(2 * figsize, figsize))
    subfigs = fig.subfigures(1, 2, wspace=0)

    # Trajectory display
    subfigs[0].suptitle(name, fontsize=displayConfig.fontsize, x=0.5, y=0.98)
    mtd.display_3D_trajectory(trajectory, subfigure=subfigs[0], one_shot=0)

    # Gradient display
    subfigs[1].suptitle("Gradients", fontsize=displayConfig.fontsize, x=0.5, y=0.98)
    mtd.display_gradients_simply(
        trajectory,
        shot_ids=[one_shot],
        figsize=figsize,
        subfigure=subfigs[1],
        uni_gradient="k",
        uni_signal="gray",
    )

    # Save figure
    filename = f"{tmp.NamedTemporaryFile().name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    return filename


image_files = joblib.Parallel(n_jobs=1)(
    joblib.delayed(draw_frame)(*data) for data in frame_setup
)


# Make a GIF of all images.
imgs = [Image.open(img) for img in image_files]
imgs[0].save(
    "mrinufft_3D_traj.gif",
    save_all=True,
    append_images=imgs[1:],
    optimize=False,
    duration=duration,
    loop=0,
)


# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path

for f in image_files:
    try:
        os.remove(f)
    except OSError:
        continue
# don't raise errors from pytest. This will only be executed for the sphinx gallery stuff
try:
    final_dir = (
        Path(os.getcwd()).parent / "docs" / "generated" / "autoexamples" / "images"
    )
    shutil.copyfile("mrinufft_3D_traj.gif", final_dir / "mrinufft_3D_traj.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# sphinx_gallery_thumbnail_path = 'generated/autoexamples/images/mrinufft_3D_traj.gif'

# %%
# .. image-sg:: /generated/autoexamples/images/mrinufft_3D_traj.gif
#    :alt: example density
#    :srcset: /generated/autoexamples/images/mrinufft_3D_traj.gif
#    :class: sphx-glr-single-img
