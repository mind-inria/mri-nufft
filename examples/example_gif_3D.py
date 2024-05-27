"""
=======================
3D Trajectories display
=======================

A collection of 3D trajectories are generated and saved as a gif.

"""

# %%
import matplotlib.pyplot as plt
import mrinufft.trajectories.display as mtd
import mrinufft.trajectories.trajectory3D as mtt
import numpy as np
import joblib
from PIL import Image, ImageSequence
from mrinufft.trajectories.display import displayConfig
from tqdm.notebook import tqdm

# %% [markdown]
# # Options

# %%
Nc = 8 * 8
Ns = 200

nb_repetitions = 8

# %%
one_shot = 0
figsize = 5

nb_fps = 12
name_slow = 1

# %% [markdown]
# # Generation

# %%
# Initialize trajectory function
functions = [
    # 3D Cones
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, nb_zigzags=0)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, nb_zigzags=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, width=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns, width=x)[::-1]),
    ("3D Cones", lambda x: mtt.initialize_3D_cones(Nc, Ns)[::-1]),
    # FLORET
    (
        "FLORET",
        lambda x: mtt.initialize_3D_floret(
            Nc // nb_repetitions,
            Ns,
            nb_revolutions=1,
            max_angle=np.pi / 2,
            axes=[0],
            cone_tilt=None,
        ),
    ),
    (
        "FLORET",
        lambda x: mtt.initialize_3D_floret(
            x * Nc // nb_repetitions,
            Ns,
            nb_revolutions=x,
            max_angle=np.pi / 2,
            axes=[0],
            cone_tilt=None,
        ),
    ),
    (
        "FLORET",
        lambda x: mtt.initialize_3D_floret(
            Nc, Ns, nb_revolutions=nb_repetitions, max_angle=x, axes=[0], cone_tilt=None
        ),
    ),
    (
        "FLORET",
        lambda x: mtt.initialize_3D_floret(
            Nc, Ns, nb_revolutions=nb_repetitions, max_angle=x, axes=[0], cone_tilt=None
        ),
    ),
    # ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns, nb_cones=nb_repetitions, max_angle=np.pi / 2, nb_revolutions=x / 2, spiral=x, axes=[0], cone_tilt=None)),
    # ("FLORET", lambda x: mtt.initialize_3D_floret(Nc, Ns, nb_cones=nb_repetitions, max_angle=np.pi / 2, nb_revolutions=x / 2, spiral=x, axes=[0], cone_tilt=None)),
    (
        "FLORET",
        lambda x: mtt.initialize_3D_floret(
            Nc,
            Ns,
            nb_revolutions=nb_repetitions,
            max_angle=np.pi / 2,
            axes=[0],
            cone_tilt=None,
        ),
    ),
    # Seiffert spirals
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(Nc, Ns, curve_index=0),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(Nc, Ns, curve_index=x),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(
            Nc, Ns, curve_index=0.9, nb_revolutions=x
        ),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(
            Nc, Ns, curve_index=0.9, nb_revolutions=x
        ),
    ),
    (
        "Seiffert spiral / Yarnball",
        lambda x: mtt.initialize_3D_seiffert_spiral(
            Nc, Ns, curve_index=0.9, nb_revolutions=1
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
    # Annular shells
    # ("Annular shells", lambda x: mtt.initialize_3D_annular_shells(x * Nc // nb_repetitions, Ns, nb_shells=x, ring_tilt=0)[::-1]),
    # ("Annular shells", lambda x: mtt.initialize_3D_annular_shells(Nc, Ns, nb_shells=nb_repetitions, ring_tilt=x)[::-1]),
    # ("Annular shells", lambda x: mtt.initialize_3D_annular_shells(Nc, Ns, nb_shells=nb_repetitions, ring_tilt=x)[::-1]),
    # ("Annular shells", lambda x: mtt.initialize_3D_annular_shells(Nc, Ns, nb_shells=nb_repetitions, ring_tilt=np.pi / 2)[::-1]),
    # Seiffert shells
    # ("Seiffert shells", lambda x: mtt.initialize_3D_seiffert_shells(x * Nc // nb_repetitions, Ns, nb_shells=x, curve_index=0)[::-1]),
    # ("Seiffert shells", lambda x: mtt.initialize_3D_seiffert_shells(Nc, Ns, nb_shells=nb_repetitions, curve_index=x)[::-1]),
    # ("Seiffert shells", lambda x: mtt.initialize_3D_seiffert_shells(Nc, Ns, nb_shells=nb_repetitions, curve_index=x)[::-1]),
    # ("Seiffert shells", lambda x: mtt.initialize_3D_seiffert_shells(Nc, Ns, nb_shells=nb_repetitions, nb_revolutions=x)[::-1]),
    # ("Seiffert shells", lambda x: mtt.initialize_3D_seiffert_shells(Nc, Ns, nb_shells=nb_repetitions, nb_revolutions=x)[::-1]),
    # ("Seiffert shells", lambda x: mtt.initialize_3D_seiffert_shells(Nc, Ns, nb_shells=nb_repetitions, nb_revolutions=1)[::-1]),
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
    # ("Wave-CAIPI", lambda x: mtt.initialize_3D_wave_caipi(2 * Nc, Ns, packing=["triangular", "square", "circular", "triangular"][x])),
    ("Wave-CAIPI", lambda x: mtt.initialize_3D_wave_caipi(2 * Nc, Ns)),
]

# %%
# Initialize trajectory arguments
arguments = [
    # 3D Cones
    [None] * (nb_fps // 4),  # None
    np.linspace(0, np.sqrt(5), nb_fps) ** 2,  # nb_zigzags
    np.linspace(1, 2, nb_fps // 2),  # width
    np.linspace(2, 1, nb_fps // 2),  # width
    [None] * (nb_fps // 4),  # None
    # FLORET
    [None] * (nb_fps // 4),  # None
    np.around(np.linspace(1, nb_repetitions, nb_fps)).astype(int),  # nb_cones
    np.linspace(np.pi / 2, np.pi / 4, nb_fps // 2),  # max_angle
    np.linspace(np.pi / 4, np.pi / 2, nb_fps // 2),  # max_angle
    # np.linspace(2, 1, nb_fps // 2), # spiral & nb_revolutions
    # np.linspace(1, 2, nb_fps // 2), # spiral & nb_revolutions
    [None] * (nb_fps // 4),  # None
    # Seiffert spiral
    [None] * (nb_fps // 4),  # None
    np.linspace(0, 0.9, nb_fps),  # curve_index
    np.linspace(1, 3, nb_fps),  # nb_revolutions
    np.linspace(3, 1, nb_fps // 2),  # nb_revolutions
    [None] * (nb_fps // 4),  # None
    # Helical shells
    np.around(np.linspace(1, nb_repetitions, nb_fps)).astype(int),  # nb_cones
    np.linspace(1, 3, nb_fps),  # spiral_reduction
    [None] * (nb_fps // 4),  # None
    # Annular shells
    # np.around(np.linspace(1, nb_repetitions, nb_fps)).astype(int), # nb_cones
    # np.linspace(0, np.pi, nb_fps), # ring_tilt
    # np.linspace(np.pi, np.pi / 2, nb_fps // 2), # ring_tilt
    # [None] * (nb_fps // 4), # None
    # Seiffert shells
    # np.around(np.linspace(1, nb_repetitions, nb_fps)).astype(int), # nb_cones
    # np.linspace(0, 0.95, nb_fps), # curve_index
    # np.linspace(0.95, 0.5, nb_fps), # curve_index
    # np.linspace(1, 3, nb_fps), # nb_revolutions
    # np.linspace(3, 1, nb_fps // 2), # nb_revolutions
    # [None] * (nb_fps // 4), # None
    # Wave-CAIPI
    np.linspace(0, 2, nb_fps),  # nb_revolutions & width
    np.linspace(2, 1, nb_fps // 2),  # nb_revolutions & width
    # np.around(np.linspace(0, 3, nb_fps)).astype(int), # packing
    [None] * (nb_fps // 4),  # None
]

# %%


frame_setup = [
    (f, i, name, arg)
    for (name, f), args in list(zip(functions, arguments))
    for i, arg in enumerate(args)
]


def draw_frame(func, index, name, arg, save_dir="/tmp/"):
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
    hashed = joblib.hash((index, name, arg))
    filename = save_dir + f"{hashed}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    return filename


image_files = joblib.Parallel(n_jobs=1)(
    joblib.delayed(draw_frame)(*data) for data in frame_setup
)


# %%
# Make a GIF of all images.
imgs = [Image.open(img) for img in image_files]
imgs[0].save(
    "mrinufft_3D_traj.gif",
    save_all=True,
    append_images=imgs[1:],
    optimize=False,
    duration=200,
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
# don't raise errors from pytest. This will only be excecuted for the sphinx gallery stuff
try:
    final_dir = (
        Path(os.getcwd()).parent / "docs" / "generated" / "autoexamples" / "images"
    )
    shutil.copyfile("mrinufft_3D_traj.gif", final_dir / "mrinufft_3D_traj.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# .. image-sg:: /generated/autoexamples/images/mrinufft_3D_traj.gif
#    :alt: example density
#    :srcset: /generated/autoexamples/images/mrinufft_3D_traj.gif
#    :class: sphx-glr-single-img
