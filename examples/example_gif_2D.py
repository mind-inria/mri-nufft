"""
=======================
2D Trajectories display
=======================

A collection of 2D trajectories are generated and saved as a gif.

"""

# %%
import matplotlib.pyplot as plt
import mrinufft.trajectories.display as mtd
import mrinufft.trajectories.trajectory2D as mtt
import numpy as np

import joblib
from PIL import Image, ImageSequence


from mrinufft.trajectories.display import displayConfig


# %% [markdown]
# # Options

# %%
Nc = 16
Ns = 200

nb_repetitions = 8

# %%
one_shot = 0
figsize = 4

nb_fps = 12
name_slow = 1

# %% [markdown]
# # Generation

# %%
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
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, spiral=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, spiral=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=x)),
    ("Spiral", lambda x: mtt.initialize_2D_spiral(Nc, Ns, nb_revolutions=0)),
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
    ("Rosette", lambda x: mtt.initialize_2D_rosette(Nc, Ns, coprime_index=0)),
    ("Rosette", lambda x: mtt.initialize_2D_rosette(Nc, Ns, coprime_index=x)),
    ("Rosette", lambda x: mtt.initialize_2D_rosette(Nc, Ns, coprime_index=30)),
    # Waves
    ("Waves", lambda x: mtt.initialize_2D_waves(Nc, Ns, nb_zigzags=6 * x, width=x)),
    ("Waves", lambda x: mtt.initialize_2D_waves(Nc, Ns, nb_zigzags=6 * x, width=x)),
    ("Waves", lambda x: mtt.initialize_2D_waves(Nc, Ns, nb_zigzags=6, width=1)),
    # Lissajous
    ("Lissajous", lambda x: mtt.initialize_2D_lissajous(Nc, Ns, density=1)),
    ("Lissajous", lambda x: mtt.initialize_2D_lissajous(Nc, Ns, density=x)),
    ("Lissajous", lambda x: mtt.initialize_2D_lissajous(Nc, Ns, density=10)),
]

# %%
# Initialize trajectory arguments
arguments = [
    # Radial
    np.around(np.linspace(1, Nc, nb_fps)).astype(int),  # Nc
    np.linspace(2 * np.pi / Nc, np.pi / Nc, nb_fps // 2),  # tilt
    np.around(np.sin(np.linspace(0, 2 * np.pi, nb_fps // 2))).astype(bool),  # in_out
    np.linspace(np.pi / Nc, 2 * np.pi / Nc, nb_fps // 2),  # tilt
    [None] * (nb_fps // 4),  # None
    # Spiral
    np.linspace(0, np.sqrt(3), nb_fps // 2) ** 2,  # spiral
    np.linspace(3, 1, nb_fps // 3),  # spiral
    np.linspace(1, 2, nb_fps // 2),  # nb_revolutions
    np.linspace(2, 0, nb_fps),  # nb_revolutions
    [None] * (nb_fps // 4),  # None
    # Cones
    np.linspace(0, 5, nb_fps // 2),  # nb_zigzags
    np.linspace(1, 2, nb_fps // 4),  # width
    np.linspace(2, 0, nb_fps // 2),  # width
    [None] * (nb_fps // 4),  # None
    # Sinusoids
    np.linspace(0, 1, nb_fps // 2),  # width & nb_zigzags
    np.linspace(1, 0, nb_fps // 2),  # width & nb_zigzags
    [None] * (nb_fps // 4),  # None
    # Rings
    np.around(np.linspace(1, nb_repetitions, nb_fps)).astype(int),  # Nc & nb_rings
    np.around(np.linspace(nb_repetitions, Nc, nb_fps // 2)).astype(int),  # Nc
    [None] * (nb_fps // 2),  # None
    # Rosette
    [None] * (nb_fps // 2),  # None
    np.around(np.linspace(0, 30, nb_fps)).astype(int),  # coprime_index
    [None] * (nb_fps // 2),  # None
    # Waves
    np.linspace(0, 2, nb_fps // 2),  # width & nb_zigzags
    np.linspace(2, 1, nb_fps // 2),  # width & nb_zigzags
    [None] * (nb_fps // 4),  # None
    # Lissajous
    [None] * (nb_fps // 2),  # None
    np.linspace(1, 10, 2 * nb_fps),  # density
    [None] * (nb_fps // 2),  # None
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
    ax = mtd.display_2D_trajectory(
        trajectory, one_shot=one_shot, figsize=figsize, subfigure=subfigs[0]
    )
    ax.set_aspect("equal")

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
    "mrinufft_2D_traj.gif",
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
    final_dir = Path(__file__).parent / "docs" / "generated" / "autoexamples" / "images"
    shutil.copyfile("mrinufft_2D_traj.gif", final_dir / "mrinufft_2D_traj.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# .. image-sg:: /generated/autoexamples/images/mrinufft_2D_traj.gif
#    :alt: example density
#    :srcset: /generated/autoexamples/images/mrinufft_2D_traj.gif
#    :class: sphx-glr-single-img
