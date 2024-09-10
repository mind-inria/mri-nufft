import os
from pathlib import Path
import shutil
import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import time
import joblib
from PIL import Image

from fastmri.models import Unet
from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial

def plot_state(axs, mri_2D, img, recon, loss=None, save_name=None):
    """
    Plot the original MRI image, the pre-training image, the reconstructed image,
    and the loss curve (if provided). Saves the plot if a filename is provided.

    Parameters
    ----------
    axs (numpy array): Array of matplotlib axes to plot on.
    mri_2D (torch.Tensor): Original MRI image.
    img (torch.Tensor): Image before training.
    recon (torch.Tensor): Reconstructed image after training.
    loss (list, optional): List of loss values to plot. Defaults to None.
    save_name (str, optional): Filename to save the plot. Defaults to None.
    """
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Objectif image")
    axs[1].imshow(np.abs(img[0][0].detach().numpy()), cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Image pre trainning")
    axs[2].imshow(np.abs(recon[0][0].detach().cpu().numpy()), cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Reconstruction")
    if loss is not None:
        axs[3].plot(loss)
        axs[3].set_title("Loss")
        axs[3].grid("on")
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

class Model(torch.nn.Module):
    def __init__(self, initial_trajectory):
        super(Model, self).__init__()
        self.operator = get_operator("gpunufft", wrt_data=True)(
            initial_trajectory,
            shape=(256, 256),
            density=True,
            squeeze_dims=False,
        )
        self.unet = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4)

    def forward(self, kspace):
        image = self.operator.adj_op(kspace)
        # plt.imshow((image[0][0].detach().float()), cmap="gray")
        # plt.show()
        recon = self.unet(image.float())
        return recon

# Initialize the U-Net model for MRI reconstruction
init_traj = initialize_2D_radial(64, 256).reshape(-1, 2).astype(np.float32)
model = Model(init_traj)

# Initialize optimizer and learning rate scheduler
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=40, gamma=0.01
)

# Load and preprocess MRI data
mri_2D = torch.Tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))[
    None
]
mri_2D = mri_2D / torch.linalg.norm(mri_2D)
kspace_mri_2D = model.operator.op(mri_2D)

# Perform a forward pass to obtain the initial reconstruction before training
model.eval()
old_recon = model(kspace_mri_2D)

#  Train the model
losses = [] # Store the loss values and create an animation
image_files = [] # Store the images to create a gif
model.train()

with tqdm(range(100), unit="steps") as tqdms:
    for i in tqdms:
        out = model(kspace_mri_2D) # Forward pass

        loss = torch.nn.functional.l1_loss(out, mri_2D[None]) # Compute loss
        tqdms.set_postfix({"loss": loss.item()}) # Update progress bar
        losses.append(loss.item()) # Store loss value

        optimizer.zero_grad() # Zero gradients
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        scheduler.step() # Update learning rate

        # Generate images for gif
        hashed = joblib.hash((i, "learn_traj", time.time()))
        filename = "/tmp/" + f"{hashed}.png"
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plot_state(
            axs,
            mri_2D,
            old_recon,
            out,
            losses,
            save_name=filename,
        )
        image_files.append(filename)


# Make a GIF of all images.
imgs = [Image.open(img) for img in image_files]
imgs[0].save(
    "mrinufft_learn_unet.gif",
    save_all=True,
    append_images=imgs[1:],
    optimize=False,
    duration=2,
    loop=0,
)

# Cleanup
for f in image_files:
    try:
        os.remove(f)
    except OSError:
        continue
# don't raise errors from pytest.
# This will only be executed for the sphinx gallery stuff

try:
    final_dir = (
        Path(os.getcwd()).parent.parent
        / "docs"
        / "generated"
        / "autoexamples"
        / "GPU"
        / "images"
    )
    shutil.copyfile("mrinufft_learn_unet.gif", final_dir / "mrinufft_learn_unet.gif")
except FileNotFoundError:
    pass

# Perform a final evaluation of the model after training
model.eval()
kspace_mri_2D = model.operator.op(mri_2D)
new_recon = model(kspace_mri_2D)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plot_state(axs, mri_2D, old_recon, new_recon, losses)
plt.show()
