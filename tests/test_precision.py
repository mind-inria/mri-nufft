import numpy as np
import matplotlib.pyplot as plt
from mrinufft import get_operator, initialize_3D_from_2D_expansion, initialize_2D_radial
import matplotlib.pyplot as plt


def get_fourier_matrix(ktraj, im_size, im_rank, do_ifft=False):
    r = [
        np.linspace(-im_size[i] / 2, im_size[i] / 2 - 1, im_size[i])
        for i in range(im_rank)
    ]
    grid_r = np.reshape(np.meshgrid(*r, indexing="ij"), (im_rank, np.prod(im_size)))
    traj_grid = ktraj @ grid_r
    if do_ifft:
        A = np.exp(1j * traj_grid)
    else:
        A = np.exp(-1j * traj_grid)
    scale = np.sqrt(np.prod(im_size)) * np.power(np.sqrt(2), im_rank)
    A = A / scale
    return A


M = 1000
im_size = (20, 30)
# kspace_loc = np.random.uniform(-np.pi, np.pi, (M, len(im_size)))
kspace_loc = (
    initialize_3D_from_2D_expansion("radial", "rotations", 10, 10, 10)
    .reshape(-1, 3)
    .astype(np.float32)
)
kspace_loc = initialize_2D_radial(Nc=10, Ns=100).reshape(-1, 2).astype(np.float32)

A = get_fourier_matrix(kspace_loc, im_size, len(im_size))
img = np.random.random(im_size)
kspace_data = A @ img.flatten()
scale = np.sqrt(np.prod(im_size)) * np.power(np.sqrt(2), len(im_size)) / np.sqrt(M)
fourier_op = get_operator("finufft")(kspace_loc, im_size, n_coils=1)
kspace_data_cpu = fourier_op.op(img)
oper_cls = get_operator("cufinufft")
fourier_op = oper_cls(kspace_loc, im_size, n_coils=1, squeeze_dim=True)
kspace_data_cufi = fourier_op.op(np.asfortranarray(img))
plt.plot(np.abs(kspace_data_cpu), label="finufft")
plt.plot(np.abs(kspace_data), label="numpy")
plt.plot(np.abs(kspace_data_cufi), label="cufinufft")
plt.legend()
plt.show()
plt.savefig("test.png")
