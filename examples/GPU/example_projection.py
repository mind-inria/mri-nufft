import numpy as np
import pylops
import pyproximal
from pyproximal.proximal import L2, EuclideanBall
from pyproximal.optimization.primal import GeneralizedProximalGradient
from mrinufft import Acquisition
import matplotlib.pyplot as plt


acq = Acquisition(fov=(0.256,0.256, 0.003), img_size=(256,256, 1))
n_time_points = 512

# 2. Input Trajectory: 2D Random Walk (The "curve c" from the paper)
step_size = 0.01
c = np.cumsum(np.random.normal(0, step_size, (n_time_points, 2)), axis=0)
c *= acq.kmax[:c.shape[-1]] * c
c_flat = c.flatten()
# 3. Operators via PyLops [cite: 234, 235]
# Gradient operator (M)
D1 = pylops.FirstDerivative((n_time_points, 2), axis=0, sampling=1, kind="forward", edge=True)

# Vertically stack them to solve the AttributeError
A = pylops.VStack([D1, D1.T*D1])

f = L2(b=c, Op=A.T)

prox_grad = EuclideanBall(center=0, radius=acq.gamma*acq.hardware.gmax*acq.raster_time)
prox_slew = EuclideanBall(center=0, radius=acq.gamma*acq.hardware.smax*acq.raster_time**2)

def first_derivative(data, dwell_time=1):
    update_value = np.diff(data, axis=1)
    first_derivative = np.zeros(data.shape, dtype=update_value.dtype)
    first_derivative[:, 1:, :] = update_value
    first_derivative = first_derivative / dwell_time
    return first_derivative


def first_derivative_transpose(data, dwell_time=1):
    update_value = - np.diff(data[:, 1:, :], axis=1)
    first_derivative = np.zeros(data.shape, dtype=update_value.dtype)
    first_derivative[:, 0, :] = -data[:, 1, :]
    first_derivative[:, 1:-1, :] = update_value
    first_derivative[:, -1, :] = data[:, -1, :]
    first_derivative = first_derivative / dwell_time
    return first_derivative

def second_derivative(data, dwell_time=1):
    first_d = first_derivative(data, dwell_time)
    second_d = first_derivative_transpose(first_d, dwell_time)
    return second_d



s_projected_flat = GeneralizedProximalGradient(
    [f], 
    [prox_grad, prox_slew], 
    x0=A*c, 
    niter=5000,
    tau=1, 
    show=True,
)

c_proj = A.T * s_projected_flat
plt.plot(*c.T, label='Original trajectory')
plt.plot(*c_proj.T, label='Projected trajectory')
plt.legend()
