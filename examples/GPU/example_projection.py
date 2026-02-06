import numpy as np
import pylops
import pyproximal
from pyproximal.proximal import L2, EuclideanBall
from pyproximal.optimization.primal import GeneralizedProximalGradient
from mrinufft import Acquisition
import matplotlib.pyplot as plt

class PylopsProjectedGradient:
    def __init__(self, kinetic_op, actual_data, linear_constraint_op=None, v=None):
        """
        kinetic_ops: List of PyLops operators (e.g., [D1, D2])
        actual_data: The observed data (b)
        linear_constraint_op: The 'A' in Ax = v (pylops.Restriction or MatrixMult)
        v: The target values for the linear constraints
        """
        # Combine kinetic operators into one: A_kin
        self.A_kin = kinetic_op
        self.b = actual_data
        
        # Linear Constraints: Ax = v
        self.C_lin = linear_constraint_op
        self.v = v

    def __call__(self, x):
        return np.linalg.norm(self.project(x))**2

    def project(self, x):
        """
        Implements s = z + A^dagger * (v - Az)
        This projects z back onto the subspace where Ax = v.
        """
        z = self.b - self.A_kin.H * x 
        if self.C_lin is None:
            return z
        
        # Calculate residual of constraints: (v - Az)
        res = self.v - (self.C_lin @ z.flatten())
        
        # Project using the pseudo-inverse (A^dagger)
        # For simple restrictions, this is just a mask. 
        # For general matrices, we use the adjoint or a solver.
        projection_step = self.C_lin.H @ res 
        return (z.flatten() + projection_step).reshape(z.shape)

    def grad(self, x):
        """
        1. Get the 's_star' (Project x onto linear constraints)
        2. Apply kinetic operators
        3. Compute Gradient
        """
        # This is your s_star logic
        s_star = self.project(x)
        
        return - self.A_kin * s_star
        
        


acq = Acquisition(fov=(0.256,0.256, 0.003), img_size=(256,256, 1))
n_time_points = 512

# 2. Input Trajectory: 2D Random Walk (The "curve c" from the paper)
step_size = 0.01
c = np.cumsum(np.random.normal(0, step_size, (n_time_points, 2)), axis=0)
c *= acq.kmax[:c.shape[-1]] * c
c_flat = c.flatten()
# 3. Operators via PyLops [cite: 234, 235]
# Gradient operator (M)
D1 = pylops.FirstDerivative((n_time_points, 2), axis=0, sampling=1, kind="backward", edge=True)
A = pylops.VStack([D1, D1.T*D1])

f = PylopsProjectedGradient(kinetic_op=A, actual_data=c)


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

dt = 1
c1 = 1/2
c2 = 1/4
L = (2*c1/dt)**2+(4*c2/dt**2)**2
s_projected_flat = GeneralizedProximalGradient(
    [f], 
    [prox_grad, prox_slew], 
    x0=A*c, 
    niter=5000,
    tau=1/L, 
    epsg=[c1, c2],
    show=True,
)

c_proj = c - A.T * s_projected_flat
plt.plot(*c.T, label='Original trajectory')
plt.plot(*c_proj.T, label='Projected trajectory')
plt.legend()
