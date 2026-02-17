import numpy as np
from mrinufft import Acquisition
import matplotlib.pyplot as plt
from mrinufft.trajectories.projection import project_trajectory
from mrinufft.trajectories.trajectory2D import initialize_2D_spiral


acq = Acquisition(fov=(0.256, 0.256, 0.003), img_size=(256, 256, 1))

step_size = 0.01
c = np.cumsum(np.random.normal(0, step_size, (Ns, 2)), axis=0)
c *= acq.kmax[: c.shape[-1]] * c

c_proj = project_trajectory(c, acq, max_iter=50000, linear_projector="no_proj")
plt.plot(*c.T, label="Original trajectory")
plt.plot(*c_proj.T, label="Projected trajectory")
plt.legend()
