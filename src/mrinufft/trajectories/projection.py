"""Functions to fit gradient constraints."""

import numpy as np
import numpy.linalg as nl
from functools import partial
from collections.abc import Callable
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from mrinufft._array_compat import get_array_module, with_numpy_cupy
from mrinufft._utils import _fill_doc, _progressbar
from mrinufft.trajectories.utils import Acquisition

PYPROXIMAL_AVAILABLE = True

try:
    from pylops import LinearOperator, FirstDerivative
    from pyproximal import ProxOperator
    import pylops
    import pyproximal
except ImportError:
    ProxOperator = object
    LinearOperator = object
    PYPROXIMAL_AVAILABLE = False


def parameterize_by_arc_length(
    trajectory: NDArray, order: int | None = None, eps: float = 1e-8
) -> NDArray:
    """Adjust the trajectory to have a uniform distribution over the arc-length.

    The trajectory is parametrized according to its arc length along a
    cubic-interpolated path and samples are repositioned to minimize
    the gradients amplitude. This solution is optimal with respect to
    gradients but can lead to excessive slew rates, and it will change
    the overall density.

    .. warning::
        - Slew rates are not minimized, and instead likely to increase
        - The sampling density will not be preserved

    Parameters
    ----------
    trajectory: NDArray
        A 2D or 3D trajectory of shape (Nc, Ns, Nd), with Nc the number of shots,
        Ns the number of samples per shot, and Nd the number of dimensions.
    order: int | None
        The order of the norm used to compute arc length, based on the convention from
        `numpy.linalg.norm`. Defaults to None (Euclidean norm).
    eps: float
        Convergence threshold for stopping the iterative refinement. Defaults to 1e-8.

    Returns
    -------
    NDArray: The reparameterized trajectory with the same shape as the input.
    """
    Nc, Ns, Nd = trajectory.shape
    new_trajectory = np.copy(trajectory)

    for i in range(Nc):
        projection = trajectory[i]

        # Ignore null gradients
        gradients = nl.norm(np.diff(projection, axis=0), ord=order, axis=-1)
        non_zero_ids = np.where(~np.isclose(gradients, 0))
        non_zero_time = np.linspace(0, 1, len(non_zero_ids[0]))
        arc_func = CubicSpline(non_zero_time, projection[non_zero_ids])

        # Setup initial conditions
        time = np.linspace(0, 1, Ns)
        projection = arc_func(time)
        old_projection = 0

        # Iterate to reduce arc length cubic approximation error
        while nl.norm(projection - old_projection) / nl.norm(projection) > eps:
            # Find mapping from arc length back to time
            arc_length = np.cumsum(
                nl.norm(np.diff(projection, axis=0), ord=order, axis=-1), axis=0
            )
            arc_length = np.concatenate([[0], arc_length])
            arc_length = arc_length / arc_length[-1]
            inv_arc_func = CubicSpline(arc_length, time)

            # Find times such that arc length is uniform
            time = inv_arc_func(np.linspace(0, 1, Ns))
            old_projection = np.copy(projection)
            projection = arc_func(time)
        new_trajectory[i] = projection
    return new_trajectory


#################################
# Projection on constraints set #
#################################


class GroupL2SoftThresholding(ProxOperator):
    r"""
    Group L2 Soft Thresholding (Shrinkage) Operator.

    This operator applies a soft-thresholding on the L2 norm of vectors
    grouped along the last dimension. It is effectively the proximal operator
    of the Group Lasso (L2,1) penalty.

    The logic follows: :math:`y = x * max(0, ||x|| - alpha) / ||x||`

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the input vector, expected to be (Nc, Ns, D).
    alphas : NDArray or float
        The threshold parameter(s). Can be a scalar or a vector of shape (D,).
        If a vector is provided, the thresholding is applied per-channel
        broadcasting against the joint L2 norm.
    """

    def __init__(self, shape: tuple[int, ...], alphas: NDArray):
        # We pass None as the function value evaluator for now,
        # or we could implement the L21 norm evaluation.
        super().__init__(None, False)
        self.shape = shape
        # Ensure alphas is the correct shape/type
        self.alphas = alphas

    @with_numpy_cupy
    def __call__(self, x: NDArray) -> bool:
        """
        Evaluate the regularization function value (L2,1 norm weighted by alphas).

        Note: This returns the penalty value, not a boolean check.
        """
        xp = get_array_module(x)
        norms = xp.linalg.norm(x.reshape(self.shape), axis=-1)
        return xp.sum(norms * np.mean(self.alphas))

    @with_numpy_cupy
    def prox(self, x: NDArray, tau: float, eps: float = 1e-10) -> NDArray:
        """
        Apply the Group L2 Soft Thresholding.

        y = x * ( ||x||_2 - (alphas * tau) )_+ / ||x||_2
        """
        xp = get_array_module(x)
        x_mat = x.reshape(self.shape)
        thresholds = self.alphas * tau
        norm2_vec = xp.linalg.norm(x_mat, axis=-1)
        denom = xp.maximum(norm2_vec, eps)[..., None]
        norm_minus_alpha = norm2_vec[..., None] - thresholds
        numer = xp.maximum(0, norm_minus_alpha)
        scaling_factor = numer / denom
        y = x_mat * scaling_factor
        return y.flatten()


_proj_docs = dict(
    proj_ref="""
References
----------

.. [Proj] N. Chauffert, P. Weiss, J. Kahn and P. Ciuciu, "A Projection Algorithm for
       Gradient Waveforms Design in Magnetic Resonance Imaging," in
       IEEE Transactions on Medical Imaging, vol. 35, no. 9, pp. 2026-2039, Sept. 2016,
       doi: 10.1109/TMI.2016.2544251.
""",
)


@_fill_doc(_proj_docs)
def linear_projection(
    x: NDArray,
    target: NDArray,
    A: LinearOperator | None = None,
    mask: NDArray | tuple | None = None,
) -> NDArray:
    r"""Implement the projection on linear constraints set given by Eq 10 in_[Proj].

    Parameters
    ----------
    x: NDArray
        The input vector to project
    target: NDArray
        The target values for the linear constraints.
    A: LinearOperator, optional
        The linear operator defining the constraints.
    mask: NDArray, optional
        A boolean mask indicating the indices of the input vector to be projected
        onto v. If provided, the projection is performed by directly setting
        z[mask] = v.

    Notes
    -----
    The linear constraint set defined by the linear operator A and the target v
    is given by:

    .. math::
            \mathcal{C} = \{ x \in \mathbb{R}^n : Ax = v \}

    The projection of an input vector z onto this set is thus:

    .. math::
            s = z + A^\dagger (v - Az)

    Alternatively, the constraint set can also be provided by a mask:

    .. math::
            \mathcal{C} = \{ x \in \mathbb{R}^n : x[mask] = v \}

    ${proj_ref}
    """
    if A is not None and mask is not None:
        raise ValueError("Provide either a linear operator A or a mask")
    elif A is None and mask is None:
        return x
    elif mask is not None:
        x[mask] = target
        return x
    else:
        return A.div(target - A * x)


@_fill_doc(_proj_docs)
class GradientLinearProjection:
    r"""
    Implements the gradient of F(q1, q2) given by Eq 11 in_[Proj].

    The gradient is given by:
    .. math::
            \nabla F(q) = - A s^*

    where s^* is the projection of the primal variable :math:`z = c - M^H q`
    onto the linear constraint set defined by A and v.

    Parameters
    ----------
    M: LinearOperator
        The kinetic operator M.
    c: NDArray
        The initial trajectory c.
    linear_projector: LinearProjection, optional
        An instance of the LinearProjection class to perform the projection
        onto the constraint set. If not provided, the projection will be
        performed without any constraints (i.e., s^* = z).

    ${proj_ref}
    """

    def __init__(
        self,
        initial_trajectory: NDArray,
        kinetic_op: LinearOperator,
        linear_projector: Callable | None = None,
    ):
        self.M = kinetic_op
        self.c = initial_trajectory
        self.linear_projector = linear_projector

    def get_primal_variables(self, q: NDArray) -> NDArray:
        """Compute the primal variables z = c - M^H q."""
        return self.c - (self.M.H * q)

    def grad(self, q: NDArray) -> NDArray:
        """
        Compute the gradient of the objective function w.r.t q.

        grad F(q) = - M * s_star
        """
        z = self.get_primal_variables(q)
        s_star = self.linear_projector(z) if self.linear_projector is not None else z
        return -(self.M * s_star).flatten()


@_fill_doc(_proj_docs)
@with_numpy_cupy
def project_trajectory(
    trajectory: NDArray,
    acq: Acquisition | None = None,
    safety_factor: float = 0.99,
    max_iter: int = 1000,
    TE_pos: float | None = -1,
    linear_projector: Callable | None = None,
    verbose: int = 1,
) -> NDArray:
    """
    Projects the trajectory onto hardware constraint set.

    This function implements ALgorithm 1 in _[Proj].

    Parameters
    ----------
    trajectory: NDArray
        The input trajectory to be projected, of shape (Nc, Ns, Nd).
    acq: Acquisition, optional
        An instance of the Acquisition class containing the gradient constraints.
        If not provided, the projection will be performed without any constraints.
    safety_factor: float
        An extra safety factor to ensure the projected trajectory is within hardware
        limits. Defaults to 0.99 (i.e., 1% margin).
    max_iter: int
        The maximum number of iterations for the projection algorithm. Defaults to 1000.
    TE_pos: float | None | -1
        Specifies the constrained position of the k-space center in the
        trajectory. If a float is provided, it should be in the range [0, 1] and
        indicates the relative position of the k-space center along the
        trajectory (e.g., 0.5 for the middle). If None, no such constraint is
        applied. Defaults to -1, which detect the position of the original
        k-space center and set the constraint accordingly.
    linear_projector: LinearProjection, optional
        An instance of the LinearProjection class to perform the projection onto the
        constraint set. This is available for advanced users who want to specify
        custom linear constraints.
        If not provided, the projection is performed according to `in_out`
    verbose: int, optional
        The verbosity level. If 0, no progress bar is shown. If 1, a progress bar is
        displayed. If 2 we show the iteration level cost function.

    Returns
    -------
    NDArray: The projected trajectory, of the same shape as the input.

    ${proj_ref}
    """
    acq = acq or Acquisition.default
    if not PYPROXIMAL_AVAILABLE:
        raise ImportError(
            "pyproximal is required for trajectory projection. "
            "Please install it to use this function."
        )
    if trajectory.ndim == 2:
        trajectory = trajectory[None]
        xp = get_array_module(trajectory)
        Nc, Ns, Nd = trajectory.shape
        D1 = FirstDerivative(
            (Nc, Ns, Nd),
            axis=1,
            sampling=1,
            kind="backward",
            edge=True,
            dtype=trajectory.dtype,
        )
        c1 = 1 / 2
        c2 = 1 / 4
        # Define the weighted first and second derivative operators
    M = pylops.VStack([c1 * D1, c2 * D1 * D1], dtype=trajectory.dtype)
    lipchitz_constant = 2  #  (2 * c1) ** 2 + (4 * c2) ** 2
    maxstep = (
        safety_factor
        * acq.gamma
        * acq.raster_time
        / xp.asarray(acq.kmax[:Nd], dtype=trajectory.dtype)
        / 2
    )
    prox_grad = GroupL2SoftThresholding((Nc, Ns, Nd), c1 * maxstep * acq.gmax)
    prox_slew = GroupL2SoftThresholding(
        (Nc, Ns, Nd), c2 * maxstep * acq.smax * acq.raster_time
    )
    prox = pyproximal.VStack([prox_grad, prox_slew], nn=[Nc * Ns * Nd] * 2)

    if TE_pos == -1:
        # detect the position of the original k-space center and
        # set the constraint accordingly
        TE_pos = np.argmin(np.linalg.norm(trajectory, axis=-1), axis=1) / Ns

    if linear_projector is None and TE_pos is not None:
        linear_projector_ = partial(
            linear_projection,
            target=xp.zeros((Nc, Nd), dtype=trajectory.dtype),
            mask=(slice(None), int(Ns * TE_pos), slice(None)),
        )
    else:
        linear_projector_ = linear_projector
        f = GradientLinearProjection(
            initial_trajectory=trajectory,
            kinetic_op=M,
            linear_projector=linear_projector_,
        )
        progressbar = _progressbar(verbose == 1, max_iter)
        q_star = pyproximal.optimization.primal.ProximalGradient(
            f,
            prox,
            x0=M * trajectory,
            niter=max_iter,
            acceleration="fista",
            tau=1 / lipchitz_constant,
            show=verbose > 1,
            callback=lambda x: progressbar.update(1),
        )
        s_s = f.get_primal_variables(q_star)
    return linear_projector_(s_s) if linear_projector_ is not None else s_s
