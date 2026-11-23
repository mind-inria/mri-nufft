"""Proximal operators for the trajectory projection.

They subclass ``pyproximal.ProxOperator``, so this module imports pyproximal, and
pylops through it, which are slow to import: `mrinufft.trajectories.projection`
loads it on first use.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pyproximal import ProxOperator

from mrinufft._array_compat import get_array_module, with_numpy_cupy
from mrinufft._utils import _fill_doc

from .projection import _proj_docs

if TYPE_CHECKING:
    from pylops import LinearOperator


class _GroupL2SoftThresholding(ProxOperator):
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

    # Its public location, for reprs, pickles and the docs.
    __module__ = "mrinufft.trajectories.projection"

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


@_fill_doc(_proj_docs)
class _GradientLinearProjection(ProxOperator):
    r"""
    Implements the gradient of F(q1, q2) given by Eq 11 in_[Proj].

    The gradient is given by:
    .. math::

       \nabla F(q) = - A s^*

    where :math:`s^*` is the projection of the primal variable :math:`z = c - M^H q`
    onto the linear constraint set defined by :math:`A` and :math:`v`.

    Parameters
    ----------
    M: LinearOperator
        The kinetic operator M.
    c: NDArray
        The initial trajectory c.
    linear_projector: LinearProjection, optional
        An instance of the LinearProjection class to perform the projection
        onto the constraint set. If not provided, the projection will be
        performed without any constraints (i.e., :math:`s^* = z`).

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

    def __call__(self, x: NDArray) -> float:
        """Evaluate the objective function."""
        z = self.get_primal_variables(x)
        s_star = self.linear_projector(z) if self.linear_projector is not None else z
        xp = get_array_module(s_star)
        return 0.5 * xp.linalg.norm(s_star) ** 2

    def grad(self, x: NDArray) -> NDArray:
        """
        Compute the gradient of the objective function w.r.t q.

        grad F(q) = - M * s_star
        """
        z = self.get_primal_variables(x)
        s_star = self.linear_projector(z) if self.linear_projector is not None else z
        return -(self.M * s_star).flatten()
