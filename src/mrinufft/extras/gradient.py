"""Conjugate gradient optimization algorithm for image reconstruction."""

import numpy as np

from mrinufft.operators.base import with_numpy


@with_numpy
def cg(operator, kspace_data, x_init=None, num_iter=10, tol=1e-4):
    """
    Perform conjugate gradient (CG) optimization for image reconstruction.

    The image is updated using the gradient of a data consistency term,
    and a velocity vector is used to accelerate convergence.

    Parameters
    ----------
    kspace_data : numpy.ndarray
              The k-space data to be used for image reconstruction.

    x_init : numpy.ndarray, optional
              An initial guess for the image. If None, an image of zeros with the same
              shape as the expected output is used. Default is None.

    num_iter : int, optional
              The maximum number of iterations to perform. Default is 10.

    tol : float, optional
              The tolerance for convergence. If the norm of the gradient falls below
              this value or the dot product between the image and k-space data is
              non-positive, the iterations stop. Default is 1e-4.

    Returns
    -------
    image : numpy.ndarray
              The reconstructed image after the optimization process.
    """
    Lipschitz_cst = operator.get_lipschitz_cst()
    image = (
        np.zeros(operator.shape, dtype=type(kspace_data[0]))
        if x_init is None
        else x_init
    )
    velocity = np.zeros_like(image)

    grad = operator.data_consistency(image, kspace_data)
    velocity = tol * velocity + grad / Lipschitz_cst
    image = image - velocity

    for _ in range(num_iter):
        grad_new = operator.data_consistency(image, kspace_data)
        if np.linalg.norm(grad_new) <= tol:
            break

        beta = np.dot(grad_new.flatten(), grad_new.flatten()) / np.dot(
            grad.flatten(), grad.flatten()
        )
        velocity = grad_new + beta * velocity

        image = image - velocity / Lipschitz_cst

    return image
