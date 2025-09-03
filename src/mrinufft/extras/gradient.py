"""Conjugate gradient optimization algorithm for image reconstruction."""

from mrinufft._array_compat import with_numpy_cupy
from mrinufft._utils import get_array_module
from tqdm import tqdm


@with_numpy_cupy
def cg(operator, kspace_data, x_init=None, num_iter=10, tol=1e-4, compute_loss=False, progressbar=True):
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
    lipschitz_cst = operator.get_lipschitz_cst()
    xp = get_array_module(kspace_data)
    if operator.uses_sense:
        init_shape = operator.shape
    else:
        init_shape = (operator.n_coils, *operator.shape)
    image = (
        xp.zeros(init_shape, dtype=kspace_data.dtype)
        if x_init is None
        else x_init.reshape(init_shape)
    )
    velocity = xp.zeros_like(image)

    grad = operator.data_consistency(image, kspace_data)
    velocity = tol * velocity + grad / lipschitz_cst
    image = image - velocity

    def calculate_loss(image):
        residual = operator.op(image) - kspace_data
        return xp.linalg.norm(residual) ** 2

    loss = [calculate_loss(image)] if compute_loss else None
    iterator = range(num_iter)
    if progressbar:
        iterator = tqdm(iterator)
    for _ in iterator:
        grad_new = operator.data_consistency(image, kspace_data)
        if xp.linalg.norm(grad_new) <= tol:
            break

        beta = xp.dot(
            grad_new.flatten(), (grad_new.flatten() - grad.flatten())
        ) / xp.dot(grad.flatten(), grad.flatten())
        beta = max(0, beta)  # Polak-Ribiere formula is used to compute the beta
        velocity = grad_new + beta * velocity

        image = image - velocity / lipschitz_cst
        grad = grad_new
        if compute_loss:
            loss.append(calculate_loss(image))
    return image if loss is None else (image, xp.array(loss))
