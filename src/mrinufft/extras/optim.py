"""Implements the LSQR algorithm."""

from collections.abc import Callable
import numpy as np
from tqdm.auto import tqdm
from numpy.typing import NDArray

from mrinufft._array_compat import get_array_module, with_numpy_cupy
from mrinufft.operators.base import FourierOperatorBase
from mrinufft._utils import MethodRegister


_optim_docs = dict(
    base_params=r"""
nufft: FourierOperatorBase
    The NUFFT operator representing the forward model.
kspace_data: NDArray
    The right-hand side vector (`kspace` data). Shape typically (n_batchs,
    n_coils, n_samples).
damp: float, optional
    Damping (regularization) parameter. Default is 0.0 (no regularization).
x0: NDArray or None, optional
    Damping vector. If None, uses zero. Shape must be
x_init: NDArray or None, optional
    Initial guess vector. If ommitted, default to x0.
callback: Callable, optional
    If provided, a callback function will be called at the end of each
    iteration with the current estimate. It should have the following signature
    ``callback(operator, kspace_data, damp, x0)``
n_iter: int, optional
    Maximum number of iterations. Default is 100.
progressbar: bool, optional
    If True (default) display a progress bar to track iterations.
""",
    returns="""
Returns
-------
NDArray:
    Solution vector with shape (n_batchs, n_coils or 1, *nufft.shape), dtype and
    device matching input.
""",
)


register_optim = MethodRegister("optim", _optim_docs)
get_optimizer = register_optim.make_getter()


def norm_batched(x) -> NDArray:
    """Compute the norm of x, preserving the first (batch) dimension."""
    xp = get_array_module(x)
    return xp.sqrt(xp.sum(abs(x) ** 2, axis=tuple(range(1, x.ndim)))).squeeze()


def bc_left(x, y):
    """Broadcast x to y shape, starting from first axis.

    Regular numpy broadcasting start from the last axis.
    """
    return x.reshape(x.shape + (1,) * (y.ndim - x.ndim))


@with_numpy_cupy
def loss_l2_reg(
    image: NDArray,
    operator: FourierOperatorBase,
    kspace_data: NDArray,
    damp: float = 0.0,
    x0: NDArray | None = None,
):
    """
    Compute the regularized least squares loss for MRI reconstruction.

    Computes the loss:
        ||A x - y||_2^2 + damp^2 ||x - x0||_2^2
    where A is the measurement operator, x is the current image estimate, y is
    the acquired k-space data, damp is a regularization parameter, and x0 is an
    initial guess.

    Parameters
    ----------
    image : NDArray
        Current image estimate. Shape and dtype must be compatible with the operator.
    operator : FourierOperatorBase
        The NUFFT (non-uniform FFT) operator used for forward modeling.
    kspace_data : NDArray
        Measured k-space data. Shape must match the output of the operator.op(image).
    damp : float or None, optional
        Regularization parameter (default=None). If None, no regularization is applied.
    x0 : NDArray or None, optional
        Reference image for regularization (default=None).

    Returns
    -------
    norm_res : float or NDArray
        The computed L2 regularized least squares loss value(s).
        If batched, shape = (n_batchs,).

    Notes
    -----
    - Batch dimension is preserved if present.
    - This function can be used as a callback in cg or lsqr method to keep track
      of the convergence.

    """
    residual = operator.op(image).reshape(operator.ksp_full_shape)
    residual -= kspace_data.reshape(operator.ksp_full_shape)
    residual.reshape(operator.n_batchs, -1)
    norm_res = norm_batched(residual).squeeze()

    if damp:
        image_ = image.reshape(operator.img_full_shape)
        if x0 is not None:
            image_ = image_ - x0.reshape(operator.img_full_shape)
        norm_damp = damp**2 * norm_batched(image_.reshape(operator.n_batchs, -1))
        norm_res += norm_damp
    return norm_res


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    """
    xp = get_array_module(a)
    if xp.any(b == 0):
        return xp.sign(a), 0, abs(a)
    elif xp.any(a == 0):
        return 0, xp.sign(b), abs(b)
    elif xp.any(abs(b) > abs(a)):
        tau = a / b
        s = xp.sign(b) / xp.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = xp.sign(a) / xp.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


@register_optim
@with_numpy_cupy
def lsqr(
    operator: FourierOperatorBase,
    kspace_data: NDArray,
    damp: float = 0.0,
    atol: float = 1e-6,
    btol: float = 1e-6,
    conlim: float = 1e8,
    n_iter: int = 100,
    x0: NDArray | None = None,
    x_init: NDArray | None = None,
    callback: Callable | None = None,
    progressbar: bool = True,
):
    r"""
    Solve a general regularized linear least-squares problem using the LSQR algorithm.

    Solves problems of the form::

        minimize ||A x - b||_2^2 + damp^2 ||x - x0||_2^2

    Stop iterating if:
    - numerical convergence is reached: ``norm(Ax-b) <= atol * norm(A) * norm(x)
      + btol * norm(b)``
    - estimation of the conditioning of the problem diverge: ``cond(A)>=conlim``
    - Maximum number of iteration reached.

    Parameters
    ----------
    $base_params

    atol : float, optional
        Stopping tolerance on the absolute error. Default is 1e-6.
    btol : float, optional
        Stopping tolerance on the relative error. Default is 1e-6.
    conlim : float, optional
        Limit on condition number. Iteration stops if condition exceeds this
        value. Default is 1e8.

    $returns

    References
    ----------
    .. [1] Paige, C. C., & Saunders, M. A. (1982). LSQR: An algorithm for sparse
           linear equations and sparse least squares. ACM Transactions on Mathematical
           Software, 8(1), 43-71.
    .. [2] S.-C. Choi, "Iterative Methods for Singular Linear Equations and
           Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    .. [3] https://github.com/scipy/scipy/blob/v1.16.2/scipy/sparse/linalg/_isolve/lsqr.py
    """
    xp = get_array_module(kspace_data)

    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim

    eps = xp.finfo(kspace_data.dtype).eps

    IMG_COIL_DIM = operator.n_coils if not operator.uses_sense else 1

    def AT(y):
        return operator.adj_op(y).reshape(
            operator.n_batchs, IMG_COIL_DIM, *operator.shape
        )

    def A(x):
        return operator.op(x).reshape(
            operator.n_batchs, operator.n_coils, operator.n_samples
        )

    kspace_data = kspace_data.reshape(
        (operator.n_batchs, operator.n_coils, operator.n_samples)
    )
    if kspace_data.ndim > 1:
        kspace_data.squeeze()

    u = kspace_data.copy()
    bnorm = norm_batched(u)

    if x_init is None:
        if x0 is None:
            x_init = xp.zeros(
                (operator.n_batchs, IMG_COIL_DIM, *operator.shape),
                dtype=operator.cpx_dtype,
            )
        else:
            x_init = xp.copy(x0)
    x = x_init

    beta = bnorm.copy()

    if x0 is not None:
        u -= A(x)
        beta = norm_batched(u)

    if xp.all(beta) > 0:
        u /= bc_left(beta, u)
        v = AT(u)
        alpha = norm_batched(v)
    else:
        v = xp.copy(x)
        alpha = xp.zeros(v.shape[0])

    if xp.any((alpha * beta) == 0):
        return x

    if xp.all(alpha) > 0:
        v /= bc_left(alpha, v)
    w = xp.copy(v)

    rhobar = alpha
    phibar = rnorm = r1norm = beta
    arnorm = alpha * beta

    ddnorm = res2 = xnorm = xxnorm = z = anorm = acond = 0.0
    dampsq = damp**2

    cs2 = -1
    sn2 = 0.0
    istop = 0
    callback_returns = []
    for _ in tqdm(range(n_iter), disable=not progressbar):
        u *= -bc_left(alpha, u)
        u += A(v)
        beta = norm_batched(u)

        if xp.all(beta) > 0:
            u /= bc_left(beta, u)
            anorm = xp.sqrt(anorm**2 + alpha**2 + beta**2 + dampsq)
            v *= -bc_left(beta, v)
            v += AT(u)
            alpha = norm_batched(v)
            if xp.all(alpha) > 0:
                v /= bc_left(alpha, v)
        if damp:
            rhobar1 = xp.sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.0
        # use a plane rotation to elimiate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        t1 = phi / rho
        t2 = -theta / rho
        dk = w / bc_left(rho, w)

        # update x and w
        x += bc_left(t1, w) * w
        w *= bc_left(t2, w)
        w += v

        ddnorm += norm_batched(dk) ** 2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = xp.sqrt(xxnorm + zbar**2)
        gamma = xp.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm += z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * xp.sqrt(ddnorm)
        res1 = phibar**2
        res2 += psi**2
        rnorm = xp.sqrt(res1 + res2)
        arnorm = alpha * xp.abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x - x0||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x - x0||^2).
        # Although there is cancellation, it might be accurate enough.
        if damp > 0:
            r1sq = rnorm**2 - dampsq * xxnorm
            r1norm = xp.sqrt(xp.abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if callback:
            callback_returns.append(
                callback(x, operator, kspace_data, damp=damp, x0=x0)
            )

        if xp.all(1 + test3 <= 1):
            istop = 6
        elif xp.all(1 + test2 <= 1):
            istop = 5
        elif xp.all(1 + t1 <= 1):
            istop = 4
        # Allow for tolerances set by the user.
        elif xp.all(test3 <= ctol):
            istop = 3
        elif xp.all(test2 <= atol):
            istop = 2
        elif xp.all(test1 <= rtol):
            istop = 1

        if istop:
            break
    if operator.squeeze_dims:
        x = operator._safe_squeeze(x)
    if callback_returns:
        return x, callback_returns
    return x


@register_optim
@with_numpy_cupy
def lsmr(
    operator: FourierOperatorBase,
    kspace_data: NDArray,
    damp: float = 0.0,
    atol: float = 1e-6,
    btol: float = 1e-6,
    conlim: float = 1e8,
    n_iter: int = 100,
    x0: NDArray | None = None,
    x_init: NDArray | None = None,
    callback: Callable | None = None,
    progressbar: bool = True,
):
    r"""
    Solve a general regularized linear least-squares problem using the LSQR algorithm.

    Solves problems of the form::

        minimize ||A x - b||_2^2 + damp^2 ||x - x0||_2^2

    Stop iterating if:
    - numerical convergence is reached: ``norm(Ax-b) <= atol * norm(A) * norm(x)
      + btol * norm(b)``
    - estimation of the conditioning of the problem diverge: ``cond(A)>=conlim``
    - Maximum number of iteration reached.

    Parameters
    ----------
    $base_params

    atol : float, optional
        Stopping tolerance on the absolute error. Default is 1e-6.
    btol : float, optional
        Stopping tolerance on the relative error. Default is 1e-6.
    conlim : float, optional
        Limit on condition number. Iteration stops if condition exceeds this
        value. Default is 1e8.

    $returns

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

    """
    xp = get_array_module(kspace_data)

    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim

    #   eps = xp.finfo(kspace_data.dtype).eps

    IMG_COIL_DIM = operator.n_coils if not operator.uses_sense else 1

    def AT(y):
        return operator.adj_op(y).reshape(
            operator.n_batchs, IMG_COIL_DIM, *operator.shape
        )

    def A(x):
        return operator.op(x).reshape(
            operator.n_batchs, operator.n_coils, operator.n_samples
        )

    kspace_data = kspace_data.reshape(
        (operator.n_batchs, operator.n_coils, operator.n_samples)
    )
    if kspace_data.ndim > 1:
        kspace_data.squeeze()

    u = kspace_data.copy()

    normb = norm_batched(u)

    if x_init is None:
        if x0 is None:
            x_init = xp.zeros(
                (operator.n_batchs, IMG_COIL_DIM, *operator.shape),
                dtype=operator.cpx_dtype,
            )
        else:
            x_init = xp.copy(x0)
    x = x_init

    beta = normb.copy()

    if x0 is not None:
        u -= A(x)
        beta = norm_batched(u)

    if xp.all(beta) > 0:
        u /= bc_left(beta, u)
        v = AT(u)
        alpha = norm_batched(v)
    else:
        v = xp.copy(x)
        alpha = xp.zeros(v.shape[0])

    if xp.any((alpha * beta) == 0):
        return x

    if xp.all(alpha) > 0:
        v /= bc_left(alpha, v)

    damp = xp.full(operator.n_batchs, damp, xp.float32)

    # initialize variable for 1st iteration
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = xp.zeros(v.shape, operator.cpx_dtype)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e100
    normA = xp.sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    normr = beta

    callback_returns = []
    for _ in tqdm(range(n_iter)):

        u *= -bc_left(alpha, u)
        u += A(v)
        beta = norm_batched(u)

        if xp.all(beta) > 0:
            u /= bc_left(beta, u)
            v *= -bc_left(beta, v)
            v += AT(u)
            alpha = norm_batched(v)
            if xp.all(alpha) > 0:
                v /= bc_left(alpha, v)

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, h_hat, x.

        hbar *= -(thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= -(thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = xp.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = xp.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = norm_batched(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = xp.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if callback:
            callback_returns.append(
                callback(x, operator, kspace_data, damp=damp, x0=x0)
            )

        if 1 + test3 <= 1:
            istop = 6
        elif 1 + test2 <= 1:
            istop = 5
        elif 1 + t1 <= 1:
            istop = 4
        # Allow for tolerances set by the user.
        elif test3 <= ctol:
            istop = 3
        elif test2 <= atol:
            istop = 2
        elif test1 <= rtol:
            istop = 1

        if istop:
            break

    if operator.squeeze_dims:
        x = operator._safe_squeeze(x)
    if callback_returns:
        return x, callback_returns
    return x


@register_optim
@with_numpy_cupy
def cg(
    operator: FourierOperatorBase,
    kspace_data: NDArray,
    damp: float = 0.0,
    x0: NDArray | None = None,
    x_init: NDArray | None = None,
    n_iter: int = 10,
    tol: float = 1e-4,
    progressbar: bool = True,
    callback: Callable | None = None,
):
    r"""
    Perform conjugate gradient (CG) optimization for image reconstruction.

    The image is updated using the gradient of a data consistency term,
    and a velocity vector is used to accelerate convergence.

    Parameters
    ----------
    $base_params

    tol: float
        Tolerance for converge check.

    $returns

    References
    ----------
    https://en.m.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    """
    lipschitz_cst = operator.get_lipschitz_cst()
    xp = get_array_module(kspace_data)
    if operator.uses_sense:
        init_shape = (operator.n_batchs, 1, *operator.shape)
    else:
        init_shape = (operator.n_batchs, operator.n_coils, *operator.shape)
    image = (
        xp.zeros(init_shape, dtype=kspace_data.dtype)
        if x_init is None
        else x_init.reshape(init_shape)
    )
    velocity = xp.zeros_like(image)

    grad = operator.data_consistency(image, kspace_data)
    if damp:
        if x0:
            grad += damp * (image - x0)
        else:
            grad += damp * image
    velocity = tol * velocity + grad / lipschitz_cst
    image = image - velocity

    callbacks_results = []
    for _ in tqdm(range(n_iter), disable=not progressbar):
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
        if callback:
            callbacks_results.append(
                callback(
                    image,
                    operator,
                    kspace_data,
                    damp=damp,
                    x0=x0,
                )
            )
    if operator.squeeze_dims:
        image = operator._safe_squeeze(image)

    if callbacks_results:
        return image, callbacks_results
    return image
