"""Functions to improve/modify gradients."""

from collections.abc import Callable
from functools import partial
from typing import Literal

from tqdm.auto import tqdm
import numpy as np
import scipy as sp
import scipy.sparse as sps
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog

from mrinufft.trajectories.utils import (
    Acquisition,
    convert_gradients_to_trajectory,
    convert_trajectory_to_gradients,
    normalize_trajectory,
    unnormalize_trajectory,
)
from mrinufft._utils import MethodRegister, _fill_doc

OSQP_AVAILABLE = True
try:
    import osqp
except ImportError:
    OSQP_AVAILABLE = False


def patch_center_anomaly(
    shot_or_params: NDArray | tuple,
    update_shot: Callable[..., NDArray] | None = None,
    update_parameters: Callable[..., tuple] | None = None,
    in_out: bool = False,
    learning_rate: float = 1e-1,
) -> tuple[NDArray, tuple]:
    """Re-position samples to avoid center anomalies.

    Some trajectories behave slightly differently from expected when
    approaching definition bounds, most often the k-space center as
    for spirals in some cases.

    This function enforces non-strictly increasing monotonicity of
    sample distances from the center, effectively reducing slew
    rates and smoothing gradient transitions locally.

    Shots can be updated with provided functions to keep fitting
    their strict definition, or it can be smoothed using cubic
    spline approximations.

    Parameters
    ----------
    shot_or_params : np.array, list
        Either a single shot of shape (Ns, Nd), or a list of arbitrary
        arguments used by ``update_shot`` to initialize a single shot.
    update_shot : Callable[..., NDArray], optional
        Function used to initialize a single shot based on parameters
        provided by ``update_parameters``. If None, cubic splines are
        used as an approximation instead, by default None
    update_parameters : Callable[..., tuple], optional
        Function used to update shot parameters when provided in
        ``shot_or_params`` from an updated shot and parameters.
        If None, cubic spline parameterization is used instead,
        by default None
    in_out : bool, optional
        Whether the shot is going in-and-out or starts from the center,
        by default False
    learning_rate : float, optional
        Learning rate used in the iterative optimization process,
        by default 1e-1

    Returns
    -------
    NDArray
        N-D trajectory based on ``shot_or_params`` if a shot or
        ``update_shot`` otherwise.
    list
        Updated parameters either in the ``shot_or_params`` format
        if params, or cubic spline parameterization as an array of
        floats between 0 and 1 otherwise.
    """
    if update_shot is None or update_parameters is None:
        single_shot = shot_or_params
    else:
        parameters = shot_or_params
        single_shot = update_shot(*parameters)
    Ns = single_shot.shape[0]

    old_index = 0
    old_x_axis = np.zeros(Ns)
    x_axis = np.linspace(0, 1, Ns)

    if update_shot is None or update_parameters is None:

        def _default_update_parameters(shot: NDArray, *parameters: list) -> list:
            return parameters

        update_parameters = _default_update_parameters
        parameters = [x_axis]
        update_shot = CubicSpline(x_axis, single_shot)

    while nl.norm(x_axis - old_x_axis) / Ns > 1e-10:
        # Determine interval to fix
        single_shot = update_shot(*parameters)
        gradient_norm = nl.norm(
            np.diff(single_shot[(Ns // 2) * in_out :], axis=0), axis=-1
        )
        arc_length = np.cumsum(gradient_norm)

        index = gradient_norm[1:] - arc_length[1:] / (
            1 + np.arange(1, len(gradient_norm))
        )
        index = np.where(index < 0, np.inf, index)
        index = max(old_index, 2 + np.argmin(index))
        start_index = (Ns // 2 + (Ns % 2) - index) * in_out
        final_index = (Ns // 2) * in_out + index

        # Balance over interval
        cbs = CubicSpline(x_axis, single_shot.T, axis=1)
        gradient_norm = nl.norm(
            np.diff(single_shot[start_index:final_index], axis=0), axis=-1
        )

        old_x_axis = np.copy(x_axis)
        new_x_axis = np.cumsum(np.diff(x_axis[start_index:final_index]) / gradient_norm)
        new_x_axis *= (x_axis[final_index - 1] - x_axis[start_index]) / new_x_axis[
            -1
        ]  # Normalize
        new_x_axis += x_axis[start_index]
        x_axis[start_index + 1 : final_index - 1] += (
            new_x_axis[:-1] - x_axis[start_index + 1 : final_index - 1]
        ) * learning_rate

        # Update loop variables
        old_index = index
        single_shot = cbs(x_axis).T
        parameters = update_parameters(single_shot, *parameters)

    single_shot = update_shot(*parameters)
    return single_shot, parameters


#########################
# Gradients connections #
#########################

base_params_no_N = """\
deltak: float
    Desired change in k-space, as (k_end - k_start) / (gamma * raster_time) [T/m]
gmax: float
    Maximum gradient amplitude [T/m]
smax: float
    Maximum slew rate * raster_time (i.e., maximum gradient step) [T/m]
gs: floats
    Starting gradient value [T/m]
ge: float
    Ending gradient value [T/m]
"""


_solver_docs = dict(
    base_params=f"""\
Parameters
----------
N: int
    Number of time points for the gradient waveform
{base_params_no_N}
""",
    returns="""\
Returns
-------
NDArray
    Optimized gradient waveform of length N.
bool
    Whether the optimization was successful.
""",
    base_params_no_N=base_params_no_N,
    params_connect="""\
kstarts: NDArray
    The starting k-space points of shape (Nshots, 3), [m^-1]
kends: NDArray
    The ending k-space points of shape (Nshots, 3), [m^-1]
gstarts: NDArray
    The starting gradient points of shape (Nshots, 3), [T/m]
gends: NDArray
    The ending gradient points of shape (Nshots, 3) [T/m]
acq: Acquisition
    The acquisition object defining hardware constraints and imaging parameters
method: str, optional
    The method to use for optimization. Options are "linprog" or "osqp".
""",
)

_solvers = MethodRegister("gradient_connection_solver", _solver_docs)
_get_solver_grad = _solvers.make_getter()


@_solvers("lp")
def _solve_lp_1d(
    N: int, deltak: float, gmax: float, smax: float, gs: float, ge: float
) -> tuple[NDArray, bool]:
    """Solve a linear programming problem for getting 1D waveform.

    Such that:

    - sum(x) = Deltakx
    - |x[i]| <= gmax
    - |x[i+1]-x[i]| <= smax
    - x[0] = gx_start
    - x[N-1] = gx_end

    $base_params

    $returns

    Notes
    -----
    The inputs are normalized by gamma and raster_time. You almost certainly want to
    use `optimize_grad` instead of this function directly.
    """
    # Croping for safety
    if abs(ge) > gmax:
        ge = gmax * ge / abs(ge)
    if abs(gs) > gmax:
        gs = gmax * gs / abs(gs)

    c = np.ones(N)
    # 2. Variable Bounds
    # The bounds apply to all variables across all dimensions
    bounds = np.full((N, 2), [-gmax, gmax])

    # 3. Equality Constraints (A_eq, b_eq)
    A_eq = np.zeros((3, N))
    b_eq = np.array([deltak, gs, ge])
    A_eq[0, :] = 1  # Sum x = Deltakx
    A_eq[1, 0] = 1  # x[0] = gx_start
    A_eq[2, N - 1] = 1  # x[N-1] = gx_end

    # 4. Inequality Constraints (A_ub, b_ub)
    # Slew rate constraints: 2 * (N-1) per dimension * 3 dimensions
    num_ub = 2 * (N - 1)
    A_ub = np.zeros((num_ub, N))
    b_ub = smax * np.ones(num_ub)
    # Fill in the A_ub matrix for each dimension
    # This creates the following pattern:
    #
    # [[ 1, -1, 0, ..., 0]
    #  [ 0 , 1, -1, ..., 0]
    #   ...
    #  [-1,  1, 0, ..., 0]
    #  [ 0, -1, 1, ..., 0]
    #   ...]
    # First N-1 rows for positive slew rate constraints (x[t+1] - x[t] <= smax)
    # Next N-1 rows for negative slew rate constraints (x[t+1] - x[t] >= -smax)
    for t in range(N - 1):
        # Constraints for X dimension
        A_ub[t, t] = -1
        A_ub[t, t + 1] = 1
        A_ub[t + N - 1, t] = 1
        A_ub[t + N - 1, t + 1] = -1

    # 5. Solve the LP
    A_ub = A_ub.astype(np.float32)
    b_ub = b_ub.astype(np.float32)
    A_eq = A_eq.astype(np.float32)
    b_eq = b_eq.astype(np.float32)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
    )
    return res.x, res.success


# create alias for lp-minslew
_solvers.registry["lp-minslew"] = _solve_lp_1d


def _build_quadratic(
    N: int, gx_start: float, gx_end: float
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Build reduced quadratic form.

    for u = x[1:-1], with endpoints fixed at gx_start and gx_end.
    """
    # Sparse second-difference matrix R (size (N-2) x N)
    data = []
    rows = []
    cols = []
    for i in range(1, N - 1):
        rows += [i - 1, i - 1, i - 1]
        cols += [i - 1, i, i + 1]
        data += [1, -2, 1]
    R = sps.csr_matrix((data, (rows, cols)), shape=(N - 2, N))

    # Quadratic form
    H_full = 2 * (R.T @ R)  # sparse symmetric

    free_idx = np.arange(1, N - 1)
    fixed_idx = np.array([0, N - 1])

    # submatrix slicing with np.ix_
    H_ff = H_full[np.ix_(free_idx, free_idx)].tocsc()
    H_fb = H_full[np.ix_(free_idx, fixed_idx)].toarray()
    H_bb = H_full[np.ix_(fixed_idx, fixed_idx)].toarray()

    b = np.array([gx_start, gx_end])

    q = H_fb @ b
    c = 0.5 * b @ (H_bb @ b)

    return H_ff, q, c


@_solvers("osqp")
def _solve_qp_osqp(
    N: int, deltak: float, gmax: float, smax: float, gs: float, ge: float
) -> tuple[NDArray, bool]:
    r"""
    Solve a quadratic programming problem for getting 1D waveform.

    Such that:

    $base_params

    $returns

    Notes
    -----
    The quadratic solver uses OQSP to minimize the variation of the gradient waveform
    while satisfying the same constraints as the LP solver.

    In particular it adds a quadratic cost on the second derivative of the gradient
    waveform (minimizing :math:`\|x_i+1 - 2*x_i + x_i-1\|^2`)/
    The inputs are normalized by gamma and raster_time. You almost certainly want to
    use `optimize_grad` instead of this function directly.

    """
    # Quadratic terms
    if OSQP_AVAILABLE is False:
        raise RuntimeError("osqp package not found. Install it with `pip install osqp`")

    H, q, c = _build_quadratic(N + 2, gs, ge)
    nvar = N

    # Constraint builder lists
    data = []
    rows = []
    cols = []
    lower = []
    upper = []

    # (1) Equality: sum(u) = Delta_kx - gx_start - gx_end
    for j in range(nvar):
        data.append(1.0)
        rows.append(0)
        cols.append(j)
    lower.append(deltak - gs - ge)
    upper.append(deltak - gs - ge)
    row_counter = 1

    # (2) Inequality: slope constraints
    # left slope: u1 - gx_start
    data += [1.0]
    rows += [row_counter]
    cols += [0]
    lower.append(-smax + gs)
    upper.append(smax + gs)
    row_counter += 1

    # right slope: gx_end - u_{N-2}
    data += [-1.0]
    rows += [row_counter]
    cols += [nvar - 1]
    lower.append(-smax - ge)
    upper.append(smax - ge)
    row_counter += 1

    # interior slopes: u[i+1] - u[i]
    for i in range(nvar - 1):
        data += [-1.0, 1.0]
        rows += [row_counter, row_counter]
        cols += [i, i + 1]
        lower.append(-smax)
        upper.append(smax)
        row_counter += 1

    # (3) Bounds: -gmax <= u[i] <= gmax
    for i in range(nvar):
        data.append(1.0)
        rows.append(row_counter)
        cols.append(i)
        lower.append(-gmax)
        upper.append(gmax)
        row_counter += 1

    # Build sparse A
    A = sps.csc_matrix((data, (rows, cols)), shape=(row_counter, nvar))
    lower = np.array(lower)
    upper = np.array(upper)

    # OSQP setup
    prob = osqp.OSQP()
    prob.setup(
        P=H, q=q, A=A, l=lower, u=upper, verbose=False, eps_abs=1e-8, eps_rel=1e-8
    )
    res = prob.solve()
    return res.x, res.info.status == "solved"


@_solvers("auto")
def _solve_auto(*arg, **kwargs):
    """Automatically select the best solver available."""
    if OSQP_AVAILABLE:
        return _solve_qp_osqp(*arg, **kwargs)
    return _solve_lp_1d(*arg, **kwargs)


def _binary_search_int(
    f: Callable[[int], tuple[NDArray, bool]], low: int, high: int
) -> tuple[NDArray, int]:
    """Perfom a binary search to get best integer that makes f success."""
    x = None
    val = 0
    while low <= high:
        mid = int(low + (high - low) * 0.8)
        new_x, success = f(mid)
        if success:
            x = new_x
            high = mid - 1
            val = mid
        else:
            low = mid + 1
    if x is None:
        raise RuntimeError(f"Could not find a solution {mid}, {high}, {low}")
    return x, val


def _binary_search_float(
    f: Callable[[float], tuple[NDArray, bool]], low: float, high: float
) -> tuple[NDArray, float]:
    """Perfom a binary search to get best float that makes f success."""
    x = None
    while (high - low) / (high + low) > 1e-3:  # relative tolerance
        mid = low + (high - low) * 0.5
        new_x, success = f(mid)
        if success:
            x = new_x
            high = mid
        else:
            low = mid
    if x is None:
        raise RuntimeError(f"Could not find a solution {mid}, {high}, {low}")
    return x, mid


@_fill_doc(_solver_docs)
def _optimize_grad(
    N: int,
    deltak: float,
    gmax: float,
    smax: float,
    gs: float,
    ge: float,
    method="lp",
) -> NDArray:
    """
    Optimize a gradient waveform under hardware constraints.

    Parameters
    ----------
    $base_params

    method: str, optional
        The method to use for optimization. Options are "linprog" or "osqp".

    Returns
    -------
    NDArray
        Optimized gradient waveform of length N.

    Raises
    ------
    RuntimeError
        If N is provided and is too small to achieve the desired k-space change
        while satisfying the gradient and slew rate constraints.

    Notes
    -----
    This function calculates the required change in k-space and uses either a
    binary search to find the minimum number of time points needed (if N is
    None) or directly solves the linear programming problem for the provided N.
    The optimization is performed independently for each dimension (x, y, z).
    """
    res = []
    solver = _get_solver_grad(method)

    res, success = solver(N, deltak, gmax, smax, gs, ge)

    if not success:
        raise RuntimeError(
            f"Failed to optimize gradient waveform with given N={N} and method {method}"
        )
    if method == "osqp":
        return res

    if method == "lp-minslew":  # minimize slew rate by searching for smallest smax
        res, new_smax = _binary_search_float(
            lambda s: solver(smax=s, N=N, deltak=deltak, gmax=gmax, gs=gs, ge=ge),
            0.001 * smax,
            smax,
        )
    if not success:
        raise RuntimeError(
            f"Failed to optimize slew-rate for gradient waveform with given N={N} and"
            f" method {method}"
        )
    return res


@_fill_doc(_solver_docs)
def min_length_connection(
    kstarts: NDArray,
    kends: NDArray,
    gstarts: NDArray,
    gends: NDArray,
    acq: Acquisition | None = None,
    method: Literal["lp", "osqp"] = "lp",
) -> int:
    """
    Get the minimum length of gradient connection for a trajectory.

    Parameters
    ----------
    $params_connect

    Returns
    -------
    int
        The minimum length of the gradient connection.

    """
    acq = acq or Acquisition.default

    # The start point is the end of the previous shot,
    # The end point is the start of the next shot.
    # We will solve for all dimension independently.
    kss = kstarts.ravel()
    kes = kends.ravel()
    gss = gstarts.ravel()
    ges = gends.ravel()

    # Goal: get the length of the connection that would satisfy the constraints:
    deltak = (kes - kss) / acq.raster_time / acq.gamma

    max_grad_step = acq.smax * acq.raster_time
    gmax = acq.gmax

    solver = _get_solver_grad(method)

    low = int(np.max(abs(deltak)) / acq.gmax) + 1
    high = low + 2 * int(acq.gmax / max_grad_step)

    # Quantized to the multiple of max_grad_step, to reduces the cases to check
    quantum = 0.5 * max_grad_step
    deltak_q = np.trunc(deltak / quantum).astype(int) + np.sign(deltak)
    gss_q = np.trunc(gss / quantum).astype(int) + np.sign(gss)
    ges_q = np.trunc(ges / quantum).astype(int) + np.sign(ges)

    # loop over all possible connections, over all dimensions
    # use memoization + quantization to speed up

    cache: dict[tuple[int, int, int], int] = {}
    for dk, gs, ge in tqdm(zip(deltak_q, gss_q, ges_q)):
        try:
            cache[(dk, gs, ge)]
        except KeyError:
            gsq = gs * quantum
            geq = ge * quantum
            dkq = dk * quantum
            solve_param = partial(
                solver, deltak=dkq, gs=gsq, ge=geq, gmax=gmax, smax=max_grad_step
            )

            # check if current lower bound works, otherwise binary search
            _, success = solve_param(low)
            if success:
                cache[(dk, gs, ge)] = low
            else:
                _, low = _binary_search_int(solve_param, low, high)
            cache[(dk, gs, ge)] = low

            if low >= high:
                high = low + 2 * int(gmax / max_grad_step)
    return low


def connect_gradient(
    kstarts: NDArray,
    kends: NDArray,
    gstarts: NDArray,
    gends: NDArray,
    acq: Acquisition | None = None,
    method: str = "lp",
    N: int | None = None,
) -> NDArray:
    """
    Get the gradient connections for a set of start and end points.

    Parameters
    ----------
    $params_connect

    N: int, optional
        Number of time points to use. If None, the function will automatically
        determine the optimal number of time points.

    Returns
    -------
    NDArray
        The gradient connections of shape (Nshots,N, 3)


    Notes
    -----
    - This functions expects k-space points that are unnormalized (i.e., in m^-1).
    - If N is not provided, the function will compute the minimum length needed
    to satisfy the hardware constraints for each connection, and use the minimum length
    that satisfies all connections.


    """
    acq = acq or Acquisition.default

    N = N or min_length_connection(
        kstarts, kends, gstarts, gends, acq=acq, method=method
    )

    nshots, ndim = kstarts.shape
    connections = np.zeros((nshots, N, ndim), dtype=kstarts.dtype)
    # TODO probably wants to be parallelized and/or memoized

    deltaks = (kends - kstarts) / acq.raster_time / acq.gamma
    max_grad_step = acq.smax * acq.raster_time
    gmax = acq.gmax
    for i in range(nshots):
        for j in range(ndim):
            connections[i, :, j] = _optimize_grad(
                N=N,
                deltak=deltaks[i, j],
                gmax=gmax,
                smax=max_grad_step,
                gs=gstarts[i, j],
                ge=gends[i, j],
                method=method,
            )
    return connections


def get_prephasors_and_spoilers(
    trajectory: NDArray,
    spoil_loc=(2, 0, 0),
    spoil_grad=(0, 0, 0),
    prephase_loc=(0, 0, 0),
    prephase_grad=(0, 0, 0),
    acq: Acquisition | None = None,
    N: int | None | tuple[int, int] = None,
    method: str = "lp",
) -> NDArray | tuple[NDArray, NDArray]:
    """
    Get the prephasors and spoiler gradients for a trajectory.

    Parameters
    ----------
    trajectory: NDArray
        The trajectory of shape (Nshots, Ns, 3)
    spoil_loc: tuple, optional
        The k-space location to spoil to [m^-1], by default ``(2, 0, 0)``
    spoil_grad: tuple, optional
        The gradient to use for the spoiler [T/m], by default ``(0, 0, 0)``
    prephase_loc: tuple, optional
        The k-space location to prephase from [m^-1], by default ``(0, 0, 0)``
    prephase_grad: tuple, optional
        The gradient to use for the prephaseer [T/m], by default ``(0, 0, 0)``
    acq: Acquisition
        The acquisition object defining hardware constraints and imaging parameters
    N: int, optional
        Number of time points to use for the connections. If None

    Returns
    -------
    NDArray | tuple[NDArray, NDArray]
        The prephase and spoiler gradients of shape (Nshots, Np, 3)

    See Also
    --------
    connect_gradient

    """
    acq = acq or Acquisition.default

    nshots, Ns, ndim = trajectory.shape
    # Get the gradient waveforms
    gradients, kstarts, kends = convert_trajectory_to_gradients(trajectory, acq, True)

    # Get the starting and ending gradients and k-space points
    gstarts = gradients[:, 0, :]
    gends = gradients[:, -1, :]
    if not isinstance(N, tuple):
        N_pre = N_spoil = N
    if prephase_loc is not None:
        prephase_loc = unnormalize_trajectory(np.array(prephase_loc), acq)
        prephase_start_locations = np.tile(prephase_loc[:ndim], (nshots, 1))

        prephase_gradients = np.tile(np.array(prephase_grad)[:ndim], (nshots, 1))
        prephasers = connect_gradient(
            prephase_start_locations,
            kstarts,
            prephase_gradients,
            gstarts,
            acq=acq,
            N=N_pre,
            method=method,
        )

    if spoil_loc is not None:
        spoil_loc = unnormalize_trajectory(np.array(spoil_loc), acq)
        spoil_loc = np.tile(spoil_loc[:ndim], (nshots, 1))

        spoil_gradients = np.tile(np.array(spoil_grad)[:ndim], (nshots, 1))
        spoilers = connect_gradient(
            kends,
            spoil_loc,
            gends,
            spoil_gradients,
            acq=acq,
            N=N_spoil,
            method=method,
        )

    if prephase_loc is not None and spoil_loc is not None:
        return prephasers, spoilers
    elif prephase_loc is not None:
        return prephasers
    elif spoil_loc is not None:
        return spoilers
    else:
        raise ValueError("Either prephase_loc or spoil_loc must be provided.")
