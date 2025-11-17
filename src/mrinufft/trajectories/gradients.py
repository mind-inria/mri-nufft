"""Functions to improve/modify gradients."""

from collections.abc import Callable

import numpy as np
import scipy as sp
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog
from functools import partial

from mrinufft.trajectories.utils import Acquisition

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


def _solve_lp_1d(
    N: int, Deltakx: float, gmax: float, smax: float, gx_start: float, gx_end: float
) -> tuple[NDArray, bool]:
    """Solve a linear programming problem for getting 1D waveform.

    Such that:

    - sum(x) = Deltakx
    - |x[i]| <= gmax
    - |x[i+1]-x[i]| <= smax
    - x[0] = gx_start
    - x[N-1] = gx_end

    Parameters
    ----------
    N: int
        Number of time points for the gradient waveform
    Deltakx: float
        Desired change in k-space
    gmax: float
        Maximum gradient amplitude
    smax: float
        Maximum slew rate
    gx_start: float
        Starting gradient value
    gx_end: float
        Ending gradient value

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of the linear programming optimization containing the optimized
        gradient waveform and information about the optimization process.

    Notes
    -----
    The inputs are normalized by gamma and raster_time. You almost certainly want to
    use `optimize_grad` instead of this function directly.
    """
    # Croping for safety
    if abs(gx_end) > gmax:
        gx_end = gmax * gx_end / abs(gx_end)
    if abs(gx_start) > gmax:
        gx_start = gmax * gx_start / abs(gx_start)

    c = np.ones(N)
    # 2. Variable Bounds
    # The bounds apply to all variables across all dimensions
    bounds = np.full((N, 2), [-gmax, gmax])

    # 3. Equality Constraints (A_eq, b_eq)
    A_eq = np.zeros((3, N))
    b_eq = np.array([Deltakx, gx_start, gx_end])
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
        method="highs-ds",
        options={"verbose": 1},
    )
    return res.x, res.success


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
    R = sp.csr_matrix((data, (rows, cols)), shape=(N - 2, N))

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


def _solve_qp_osqp(
    N: int, Delta_kx: float, gmax: float, smax: float, gx_start: float, gx_end: float
) -> tuple[NDArray, bool]:
    # Quadratic terms
    H, q, c = _build_quadratic(N, gx_start, gx_end)
    nvar = N - 2

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
    lower.append(Delta_kx - gx_start - gx_end)
    upper.append(Delta_kx - gx_start - gx_end)
    row_counter = 1

    # (2) Inequality: slope constraints
    # left slope: u1 - gx_start
    data += [1.0]
    rows += [row_counter]
    cols += [0]
    lower.append(-smax + gx_start)
    upper.append(smax + gx_start)
    row_counter += 1

    # right slope: gx_end - u_{N-2}
    data += [-1.0]
    rows += [row_counter]
    cols += [nvar - 1]
    lower.append(-smax - gx_end)
    upper.append(smax - gx_end)
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
    A = sp.csc_matrix((data, (rows, cols)), shape=(row_counter, nvar))
    lower = np.array(lower)
    upper = np.array(upper)

    # OSQP setup
    prob = osqp.OSQP()
    prob.setup(
        P=H, q=q, A=A, l=lower, u=upper, verbose=False, eps_abs=1e-8, eps_rel=1e-8
    )
    res = prob.solve()
    return res.x, res.info.status == "solved"


def _binary_search_int(
    f: Callable[[int], tuple[NDArray, bool]], low: int, high: int
) -> tuple[NDArray, int]:
    """Perfom a binary search to get best optimal result on f."""
    i = 0
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
        i += 1
    if x is None:
        raise RuntimeError(f"Could not find a solution {i}, {mid}, {high}, {low}")
    return x, val


def _binary_search_float(
    f: Callable[[float], tuple[NDArray, bool]], low: float, high: float
) -> tuple[NDArray, float]:
    """Perfom a binary search to get best optimal result on f."""
    i = 0
    x = None
    while low <= high:
        mid = low + (high - low) * 0.8
        new_x, success = f(mid)
        if success:
            x = new_x
            high = mid
        else:
            low = mid
        i += 1
    if x is None:
        raise RuntimeError(f"Could not find a solution {i}, {mid}, {high}, {low}")
    return x, mid


def _optimize_grad_dimless(
    deltak: NDArray,
    gmax: float,
    smax: float,
    ge: NDArray,
    gs: NDArray,
    N_max: int = 5000,
    method="osqp",
) -> NDArray:
    """Optimize gradient waveform in time-dimensionless units.

    Parameters
    ----------
    deltak:
        desired k-space change for each dimension (x,y,z)
    gmax:
        maximum gradient amplitude
    smax:
        maximum slew rate (gradient change per time point)
    gs:
        starting gradient value for each dimension (x,y,z)
    ge:
        ending gradient value for each dimension (x,y,z)

    N_max:
        maximum number of time points to use.

    Returns
    -------
    NDArray
        Optimized gradient waveform of shape (N, len(deltak))

    Raises
    ------
    RuntimeError
        If no solution is found within the maximum number of time points.

    Notes
    -----
    This function uses a binary search to find the minimum number of time points
    required to achieve the desired k-space change while satisfying the gradient
    and slew rate constraints. The first dimension with the largest k-space change
    is used to guide the search, and then the solution is applied to all dimensions.

    Each dimension is optimized independently using linear programming.
    """
    solver = _solve_lp_1d
    if method == "osqp":
        if not OSQP_AVAILABLE:
            raise ValueError(
                "osqp package not found. Install it with `pip install osqp`"
            )
        solver = _solve_qp_osqp

    idx_max = np.argmax(abs(deltak))
    deltak_max = deltak[idx_max]
    ge_max = ge[idx_max]
    gs_max = gs[idx_max]

    # Lower bound: Assuming maximum gradient all the time
    low = (abs(deltak_max) / gmax).astype('int') + 1
    # Upper bound: Lower bound + time to go back and forth at max slew rates
    high = low + 2 * int(gmax / smax)
    high = np.min([high, np.ones_like(high)*N_max], axis=0)

    x, N = _binary_search_int(
        partial(solver, deltak=deltak_max, gmax=gmax, smax=smax, gs=gs_max, ge=ge_max),
        low,
        high,
    )

    final = np.zeros((len(x), 3))
    # now try to reduce the slew rate to smooth the waveform
    for idx in range(len(deltak)):
        if method == "lp":
            x, _ = _binary_search_float(
                partial(
                    solver, N=N, deltak=deltak[idx], gmax=gmax, gs=gs[idx], ge=ge[idx]
                ),
                low=0.001 * smax,
                high=smax,
            )
        else:
            x, success = solver(N, deltak[idx], gmax, smax, gs[idx], ge[idx])
        if x is None or not success:
            raise ValueError("Failed to complete optimization.")
        final[:, idx] = x
    return final

def _set_defaults_gradient_calc(
    kspace_end_loc: NDArray,
    kspace_start_loc: NDArray | None = None,
    end_gradients: NDArray | None = None,
    start_gradients: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    kspace_end_loc = np.atleast_2d(kspace_end_loc)
    if kspace_start_loc is None:
        kspace_start_loc = np.zeros_like(kspace_end_loc)
    if start_gradients is None:
        start_gradients = np.zeros_like(kspace_end_loc)
    if end_gradients is None:
        end_gradients = np.zeros_like(kspace_end_loc)
    kspace_start_loc = np.atleast_2d(kspace_start_loc)
    start_gradients = np.atleast_2d(start_gradients)
    end_gradients = np.atleast_2d(end_gradients)
    assert (
        kspace_start_loc.shape
        == kspace_end_loc.shape
        == start_gradients.shape
        == end_gradients.shape
    ), "All input arrays must have shape (nb_shots, nb_dimension)"
    return kspace_end_loc, kspace_start_loc, start_gradients, end_gradients


def optimize_grad(
    kspace_end_loc: NDArray,
    kspace_start_loc: NDArray | None = None,
    end_gradients: NDArray | None = None,
    start_gradients: NDArray | None = None,
    acq: Acquisition | None = None,
    N: int | None = None,
    method="lp",
) -> NDArray:
    """
    Optimize a gradient waveform under hardware constraints.

    Parameters
    ----------
    ks: NDArray
        Starting k-space position (1/m)
    ke: NDArray
        Ending k-space position (1/m)
    gs: NDArray
        Starting gradient value (T/m)
    ge: NDArray
        Ending gradient value (T/m)
    acq: Acquisition
        Acquisition object defining hardware constraints and imaging parameters
    N: int, optional
        Number of time points to use. If None, the function will automatically
        determine the optimal number of time points.

    Returns
    -------
    NDArray
        Optimized gradient waveform of shape (N, len(ks))

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
    The optimization is performed independently for each dimension (x, y, z)
    using the `solve_lp_1d` function.
    """
    acq = acq or Acquisition.default
    kspace_end_loc, kspace_start_loc, start_gradients, end_gradients = _set_defaults_gradient_calc(
        kspace_end_loc, kspace_start_loc, end_gradients, start_gradients
    )
    deltak = (kspace_end_loc - kspace_start_loc) / acq.raster_time / acq.gamma
    if N is None:  # Auto find the best connection
        return _optimize_grad_dimless(
            deltak, acq.gmax, acq.smax * acq.raster_time, start_gradients, end_gradients, method=method
        )

    res = []
    solver = _solve_lp_1d
    if method == "osqp":
        if not OSQP_AVAILABLE:
            raise ValueError("osqp is not availble. install it with `pip install osqp`")
        solver = _solve_qp_osqp

    for i in range(len(kspace_start_loc)):
        res.append(
            solver(N, deltak[i], acq.gmax, acq.smax * acq.raster_time, start_gradients[i], end_gradients[i])
        )

    # now try to reduce the slew rate to smooth the waveform
    if all(r[1] for r in res) and method == "lp":
        final = np.zeros((len(res[0][0]), 3))
        orig_smax = acq.smax * acq.raster_time
        gmax = acq.gmax
        for idx in range(len(deltak)):
            max_smax = orig_smax
            min_smax = 0.01 * orig_smax
            while (max_smax - min_smax) / min_smax >= 0.1:
                smax = min_smax + (max_smax - min_smax) * 0.8
                x, success = solver(N, deltak[idx], gmax, smax, start_gradients[idx], end_gradients[idx])
                if success:
                    max_smax = smax
                    best = x
                else:
                    min_smax = smax
            final[:, idx] = best
    else:
        raise RuntimeError("N submitted and too short")
    return final
