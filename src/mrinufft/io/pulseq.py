"""Pulseq Trajectory Reader and writers.

Reads Pulseq `.seq` files to extract k-space trajectory and other parameters. It
also provides functionality to create pulseq block and shape objects to
facilitate the integration of arbitrary k-space trajectories into Pulseq
sequences. Requires the `pypulseq` package to be installed.
"""

import warnings
from types import SimpleNamespace
import numpy as np
import pypulseq as pp
from numpy.typing import NDArray
from .utils import prepare_trajectory_for_seq
from mrinufft.trajectories.utils import KMAX


def read_pulseq_traj(seq, trajectory_delay=0.0, gradient_offset=0.0):
    """Extract k-space trajectory from a Pulseq sequence file.

    The sequence should be a valid Pulseq `.seq` file, with arbitrary gradient
    waveforms, which all have the same length.


    Parameters
    ----------
    sequence : pulseq Sequence, or Path to the Pulseq `.seq` file.
        The Pulseq sequence object or the path to the `.seq` file.
    trajectory_delay : float, optional
        Compensation factor in seconds (s) to align ADC and gradients in the reconstruction.
    gradient_offset : float, optional
        Simulates background gradients (specified in Hz/m)
    Returns
    -------
    description : dict
        the [DESCRIPTION] block from the sequence file.
    np.ndarray
        The k-space trajectory as a numpy array of shape (n_shots, n_samples, 3),
        where the last dimension corresponds to the x, y, and z coordinates in k-space.
    """
    if not isinstance(seq, pp.Sequence):
        filename = seq
        seq = pp.Sequence()
        seq.read(filename)

    kspace_adc, _, t_exc, t_refocus, t_adc = seq.calculate_kspace(
        trajectory_delay=trajectory_delay, gradient_offset=gradient_offset
    )

    if not t_adc:
        raise ValueError(
            "The sequence does not contain any ADC events. "
            "Please ensure that the sequence has ADC events defined."
        )
    # split t_adc with t_exc and t_refocus, the index are then used to split kspace_adc
    FOV = seq.get_definition("FOV")
    t_splits = np.sort(np.concatenate([t_exc, t_refocus]))
    idx_prev = 0
    kspace_shots = []
    for t in t_splits:
        idx_next = np.searchsorted(t_adc, t, side="left")
        if idx_next == idx_prev:
            continue
        kspace_shots.append(kspace_adc[:, idx_prev:idx_next].T)
        if idx_next == kspace_adc.shape[1] and t > t_adc[-1]:  # last useful point
            break
        idx_prev = idx_next
    if idx_next < kspace_adc.shape[1]:
        kspace_shots.append(kspace_adc[:, idx_next:].T)  # add remaining gradients.
    # convert to KMAX standard.
    kspace_shots = np.ascontiguousarray(kspace_shots) * KMAX * 2 * np.asarray(FOV)
    return seq.definitions, kspace_shots


def _check_timings(seq):
    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        return None
    else:
        warnings.warn("Timing check failed. Error listing follows:" + str(error_report))


def pulseq_gre(
    trajectory: NDArray,
    fov: tuple[float, float, float],
    img_size: tuple[int, int, int],
    TR: float,
    TE: float,
    TE_pos: float = 0.5,
    FA: float | None = None,
    gre_2D: bool = False,
    slice_overlap: float = 0.0,
    rf_pulse: SimpleNamespace | None = None,
    rf_spoiling_inc: float = 0.0,
    grad_spoil_factor: float = 2.0,
    system: pp.Opts | None = None,
    osf: int = 1,
) -> pp.Sequence:
    """Create a Pulseq 3D-GRE sequence for arbitrary trajectories.

    Parameters
    ----------
    trajectory : np.ndarray
        The k-space trajectory as a numpy array of shape (n_shots, n_samples, 3),
        where the last dimension corresponds to the x, y, and z coordinates in k-space.
    TR: float
        The repetition time in milliseconds (ms).
    TE: float
        The echo time in milliseconds (ms).
    FA: float, optional, incompatible with `rf_pulse`
        The flip angle in degrees (°).
    TE_pos: float, optional
        The relative (0-1) position of the echo time within each kspace_shot.
    rf_pulse: SimpleNamespace, optional, incompatible with `FA`
        A custom radio-frequency pulse object. If not provided, a block pulse
        with the specified flip angle and a duration of 4 ms will be created.
        see `pypulseq.make_block_pulse` or `pypulseq.make_arbitrary_rf` for more
        details.
    rf_spoiling_inc: float, optional
        The increment in the RF phase (in degree) for spoiling. Default is 0.0,
        which means no spoiling.
    gre_2D: bool, optional
        If True, the sequence will be a 2D GRE sequence. Default is False,
        which means a 3D GRE sequence.
    slice_overlap: float, optional
        The slice overlap proportion for 2D GRE sequences. Default is 0.0
        Positive values indicates an overlap, negative values indicates a gap.
    grad_spoil_factor: float, optional
        The factor by which the spoiler gradient moves to the edge of k-space. Default is 2.0.
    osf: int, optional
        The oversampling factor for the ADC. Default is 1, which means no oversampling.

    system : pypulseq.Opts, optional
        The system options for the Pulseq sequence. Default is `pp.Opts.default`.

    Notes
    -----
    The  Sequence cycle can be summarized as follows:

    1. RF pulse
    2. Delay to sync TE
    3. Gradients plays: The gradients consist in a prewind to the first point
    of the trajectory, the trajectory itself, and a rewind to the edge of
    k-space.
    3bis. The ADC is opened on the trajectory points (ignoring the prewind and
    rewinds parts)
    4. Gradient spoilers
    5. Delay to sync the next TR


    If `gre_2D` is True, the sequence will be a 2D GRE sequence, and the slice
    thickness will be determined as :math:`(FOV[2] / img_size[2]) *
    (1+slice_overlap)`

    Returns
    -------
    pp.Sequence
        A Pulseq sequence object with the specified arbitrary gradient waveform.
    """
    TR = TR / 1000.0  # convert to seconds
    TE = TE / 1000.0  # convert to seconds
    if system is None:
        system = pp.Opts.default
    seq = pp.Sequence(system=system)

    if gre_2D and rf_pulse is not None:
        raise ValueError(
            "2D GRE sequences do not support custom RF pulses. "
            "Please set `rf_pulse` to None or use a block pulse."
        )
    if rf_pulse is None and FA is not None:
        rf_pulse = pp.make_block_pulse(flip_angle=FA, system=system, use="excitation")
    elif rf_pulse is not None and FA is not None:
        raise ValueError(
            "Cannot specify both `rf_pulse` and `FA`. Please provide only one."
        )
    elif rf_pulse is None and FA is None:
        raise ValueError("Either `rf_pulse` or `FA` must be provided.")

    if not isinstance(rf_pulse, SimpleNamespace):
        raise TypeError(
            "The `rf_pulse` parameter must be a SimpleNamespace object, "
            "as returned by `pypulseq.make_block_pulse` or `pypulseq.make_arbitrary_rf`"
        )
    if FA is None:
        # Compute the flip angle from the RF pulse
        # following https://github.com/imr-framework/pypulseq/tree/src/pypulseq/Sequence/ext_test_report.py#L24
        FA = float(
            abs(np.sum(rf_pulse.signal[:-1] * (rf_pulse.t[1:] - rf_pulse.t[:-1]))) * 360
        )

    seq.set_definition("FOV", fov)
    seq.set_definition("ImgSize", img_size)
    seq.set_definition("TR", TR)
    seq.set_definition("TE", TE)
    seq.set_definition("FA", FA)

    full_grads, skip_start, skip_end = prepare_trajectory_for_seq(
        trajectory,
        fov=fov,
        img_size=img_size,
        raster_time=system.grad_raster_time,
        gamma=system.gamma,
        gmax=system.max_grad,
        smax=system.max_slew,
        pregrad="prephase",
        postgrad="slowdown_to_edge",
    )
    full_grads = full_grads * system.gamma  # convert to rad/s/m
    traj_length = full_grads.shape[1] - skip_start - skip_end + 1
    shot_duration = system.grad_raster_time * traj_length  # in seconds
    adc = pp.make_adc(
        num_samples=traj_length * osf,
        system=system,
        duration=shot_duration,
        delay=skip_start * system.grad_raster_time,
    )

    # Add a spoiler gradient that move to twice the edge of k-space in the x direction.
    spoiler = pp.make_trapezoid(
        channel="x", area=grad_spoil_factor * img_size[0] / fov[0], system=system
    )

    delay_before_grad = pp.make_delay(
        TE
        - pp.calc_rf_center(rf_pulse)[0]
        - rf_pulse.delay
        - TE_pos * shot_duration
        - skip_start * system.grad_raster_time
    )
    delay_end_TR = pp.make_delay(
        TR
        - pp.calc_duration(rf_pulse)
        - delay_before_grad.delay
        - full_grads.shape[1] * system.grad_raster_time
    )
    if gre_2D:
        return _pulseq_gre_2D(
            seq,
            full_grads[:, :2],
            rf_spoiling_inc,
            adc,
            spoiler,
            delay_before_grad,
            delay_end_TR,
            thickness=fov[2] / img_size[2] * (1 + slice_overlap),
            slice_locs=trajectory[:,2]
        )

    rf_phase = 0.0
    for grad_xyz in full_grads:
        rf_phase = divmod(rf_phase + rf_spoiling_inc, 360.0)[1]
        rf_pulse.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        seq.add_block(rf_pulse)  # RF pulse
        seq.add_block(delay_before_grad)  # delay to sync TE
        # Add the gradient waveform, the first/last points are set to 0
        # to avoid accumulating offsets between shots
        # https://github.com/imr-framework/pypulseq/discussions/175
        seq.add_block(
            *[
                pp.make_arbitrary_grad(
                    channel=c, waveform=grad_xyz[:, i], first=0, last=0, system=system
                )
                for i, c in enumerate("xyz")
            ],
            adc,
        )
        seq.add_block(spoiler, delay_end_TR)

    _check_timings(seq)
    return seq


def _pulseq_gre_2D(
    seq: pp.Sequence,
    full_grads: NDArray,
    rf_spoiling_inc: float,
    adc: SimpleNamespace,
    spoiler: SimpleNamespace,
    delay_before_grad: SimpleNamespace,
    delay_end_TR: SimpleNamespace,
    thickness: float,
    slice_locs: NDArray
) -> pp.Sequence:
    """Create a Pulseq 2D-GRE sequence for arbitrary trajectories."""

    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=float(seq.get_definition("FA")),
        slice_thickness=thickness,
        return_gz=True,
        system=seq.system,
    )  # type: ignore[assignment]

    rf_phase = 0.0
    nz = seq.get_definition("ImgSize")[2]
    for grad_xyz, kz in zip(full_grads, slice_locs):
        rf_phase = divmod(rf_phase + rf_spoiling_inc, 360.0)[1]

        # Update the Gz component
        # Update the RF pulse phase offset
        rf.frequency_offset = gz.amplitude * thickness * kz * nz
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        seq.add_block(rf, gz)  # RF pulse and slice selection gradient for a 2D
        seq.add_block(adc)  # ADC
        seq.add_block(delay_before_grad)  # delay to sync TE
        # Add the gradient waveform, the first/last points are set to 0
        # to avoid accumulating offsets between shots
        seq.add_block(
            *[
                pp.make_arbitrary_grad(
                    channel=c,
                    waveform=grad_xyz[:, i],
                    first=0,
                    last=0,
                    system=seq.system,
                )
                for i, c in enumerate("xyz")
            ], # Gx and Gy are arbitrary gradients
            gzr, # Refocusing gradient for Gzr
        )
        seq.add_block(spoiler, delay_end_TR)

    _check_timings(seq)
    return seq
