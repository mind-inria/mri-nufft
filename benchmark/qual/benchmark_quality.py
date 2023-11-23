"""Benchmark for quality of reconstruction.

Requires PySAP-MRI to be installed.

"""
import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
from hydra_callbacks.logger import PerfLogger
from hydra_callbacks.monitor import ResourceMonitorService
from modopt.math.metrics import snr, ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold

from mrinufft import get_operator
from mrinufft.trajectories.density import voronoi
from mrinufft.trajectories.io import read_trajectory

from mri.operators.proximity import AutoWeightedSparseThreshold

from solver_utils import get_grad_op, OPTIMIZERS, initialize_opt, WaveletTransform

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="ismrm2024")
def main(cfg):
    """Run benchmark of iterative reconstruction."""
    traj, params = read_trajectory(str(Path(__file__).parent / (cfg.trajectory.file)))
    traj = np.float32(traj)
    shape = tuple(params["img_size"])
    ref_data = np.load(Path(__file__).parent / cfg.ref_data)

    if ref_data.shape != shape:
        raise ValueError("shape mismatch between reference data and trajectory.")

    traj_base = Path(cfg.trajectory.file).stem
    cache_dir = Path(__file__).parent / cfg.cache_dir
    ksp_file = cache_dir / f"{traj_base}_ksp.npy"
    try:
        ksp_data = np.load(ksp_file)
    except FileNotFoundError:
        # Generate the kspace data with high precision nufft and cache it.
        finufft = get_operator(
            "finufft",
            traj,
            params["img_size"],
            n_coils=1,
            smaps=None,
            density=False,
            eps=6e-8,
        )
        ksp_data = finufft.op(ref_data)
        if getattr(cfg, "cache_dir", None):
            os.makedirs(cache_dir, exist_ok=True)
            np.save(ksp_file, ksp_data)

    # Load density weights if density is voronoi
    if cfg.trajectory.density == "voronoi":
        density = voronoi(traj)
    else:
        density = cfg.trajectory.density

    # Initialize the Fourier Operator to benchmark (n_coils = 1)
    fourier_op = get_operator(
        cfg.backend.name,
        traj,
        shape,
        n_coils=1,
        smaps=None,
        density=density,
        eps=cfg.backend.eps,
        upsampfac=cfg.backend.upsampfac,
        squeeze_dims=True,
    )

    # Setup the operators
    linear_op = WaveletTransform(
        wavelet_name=cfg.solver.wavelet.base,
        shape=shape,
        level=cfg.solver.wavelet.nb_scale,
        n_coils=1,
        mode="periodization",
    )

    regularizer_op = AutoWeightedSparseThreshold(
        linear_op.coeffs_shape,
        linear=Identity(),
        update_period=0,  # the weight is updated only once.
        sigma_range="global",
        thresh_range="global",
        threshold_estimation="sure",
        thresh_type="soft",
    )

    grad_op = get_grad_op(
        fourier_op,
        OPTIMIZERS[cfg.solver.optimizer],
        linear_op,
    )
    grad_op._obs_data = ksp_data

    solver = initialize_opt(
        cfg.solver.optimizer,
        grad_op,
        linear_op,
        regularizer_op,
        opt_kwargs={"cost": None, "progress": True},
    )
    logger.info(f"Grad inv spec rad {grad_op.inv_spec_rad}")
    backend_sig = f"{cfg.backend.name}_{cfg.backend.eps:.0e}_{cfg.backend.upsampfac}"

    # Start Reconstruction
    with (
        ResourceMonitorService(
            interval=cfg.monitor.interval, gpu_monit=cfg.monitor.gpu
        ) as monit,
        PerfLogger(logger, name=backend_sig) as perflog,
    ):
        solver.iterate(max_iter=cfg.solver.max_iter)
        if OPTIMIZERS[cfg.solver.optimizer] == "synthesis":
            x_final = linear_op.adj_op(solver.x_final)
        else:
            x_final = solver.x_final
        image_rec = np.abs(x_final)
        # Calculate SSIM
        recon_ssim = ssim(image_rec, ref_data)
        recon_snr = snr(image_rec, ref_data)

    np.save(f"recon_{backend_sig}_{traj_base}.npy", image_rec)
    logger.info(f"{backend_sig}")
    logger.info(f"SSIM, SNR: {recon_ssim}, {recon_snr}")
    results = {
        "backend": cfg.backend.name,
        "trajectory": traj_base,
        "eps": cfg.backend.eps,
        "upsampfac": cfg.backend.upsampfac,
        "end_snr": recon_snr,
        "end_ssim": recon_ssim,
        "image_rec": f"recon_{backend_sig}_{traj_base}.npy",
    }
    monit_values = monit.get_values()

    results["mem_peak"] = np.max(monit_values["rss_GiB"])
    results["run_time"] = perflog.get_timer(backend_sig)
    if cfg.monitor.gpu:
        gpu_keys = [k for k in monit_values.keys() if "gpu" in k]
        for k in gpu_keys:
            results[f"{k}_avg"] = np.mean(monit_values[k])
            results[f"{k}_peak"] = np.max(monit_values[k])

    with open(f"results_{backend_sig}.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
