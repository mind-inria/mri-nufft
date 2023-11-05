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
from mri.operators import WaveletN
from mri.reconstructors import SingleChannelReconstructor

from mrinufft import get_operator
from mrinufft.trajectories.density import voronoi
from mrinufft.trajectories.io import read_trajectory

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf-qual", config_name="ismrm2024")
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
    )

    monit = ResourceMonitorService(
        interval=cfg.monitor.interval, gpu_monit=cfg.monitor.gpu
    )

    # Setup the operators
    linear_op = WaveletN(
        wavelet_name=cfg.solver.wavelet.base,
        dim=len(shape),
        nb_scales=cfg.solver.wavelet.nb_scale,
    )

    # Manual tweak of the regularisation parameter
    regularizer_op = SparseThreshold(Identity(), cfg.solver.lmbd, thresh_type="soft")
    # Setup Reconstructor
    reconstructor = SingleChannelReconstructor(
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=regularizer_op,
        gradient_formulation="synthesis",
        verbose=1,
    )

    backend_sig = f"{cfg.backend.name}_{cfg.backend.eps:.0e}_{cfg.backend.upsampfac}"

    # Start Reconstruction
    with (
        monit,
        PerfLogger(logger, name=backend_sig) as perflog,
    ):
        x_final, costs, metrics = reconstructor.reconstruct(
            kspace_data=ksp_data,
            optimization_alg="pogm",
            num_iterations=cfg.solver.max_iter,
            cost_op_kwargs={"cost_interval": None},
            metric_call_period=1,
            metrics={
                "snr": {
                    "metric": snr,
                    "mapping": {"x_new": "test"},
                    "cst_kwargs": {"ref": ref_data},
                    "early_stopping": False,
                },
                "ssim": {
                    "metric": ssim,
                    "mapping": {"x_new": "test"},
                    "cst_kwargs": {"ref": ref_data},
                    "early_stopping": False,
                },
            },
        )

        image_rec = np.abs(x_final)
        # image_rec.show()
        # Calculate SSIM
        recon_ssim = ssim(image_rec, ref_data)
        recon_snr = snr(image_rec, ref_data)

    logger.info(f"{backend_sig}")
    logger.info(f"SSIM, SNR: {recon_ssim}, {recon_snr}")
    results = {
        "backend": cfg.backend,
        "eps": cfg.backend.eps,
        "upsampfac": cfg.backend.upsampfac,
        "end_snr": recon_snr,
        "end_ssim": recon_ssim,
    }
    monit_values = monit.get_values()

    results["mem_peak"] = np.max(monit_values["rss_GiB"])
    results["run_time"] = perflog.get_time(backend_sig)
    if cfg.monitor.gpu:
        gpu_keys = [k for k in monit_values.keys() if "gpu" in k]
        for k in gpu_keys:
            results[f"{k}_avg"] = np.mean(monit_values[k])
            results[f"{k}_peak"] = np.max(monit_values[k])

    with open(f"results_{backend_sig}.json") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
