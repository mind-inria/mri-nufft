import warnings
import csv
import hydra
import logging
import os
import numpy as np
from itertools import product
from omegaconf import DictConfig
from time import perf_counter
from pathlib import Path

from mrinufft import get_operator
from mrinufft.trajectories.io import read_trajectory
from hydra_callbacks.monitor import ResourceMonitorService
from hydra_callbacks.logger import PerfLogger

from utils import get_smaps


logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    "Samples will be rescaled to .*",
    category=UserWarning,
    module="mrinufft",
)


def get_data(cfg):
    """Initialize all the data for the benchmark."""
    # trajectory init
    if cfg.trajectory.endswith(".bin"):
        trajectory, params = read_trajectory(
            str(Path(__file__).parent / cfg.trajectory)
        )
    else:
        eval(trajectory.name)(**trajectory.kwargs)

    cpx_type = np.dtype(cfg.data.dtype)
    if cpx_type == np.complex64:
        trajectory = trajectory.astype(np.float32)
    C = cfg.data.n_coils
    XYZ = tuple(params["img_size"])
    K = np.prod(trajectory.shape[:-1])

    if data_file := getattr(cfg.data, "file", None):
        data = np.load(data_file)
        if data.shape != XYZ:
            logger.warning("mismatched shape between data and trajectory file.")

    else:
        data = (1j * np.random.rand(*XYZ)).astype(cpx_type)
        data += np.random.rand(*XYZ).astype(cpx_type)
        # kspace data
    ksp_data = 1j * np.random.randn(C, K).astype(cpx_type)
    ksp_data += np.random.randn(C, K).astype(cpx_type)

    # smaps
    smaps_true = get_smaps(XYZ, C)
    if cfg.data.smaps:
        smaps = smaps_true
    else:
        # expand the data to multicoil
        data = data[None, ...] * smaps_true
        smaps = None

    return (data, ksp_data, trajectory, smaps, XYZ, C)


@hydra.main(
    config_path="conf",
    config_name="benchmark_config",
    version_base=None,
)
def main_app(cfg: DictConfig) -> None:
    """Run the benchmark."""
    # TODO Add a DSL like bart::extra_args:value::extra_arg2:value2 etc
    nufftKlass = get_operator(cfg.backend)
    nufft = None

    data, ksp_data, trajectory, smaps, shape, n_coils = get_data(cfg)
    print(data.shape, ksp_data.shape, trajectory.shape, n_coils, shape)
    monit = ResourceMonitorService(
        os.getpid(),
        interval=cfg.monitor.interval,
        gpu_monit=cfg.monitor.gpu,
    )

    for task, i in product(cfg.task, range(cfg.n_runs)):
        nufft = nufftKlass(
            trajectory,
            shape,
            n_coils=n_coils,
            smaps=smaps,
            **getattr(cfg.backend, "kwargs", {}),
        )
        with (
            monit,
            PerfLogger(logger, name=f"{nufft.backend}_{task}, #{i}") as perflog,
        ):
            if task == "forward":
                nufft.op(data)
            elif task == "adjoint":
                nufft.adj_op(ksp_data)
            elif task == "grad":
                nufft.get_grad(data, ksp_data)
            else:
                raise ValueError(f"Unknown task {task}")

        values = monit.get_values()
        monit_values = {
            "backend": cfg.backend,
            "task": task,
            "run": i,
            "run_time": perflog.get_timer(f"{nufft.backend}_{task}, #{i}"),
            "mem_avg": np.mean(values["rss_GiB"]),
            "mem_peak": np.max(values["rss_GiB"]),
            "cpu_avg": np.mean(values["cpus"]),
            "cpu_peak": np.max(values["cpus"]),
        }
        if cfg.monitor.gpu:
            gpu_keys = [k for k in values.keys() if "gpu" in k]
            for k in gpu_keys:
                monit_values[f"{k}_avg"] = np.mean(values[k])
                monit_values[f"{k}_peak"] = np.max(values[k])

            del nufft
        with open("results.csv", "a") as f:
            writer = csv.DictWriter(f, fieldnames=monit_values.keys())
            f.seek(0, os.SEEK_END)
            if not f.tell():
                writer.writeheader()
            writer.writerow(monit_values)


if __name__ == "__main__":
    main_app()
