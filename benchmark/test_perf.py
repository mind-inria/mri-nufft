"""Test the performance of the operators."""
from pytest_cases import parametrize_with_cases, parametrize, fixture
import numpy as np
from mrinufft import get_operator, list_backends
from mrinufft.trajectories import initialize_2D_radial
from mrinufft.operators.stacked import stacked2traj3d


def traj_radial2D(Nc=16, Ns=512):
    """Create a 2D radial trajectory."""
    trajectory = initialize_2D_radial(Nc, Ns)
    return trajectory


N_COILS_BIG = 32
N_COILS_SMALL = 4

SELECT_BACKENDS = [
    "cufinufft",
    "finufft",
    "gpunufft",
    "bart",
    #   "sigpy",
]
SHAPE = (384, 384, 208)
STACKED_BACKENDS = [(b, None) for b in list_backends(True) if "stacked-" in b]
STACKED_BACKENDS += [("stacked", f"{{'backend':'{b}'}}") for b in SELECT_BACKENDS]
BACKENDS = [(b, None) for b in SELECT_BACKENDS]


@fixture(scope="module")
@parametrize_with_cases("kspace_locs", cases=".", prefix="traj_")
@parametrize("backend, backend_kwargs", BACKENDS + STACKED_BACKENDS)
@parametrize("z_index", ["random_mask"])
@parametrize(
    "n_coils, sense",
    [
        (1, False),
        (N_COILS_SMALL, False),
        (N_COILS_SMALL, True),
        (N_COILS_BIG, False),
        (N_COILS_BIG, True),
    ],
)
def operator(
    request,
    kspace_locs,
    backend,
    backend_kwargs,
    z_index,
    n_coils,
    sense,
):
    """Initialize the stacked and non-stacked operators."""
    shape3d = SHAPE
    if backend_kwargs:
        backend_kwargs = eval(backend_kwargs)
    else:
        backend_kwargs = {}
    if z_index == "full":
        z_index = np.arange(shape3d[-1])
    elif z_index == "random_mask":
        z_index = np.random.choice(shape3d[-1], size=shape3d[-1] // 4, replace=False)
    # smaps support
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape3d)
        smaps += np.random.rand(n_coils, *shape3d)
    else:
        smaps = None

    if "stacked" in backend:
        return get_operator(backend)(
            samples=kspace_locs,
            shape=shape3d,
            z_index=z_index,
            n_coils=n_coils,
            smaps=smaps,
            **backend_kwargs,
        )

    kspace_locs3d = stacked2traj3d(kspace_locs, z_index, shape3d[-1])
    return get_operator(backend)(
        kspace_locs3d,
        shape=shape3d,
        n_coils=n_coils,
        smaps=smaps,
        **backend_kwargs,
    )


def test_forward_perf(benchmark, operator):
    """Generate a random image."""
    if operator.uses_sense:
        shape = operator.shape
    else:
        shape = (operator.n_coils, *operator.shape)
    image_data = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    image_data += np.random.rand(*shape).astype(operator.cpx_dtype)
    """Benchmark forward operation."""
    benchmark(operator.op, image_data)


def test_adjoint_perf(benchmark, operator):
    """Benchmark adjoint operation."""
    shape = (operator.n_coils, operator.n_samples)
    kspace_data = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    kspace_data += np.random.rand(*shape).astype(operator.cpx_dtype)
    benchmark(operator.adj_op, kspace_data)


def test_grad_perf(benchmark, operator):
    """Benchmark data consistency operation."""
    if operator.uses_sense:
        shape = operator.shape
    else:
        shape = (operator.n_coils, *operator.shape)
    image_data = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    image_data += np.random.rand(*shape).astype(operator.cpx_dtype)

    shape = (operator.n_coils, operator.n_samples)
    kspace_data = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    kspace_data += np.random.rand(*shape).astype(operator.cpx_dtype)

    benchmark(operator.get_grad, image_data, kspace_data)
