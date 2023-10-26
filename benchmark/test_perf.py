"""Test the performance of the operators."""
from pytest_cases import parametrize_with_cases, parametrize, fixture
import numpy as np
from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial
from mrinufft.operators.stacked import stacked2traj3d


def traj_radial2D(Nc=10, Ns=500):
    """Create a 2D radial trajectory."""
    trajectory = initialize_2D_radial(Nc, Ns)
    return trajectory


@fixture(scope="module")
@parametrize_with_cases("kspace_locs", cases=".", prefix="traj_")
@parametrize(
    "backend, backend_kwargs",
    [
        ("finufft", None),
        # ("sigpy", None),
        # ("cufinufft", {}),
        # ("gpunufft", {}),
        ("stacked", "{'backend':'finufft'}"),
        # ("stacked-cufinufft", {}),
    ],
)
@parametrize("z_index", ["full", "random_mask"])
@parametrize(
    "n_coils, sense",
    [(1, False), (4, False), (4, True)],
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
    shape = (64, 64)
    shape3d = (*shape, shape[-1])  # add a 3rd dimension

    if backend_kwargs:
        backend_kwargs = eval(backend_kwargs)
    else:
        backend_kwargs = {}
    if z_index == "full":
        z_index = np.arange(shape3d[-1])
    elif z_index == "random_mask":
        z_index = np.random.choice(shape3d[-1], size=shape3d[-1] // 2, replace=False)
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

    kspace_locs3d = stacked2traj3d(kspace_locs, z_index, shape[-1])
    return get_operator(backend)(
        kspace_locs3d,
        shape=shape3d,
        n_coils=n_coils,
        smaps=smaps,
        **backend_kwargs,
    )


@fixture(scope="module")
def image_data(request, operator):
    """Generate a random image."""
    if operator.uses_sense:
        shape = operator.shape
    else:
        shape = (operator.n_coils, *operator.shape)
    img = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    img += np.random.rand(*shape).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def kspace_data(request, operator):
    """Generate a random image."""
    shape = (operator.n_coils, operator.n_samples)
    kspace = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    kspace += np.random.rand(*shape).astype(operator.cpx_dtype)
    return kspace


def test_forward_perf(benchmark, operator, image_data):
    """Benchmark forward operation."""
    benchmark(operator.op, image_data)


def test_adjoint_perf(benchmark, operator, kspace_data):
    """Benchmark adjoint operation."""
    benchmark(operator.adj_op, kspace_data)
