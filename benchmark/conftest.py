"""Configuration for pytest benchmark."""


def pytest_addoption(parser):
    """Add options to pytest."""
    parser.addoption(
        "--res",
        action="store",
        nargs="3",
        type=int,
        default=(192, 192, 128),
        help="resolution of image domain",
    )
    parser.addoption(
        "--ncoils",
        action="store",
        type=int,
        default=8,
        help="number of coil",
    )
