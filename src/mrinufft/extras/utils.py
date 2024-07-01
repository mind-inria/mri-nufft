"""Utils for extras module."""

from mrinufft._utils import MethodRegister

register_smaps = MethodRegister("sensitivity_maps")


def get_smaps(name, *args, **kwargs):
    """Get the sensitivity_maps function from its name."""
    try:
        method = register_smaps.registry["sensitivity_maps"][name]
    except KeyError as e:
        raise ValueError(
            f"Unknown sensitivity_maps method {name}. Available methods are \n"
            f"{list(register_smaps.registry['sensitivity_maps'].keys())}"
        ) from e

    if args or kwargs:
        return method(*args, **kwargs)
    return method
