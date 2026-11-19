"""Test cases, as plain functions usable as fixture or test parameters."""


def cases_of(cases, *names):
    """Return the ``case_*`` methods of a cases class, in definition order.

    Parameters
    ----------
    cases : type
        Class holding ``case_<name>`` methods, which return the case data.
    names : str
        Only return these cases (``<name>``, without the ``case_`` prefix).
    """
    instance = cases()
    all_names = [n.removeprefix("case_") for n in vars(cases) if n.startswith("case_")]
    if unknown := set(names) - set(all_names):
        raise ValueError(f"Unknown cases {unknown} in {cases.__name__}.")
    return [getattr(instance, f"case_{n}") for n in (names or all_names)]


def case_id(case):
    """Return the test id of a case: its name, without the ``case_`` prefix."""
    return case.__name__.removeprefix("case_")
