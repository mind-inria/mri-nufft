"""Release helpers for mri-nufft.

``prepare X.Y.Z``
    Check that no deprecated object is due for removal in ``X.Y.Z``, then turn the
    *Unreleased* section of the changelog into the ``X.Y.Z`` one, and open a new
    *Unreleased* section.
``notes X.Y.Z``
    Print the ``X.Y.Z`` section of the changelog in Markdown, for the GitHub
    release notes.

``notes`` only needs the standard library: the release workflow runs it without
installing the package.
"""

import argparse
import datetime
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "docs" / "changelog.rst"
UNRELEASED = "Unreleased"


def _release_version(text):
    if not re.fullmatch(r"\d+\.\d+\.\d+", text):
        raise argparse.ArgumentTypeError(f"expected X.Y.Z, got {text!r}")
    return text


def _as_tuple(version):
    return tuple(int(n) for n in version.split("."))


def _sections(lines):
    """Return ``(title, start, end)`` for each release section of the changelog."""
    starts = [
        i
        for i in range(len(lines) - 1)
        if lines[i]
        and not lines[i][0].isspace()
        and re.fullmatch(r"-{3,}", lines[i + 1])
    ]
    ends = [*starts[1:], len(lines)]
    return [(lines[i], i, end) for i, end in zip(starts, ends)]


def _to_markdown(lines):
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.fullmatch(r"\^{3,}", stripped):
            continue
        if i + 1 < len(lines) and re.fullmatch(r"\^{3,}", lines[i + 1].strip()):
            out.append(f"## {stripped}")
        elif (
            stripped
            and out
            and out[-1].strip()
            and not out[-1].startswith("#")
            and not re.match(r"\s*- ", line)
        ):
            # GitHub renders the line breaks of a release description.
            out[-1] += " " + stripped
        else:
            out.append(line.rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

    text = re.sub(r":gh:`(\d+)`", r"#\1", text)
    text = re.sub(r":ref:`([^`<]+?)\s*<[^`>]+>`", r"\1", text)
    # Like Sphinx, a leading ``~`` only shows the last component.
    text = re.sub(r":\w+:`~[\w.]*?(\w+)`", r"`\1`", text)
    text = re.sub(r":\w+:`([^`]+)`", r"`\1`", text)
    text = re.sub(r"`([^`<]+?)\s*<([^>]+)>`__?", r"[\1](\2)", text)
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def _check_deprecations(version):
    if importlib.util.find_spec("pytest") is None:
        sys.exit("pytest is missing: run `uv run --extra test tools/release.py ...`.")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_deprecation.py::test_no_expired_deprecation",
        ],
        cwd=ROOT,
        env={**os.environ, "MRINUFFT_RELEASE": version},
    )
    if result.returncode:
        sys.exit(f"Remove the deprecated objects due in {version} first.")


def prepare(version):
    """Turn the *Unreleased* changelog section into the ``version`` one."""
    lines = CHANGELOG.read_text().splitlines()
    sections = _sections(lines)
    if not sections or sections[0][0] != UNRELEASED:
        sys.exit(f"The first section of {CHANGELOG} must be {UNRELEASED!r}.")
    _, start, end = sections[0]
    if not "".join(lines[start + 2 : end]).strip():
        sys.exit(f"The {UNRELEASED!r} section of {CHANGELOG} is empty.")
    if len(sections) > 1:
        latest = sections[1][0].split()[0]
        if _as_tuple(version) <= _as_tuple(latest):
            sys.exit(f"{version} does not come after the latest release, {latest}.")

    _check_deprecations(version)

    heading = f"{version} ({datetime.date.today().isoformat()})"
    lines[start : start + 2] = [
        UNRELEASED,
        "-" * len(UNRELEASED),
        "",
        heading,
        "-" * len(heading),
    ]
    CHANGELOG.write_text("\n".join(lines) + "\n")
    print(f"{CHANGELOG.relative_to(ROOT)}: {UNRELEASED!r} is now {heading!r}.")


def notes(version):
    """Print the ``version`` changelog section in Markdown."""
    lines = CHANGELOG.read_text().splitlines()
    for title, start, end in _sections(lines):
        if title.split()[0] == version:
            text = _to_markdown(lines[start + 2 : end])
            if not text:
                sys.exit(f"The {version} section of {CHANGELOG} is empty.")
            print(text)
            return
    sys.exit(f"{CHANGELOG} has no {version} section.")


def main():
    """Run the command given on the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["prepare", "notes"])
    parser.add_argument("version", type=_release_version)
    args = parser.parse_args()
    {"prepare": prepare, "notes": notes}[args.command](args.version)


if __name__ == "__main__":
    main()
