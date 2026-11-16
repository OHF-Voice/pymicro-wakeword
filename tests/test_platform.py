"""Tests for build-time library selection and runtime platform detection.

These do not need the TensorFlow Lite library to load; they check that the
right one is chosen, and that a platform with no vendored library fails loudly
instead of producing a wheel that cannot load.
"""

import importlib.util
import platform
from pathlib import Path

import pytest

from pymicro_wakeword.wakeword import get_platform

_PROGRAM_DIR = Path(__file__).parent.parent


def _load_setup():
    """Import setup.py without running setup()."""
    spec = importlib.util.spec_from_file_location(
        "pymicro_wakeword_setup", _PROGRAM_DIR / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_setup = _load_setup()


@pytest.mark.parametrize(
    "machine,system,expected",
    [
        ("x86_64", "linux", "linux_amd64"),
        ("amd64", "linux", "linux_amd64"),
        ("aarch64", "linux", "linux_arm64"),
        ("arm64", "linux", "linux_arm64"),
        ("armv6l", "linux", "linux_armv6l"),
        ("armv7l", "linux", "linux_armv7l"),
        ("armv8l", "linux", "linux_armv7l"),
        ("armhf", "linux", "linux_armv7l"),
        ("AMD64", "win32", "windows_amd64"),
        ("arm64", "darwin", "darwin_arm64"),
        # No vendored library.
        ("i686", "linux", None),
        ("x86_64", "darwin", None),
    ],
)
def test_target_platform(machine, system, expected):
    assert _setup.target_platform(machine, system) == expected


def test_select_library_copies_the_target(tmp_path):
    src = tmp_path / "lib"
    for name in ("linux_amd64", "linux_armv7l"):
        (src / name).mkdir(parents=True)
        (src / name / "libtensorflowlite_c.so").write_bytes(name.encode())

    dest = tmp_path / "pymicro_wakeword" / "lib"
    _setup.select_library("armv7l", "linux", src, dest)

    assert (dest / "libtensorflowlite_c.so").read_bytes() == b"linux_armv7l"


def test_select_library_replaces_a_stale_default(tmp_path):
    """A leftover x86-64 default must not survive the build."""
    src = tmp_path / "lib"
    (src / "linux_armv7l").mkdir(parents=True)
    (src / "linux_armv7l" / "libtensorflowlite_c.so").write_bytes(b"armv7l")

    dest = tmp_path / "pymicro_wakeword" / "lib"
    dest.mkdir(parents=True)
    (dest / "libtensorflowlite_c.so").write_bytes(b"x86-64")
    (dest / "stale.txt").write_text("left over")

    _setup.select_library("armv7l", "linux", src, dest)

    assert (dest / "libtensorflowlite_c.so").read_bytes() == b"armv7l"
    assert not (dest / "stale.txt").exists()


def test_select_library_uses_the_armv6_library(tmp_path):
    """armv6l must get its own library, never the armv7l one.

    The armv7l library needs armv7-a + neon-vfpv4, which ARM1176 does not have,
    so falling through to it would produce a wheel that raises SIGILL on import.
    """
    src = tmp_path / "lib"
    for name in ("linux_armv6l", "linux_armv7l"):
        (src / name).mkdir(parents=True)
        (src / name / "libtensorflowlite_c.so").write_bytes(name.encode())

    dest = tmp_path / "pymicro_wakeword" / "lib"
    _setup.select_library("armv6l", "linux", src, dest)

    assert (dest / "libtensorflowlite_c.so").read_bytes() == b"linux_armv6l"


def test_select_library_fails_when_the_directory_is_missing(tmp_path):
    src = tmp_path / "lib"
    src.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        _setup.select_library("x86_64", "linux", src, tmp_path / "dest")

    assert "linux_amd64" in str(excinfo.value)


@pytest.mark.parametrize(
    "machine,expected",
    [
        ("x86_64", "linux_amd64"),
        ("aarch64", "linux_arm64"),
        ("armv6l", "linux_armv6l"),
        ("armv7l", "linux_armv7l"),
        ("armv8l", "linux_armv7l"),
    ],
)
def test_get_platform(monkeypatch, machine, expected):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr("sys.platform", "linux")
    assert get_platform() == expected
