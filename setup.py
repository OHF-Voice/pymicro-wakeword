"""Build configuration for pymicro-wakeword.

The TensorFlow Lite C library is a prebuilt binary, not something this project
compiles. It is vendored per platform under ``lib/<platform>/`` and copied into
the package at build time.

That copy used to happen only in ``script/copy_lib``, which the release workflow
runs. Any other builder — piwheels is the important one — built straight from
the sdist and therefore packaged whatever ``pymicro_wakeword/lib/`` happened to
contain, which was the x86-64 default. The resulting wheel was tagged for the
builder's platform but carried an x86-64 binary, so it installed and then failed
at ``ctypes.CDLL`` with "wrong ELF class".

Selecting the library in ``build_py`` instead means every builder gets the right
binary, and a platform with no vendored library fails at build time with a
message that says so, rather than shipping a wheel that cannot load.
"""

import platform
import shutil
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:
    # setuptools >= 70.1 ships the command; the standalone wheel package is
    # deprecated and warns when imported.
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

_PROGRAM_DIR = Path(__file__).parent
_SRC_LIB_DIR = _PROGRAM_DIR / "lib"
_DEST_LIB_DIR = _PROGRAM_DIR / "pymicro_wakeword" / "lib"

# Vendored libraries, keyed by the value of platform.machine() they run on.
#
# armv6l must not fall through to linux_armv7l. The armv7l library is built for
# armv7-a + neon-vfpv4, and ARM1176 (Pi Zero W, Pi 1) has neither, so loading it
# raises SIGILL rather than a clear error. It has its own library instead.
_MACHINE_TO_PLATFORM = {
    "x86_64": "linux_amd64",
    "amd64": "linux_amd64",
    "aarch64": "linux_arm64",
    "arm64": "linux_arm64",
    "armv6l": "linux_armv6l",
    "armv7l": "linux_armv7l",
    "armv8l": "linux_armv7l",
    "armhf": "linux_armv7l",
    "arm": "linux_armv7l",
}


def target_platform(machine=None, system=None):
    """The vendored library directory for this build target, or None.

    ``machine`` and ``system`` default to the running interpreter's, and exist
    so the mapping can be tested without the hardware.
    """
    machine = (machine or platform.machine()).lower()
    system = system or sys.platform

    if system.startswith("linux"):
        return _MACHINE_TO_PLATFORM.get(machine)

    if system == "win32" and machine in ("x86_64", "amd64"):
        return "windows_amd64"

    if system == "darwin" and machine in ("aarch64", "arm64"):
        return "darwin_arm64"

    return None


def select_library(machine=None, system=None, src_lib_dir=None, dest_lib_dir=None):
    """Copy the vendored library for this target into the package.

    Returns the destination directory. Raises SystemExit with an actionable
    message when the target has no vendored library, because a wheel that
    installs and then cannot load is worse than a build that stops.
    """
    src_lib_dir = Path(src_lib_dir or _SRC_LIB_DIR)
    dest_lib_dir = Path(dest_lib_dir or _DEST_LIB_DIR)
    machine = (machine or platform.machine()).lower()
    system = system or sys.platform

    platform_name = target_platform(machine, system)
    if platform_name is None:
        available = (
            ", ".join(sorted(p.name for p in src_lib_dir.iterdir() if p.is_dir()))
            if src_lib_dir.is_dir()
            else "none"
        )
        raise SystemExit(
            f"No libtensorflowlite_c library for {machine!r} ({system}). "
            f"Vendored: {available}. "
            "Build one and pass libtensorflowlite_c_path= to "
            "MicroWakeWord.from_config(), or install on a supported platform."
        )

    source = src_lib_dir / platform_name
    if not source.is_dir():
        raise SystemExit(
            f"lib/{platform_name} is missing from this source tree. "
            "The sdist should contain every lib/<platform>/ directory."
        )

    if dest_lib_dir.exists():
        shutil.rmtree(dest_lib_dir)
    shutil.copytree(source, dest_lib_dir)
    return dest_lib_dir


class PlatformLibraryBuildPy(build_py):
    """Populate pymicro_wakeword/lib/ before package data is collected."""

    def run(self):
        select_library()
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # platform wheel
        return True


class fixed_tag_bdist_wheel(_bdist_wheel):
    def get_tag(self):
        py, abi, plat = super().get_tag()
        print(f"[bdist_wheel.get_tag] computed: python={py} abi={abi} plat={plat}")
        # Force version-agnostic tag for Python 3
        return ("py3", "none", plat)


if __name__ == "__main__":
    setup(
        distclass=BinaryDistribution,
        cmdclass={
            "bdist_wheel": fixed_tag_bdist_wheel,
            "build_py": PlatformLibraryBuildPy,
        },
    )
