#!/bin/bash
# Build a Raspbian bookworm (armv6) sysroot from the Raspbian archive.
#
# Used by build-tflite-armv6.sh. Run it inside a Debian bookworm container:
#
#   podman run --rm -v "$PWD":/io:Z -w /io debian:bookworm \
#     bash -c 'apt-get update && apt-get install -y binutils-arm-linux-gnueabihf
#              && bash /io/script/make-armv6-sysroot.sh'
#
# Why a sysroot at all: Debian's armhf crt objects and libgcc are built for
# armv7-a. `crtbeginS.o` contains Thumb-2 in `register_tm_clones`, which is an
# illegal instruction on ARM1176, and libgcc's `__sync_*` helpers contain
# `movw`/`movt`. Raspbian armhf is built for ARMv6, so its files are correct.
#
# Why bookworm and not trixie: the sysroot sets the glibc floor. Trixie gives
# GLIBC_2.38 / GLIBCXX_3.4.32, which excludes Raspbian bookworm (glibc 2.36,
# GCC 12). Bookworm gives GLIBC_2.36 / GLIBCXX_3.4.30, matching linux_armv7l's
# intent of running on bookworm and newer.
set -euxo pipefail

SYSROOT="${SYSROOT:-/io/armv6-sysroot-bookworm}"
DEBS="${DEBS:-/tmp/raspbian-debs}"

rm -rf "$SYSROOT" "$DEBS"
mkdir -p "$SYSROOT" "$DEBS"

dpkg --add-architecture armhf
cat > /etc/apt/sources.list.d/raspbian.list <<'EOF'
deb [trusted=yes arch=armhf] http://raspbian.raspberrypi.com/raspbian/ bookworm main
EOF
apt-get update -qq

cd "$DEBS"
apt-get download \
  libc6:armhf \
  libc6-dev:armhf \
  libgcc-s1:armhf \
  libgcc-12-dev:armhf \
  libstdc++6:armhf \
  libstdc++-12-dev:armhf \
  gcc-12-base:armhf \
  linux-libc-dev:armhf

for deb in *.deb; do
  dpkg-deb -x "$deb" "$SYSROOT"
done

echo "=== crt objects (must be Tag_CPU_arch: v6) ==="
for f in "$SYSROOT"/usr/lib/arm-linux-gnueabihf/crti.o \
         "$SYSROOT"/usr/lib/arm-linux-gnueabihf/crtn.o \
         "$SYSROOT"/usr/lib/gcc/arm-linux-gnueabihf/12/crtbeginS.o \
         "$SYSROOT"/usr/lib/gcc/arm-linux-gnueabihf/12/crtendS.o; do
  printf '%-70s ' "$f"
  arm-linux-gnueabihf-readelf -A "$f" 2>/dev/null | grep "Tag_CPU_arch:" | head -1 || echo MISSING
done

echo "=== libgcc.a ==="
arm-linux-gnueabihf-readelf -A "$SYSROOT"/usr/lib/gcc/arm-linux-gnueabihf/12/libgcc.a 2>/dev/null \
  | grep "Tag_CPU_arch:" | sort | uniq -c
