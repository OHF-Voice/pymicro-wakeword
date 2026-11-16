#!/bin/bash
# Cross-build libtensorflowlite_c.so for ARMv6 (ARM1176, Pi Zero W / Pi 1).
#
# Run inside a Debian bookworm container, after make-armv6-sysroot.sh:
#
#   podman run --rm -v "$PWD":/io:Z -w /io debian:bookworm bash -c '
#     apt-get update && apt-get install -y gcc-arm-linux-gnueabihf \
#       g++-arm-linux-gnueabihf cmake ninja-build git python3 file unzip curl \
#       binutils-arm-linux-gnueabihf
#     bash /io/script/make-armv6-sysroot.sh
#     bash /io/script/build-tflite-armv6.sh'
#
# The result is written to lib/linux_armv6l/libtensorflowlite_c.so.
#
# Three flags matter, and all are needed:
#
# 1. `-marm`. ARMv6's Thumb mode is Thumb-1, which cannot use the hard-float
#    VFP ABI, so the toolchain's Thumb default fails to compile even a trivial
#    test program ("sorry, unimplemented: Thumb-1 'hard-float' VFP ABI").
#
# 2. `--sysroot` pointing at the Raspbian bookworm armv6 sysroot. Debian's
#    armhf crt objects and libgcc are armv7-a and contain Thumb-2 and
#    `movw`/`movt`, which ARM1176 cannot execute. The sysroot also sets the
#    glibc floor to 2.36 / GLIBCXX_3.4.30.
#
# 3. `-march=armv6k`. ARM1176 implements ARMv6K, which adds
#    `ldrexb`/`strexb`/`ldrexd`/`strexd`. With plain `armv6` GCC emits libcalls
#    such as `__sync_val_compare_and_swap_1` and `__atomic_load_8`, which
#    Raspbian's armv6 libgcc does not provide, so the library links but fails
#    to load.
set -euxo pipefail

TF_VERSION="${TF_VERSION:-v2.17.1}"
SYSROOT="${SYSROOT:-/io/armv6-sysroot-bookworm}"
OUT="${OUT:-/io/lib/linux_armv6l/libtensorflowlite_c.so}"
WORK="${WORK:-/io/.build-armv6}"

GCCDIR="$SYSROOT/usr/lib/gcc/arm-linux-gnueabihf/12"
LIBDIR="$SYSROOT/usr/lib/arm-linux-gnueabihf"
FLAGS="-O3 -marm -march=armv6k -mfpu=vfp -mfloat-abi=hard --sysroot=$SYSROOT -B$GCCDIR/ -B$LIBDIR/"

mkdir -p "$WORK"
cd "$WORK"

if [ ! -d tensorflow ]; then
  git clone --depth 1 --branch "$TF_VERSION" https://github.com/tensorflow/tensorflow.git
fi

cd tensorflow
rm -rf build-armv6
mkdir -p build-armv6
cd build-armv6

cmake ../tensorflow/lite/c \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=armv6l \
  -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
  -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
  -DCMAKE_C_FLAGS="$FLAGS" \
  -DCMAKE_CXX_FLAGS="$FLAGS" \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_EXTERNAL_DELEGATE=OFF \
  -DTFLITE_ENABLE_NNAPI=OFF \
  -DTFLITE_ENABLE_MMAP=OFF

ninja -j"$(nproc)"

mkdir -p "$(dirname "$OUT")"
cp libtensorflowlite_c.so "$OUT"

echo "=== result ==="
arm-linux-gnueabihf-readelf -A "$OUT" | grep -E "Tag_CPU_arch:|Tag_FP_arch|Tag_Advanced_SIMD" | sort -u
arm-linux-gnueabihf-objdump -T "$OUT" | grep -oE "GLIBC_[0-9.]+|GLIBCXX_[0-9.]+" | sort -uV | tail -4
