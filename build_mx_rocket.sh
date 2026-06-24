#!/usr/bin/env bash

# Build the gemmini-rocc-tests for the standalone MxGemminiRocketConfig.
# Adds -DMX_ROCKET so the dual-mode tests take the standalone path (real RoCC instead of the
# MMIO command mimic, flat scale-factor window, scratchpad mvout instead of shared memory).
# Outputs land in build_mx_rocket/ (e.g. build_mx_rocket/bareMetalC/<test>-baremetal).
#
# Usage: ./build_mx_rocket.sh [make targets...]   (targets are top-level dirs, like build.sh)
#   ./build_mx_rocket.sh bareMetalC    # build all bareMetalC tests (incl. the converted MX ones)
#   ./build_mx_rocket.sh               # build everything
# To rebuild a single test after the first run:
#   ( cd build_mx_rocket/bareMetalC && make -f .../bareMetalC/Makefile EXTRA_CFLAGS=-DMX_ROCKET \
#       matmul_tiled_fp8_64x64_requant-baremetal )

if [ ! -d "build_mx_rocket" ] ; then
    autoconf && \
        mkdir build_mx_rocket && cd build_mx_rocket && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build_mx_rocket

if [[ $(which riscv64-unknown-linux-gnu-gcc) ]] ; then
    make -j EXTRA_CFLAGS=-DMX_ROCKET $@
else
    make -j EXTRA_CFLAGS=-DMX_ROCKET BAREMETAL_ONLY=1 $@
fi
