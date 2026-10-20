#!/usr/bin/env bash
# Fetch mini-chameleon, install our kernels into it and build.
#
# Usage: ./setup.sh [target_dir]
#
# Environment variables:
#   UPSTREAM_URL     mini-chameleon repository (default: public gitlab.inria.fr repo)
#   UPSTREAM_COMMIT  upstream commit the patches apply to (default: 2a8b12b)
#   ENABLE_MPI       ON/OFF (default: ON if mpicc is found)
#   JOBS             parallel build jobs (default: 4)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${HERE}/mini-chameleon}"

UPSTREAM_URL="${UPSTREAM_URL:-https://gitlab.inria.fr/solverstack/mini-examples/mini-chameleon.git}"
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-2a8b12b098c9348af3bbf77166afbc15644a313f}"
JOBS="${JOBS:-4}"

if [[ -z "${ENABLE_MPI:-}" ]]; then
  if command -v mpicc > /dev/null; then ENABLE_MPI=ON; else ENABLE_MPI=OFF; fi
fi

if [[ ! -d "${TARGET}/.git" ]]; then
  git clone "${UPSTREAM_URL}" "${TARGET}"
fi

cd "${TARGET}"
git checkout --quiet --force "${UPSTREAM_COMMIT}"
git submodule update --init --recursive

# Install the kernels and register them in the build
cp "${HERE}"/kernels/*.c myblas/
git apply --whitespace=nowarn "${HERE}/patches/integration.patch"
git apply --whitespace=nowarn "${HERE}/patches/upstream-kernels.patch"

cmake -B build/default \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_MPI="${ENABLE_MPI}" \
      -DENABLE_STARPU=OFF \
      -DENABLE_CUDA=OFF \
      -DENABLE_MIPP=OFF
cmake --build build/default -j "${JOBS}"

echo "Build done in ${TARGET}/build/default (MPI=${ENABLE_MPI})"
