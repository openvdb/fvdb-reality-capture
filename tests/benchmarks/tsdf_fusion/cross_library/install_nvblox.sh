#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible nvblox_torch install into a dedicated `nvblox` conda
# env. We build from source because:
#
#   * `pip install nvblox_torch` doesn't work -- it's not on pypi.
#   * The wheel on NVIDIA's Isaac index targets CUDA 12 + torch<=2.9.1;
#     our `fvdb` env is on CUDA 13 + torch 2.10 and downgrading would
#     break fvdb.
#   * Running nvblox out-of-process from the benchmark driver (via
#     `nvblox_runner.py`) is actually desirable anyway: it isolates
#     nvblox's CUDA context + block-hash allocator pool from fvdb's,
#     so the two libraries can't fight over GPU memory.
#
# End state after this script:
#
#   conda env `nvblox` with python 3.11, CUDA 12.4, torch 2.6.0+cu124,
#   nvblox_torch 0.0.9 importable with `Mapper` + `Sensor.from_lidar`
#   working. Invoke via
#   `/home/fwilliams/bin/miniconda3/envs/nvblox/bin/python nvblox_runner.py`.
#
# Prereqs: a working miniconda + git. No root / sudo needed.
#
# Idempotent: skips work that's already done.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/fwilliams/bin/miniconda3}"
NVBLOX_SRC="${NVBLOX_SRC:-/tmp/nvblox}"
CUDA_STAGED="${CUDA_STAGED:-/tmp/cuda-root}"

echo "=== [1/5] Create conda env 'nvblox' with CUDA 12.4 ==="
if [[ ! -d "${CONDA_ROOT}/envs/nvblox" ]]; then
    "${CONDA_ROOT}/bin/conda" create -n nvblox -c conda-forge -c nvidia -y \
        python=3.11 cuda-toolkit=12.4 cuda-nvcc=12.4 cmake ninja git
fi
NVBLOX_ENV="${CONDA_ROOT}/envs/nvblox"

echo "=== [2/5] Install torch 2.6 + CUDA-12.4 wheel ==="
if ! "${NVBLOX_ENV}/bin/python" -c "import torch; assert torch.version.cuda.startswith('12')" 2>/dev/null; then
    "${NVBLOX_ENV}/bin/pip" install 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' \
        --index-url https://download.pytorch.org/whl/cu124
fi

echo "=== [3/5] Stage conda CUDA toolkit as a FindCUDA-friendly prefix ==="
# Torch's CMake uses the legacy FindCUDA module which wants
# ${CUDA_TOOLKIT_ROOT_DIR}/include/cuda_runtime.h at the top level.
# Conda installs headers under `targets/x86_64-linux/include/` and
# the prefix dir is read-only so we can't symlink into it. Stage a
# parallel layout in /tmp that points at the real conda paths.
if [[ ! -f "${CUDA_STAGED}/include/cuda_runtime.h" ]]; then
    rm -rf "${CUDA_STAGED}"
    mkdir -p "${CUDA_STAGED}/include"
    for f in "${NVBLOX_ENV}/targets/x86_64-linux/include/"*; do
        ln -sf "$f" "${CUDA_STAGED}/include/$(basename "$f")"
    done
    # nvblox's `#include <nvtx3/nvToolsExt.h>` needs an nvtx3 dir.
    # CUDA 12 dropped the old libnvToolsExt; the nvtx3 headers live
    # in the `nvidia.nvtx` pip subpackage that torch's install
    # dragged in. Point the staged include at that.
    ln -sf "${NVBLOX_ENV}/lib/python3.11/site-packages/nvidia/nvtx/include/nvtx3" \
        "${CUDA_STAGED}/include/nvtx3"
    ln -sf "${NVBLOX_ENV}/bin" "${CUDA_STAGED}/bin"
    ln -sf "${NVBLOX_ENV}/lib" "${CUDA_STAGED}/lib"
    ln -sf "${NVBLOX_ENV}/lib" "${CUDA_STAGED}/lib64"
    ln -sf "${NVBLOX_ENV}/nvvm" "${CUDA_STAGED}/nvvm"
fi

echo "=== [4/5] Clone + build nvblox C++ (static lib + pybind) ==="
if [[ ! -d "${NVBLOX_SRC}" ]]; then
    git clone --depth 1 https://github.com/nvidia-isaac/nvblox.git "${NVBLOX_SRC}"
fi
if [[ ! -f "${NVBLOX_SRC}/build/nvblox_torch/cpp/libpy_nvblox.so" ]]; then
    export PATH="${NVBLOX_ENV}/bin:${PATH}"
    export CMAKE_PREFIX_PATH="${NVBLOX_ENV}/lib/python3.11/site-packages/torch/share/cmake:${NVBLOX_ENV}"
    export CUDA_TOOLKIT_ROOT_DIR="${CUDA_STAGED}"
    export CUDAToolkit_ROOT="${CUDA_STAGED}"
    rm -rf "${NVBLOX_SRC}/build" && mkdir "${NVBLOX_SRC}/build" && cd "${NVBLOX_SRC}/build"
    # Build only `py_nvblox` (the pytorch binding) and its required
    # `libnvblox_lib.so` + `libnvblox_gpu_hash.a`. nvblox's
    # executables (load_map_and_mesh, fuse_*) fail to link without
    # -rpath-link wiring and we don't need them for the bench.
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DBUILD_PYTORCH_WRAPPER=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXPERIMENTS=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_STAGED}" \
        -DCUDAToolkit_ROOT="${CUDA_STAGED}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -Dnvtx3_dir="${CUDA_STAGED}/include/nvtx3" \
        -DPython_EXECUTABLE="${NVBLOX_ENV}/bin/python" \
        -DCMAKE_CXX_FLAGS="-I${CUDA_STAGED}/include" \
        -DCMAKE_CUDA_FLAGS="-I${CUDA_STAGED}/include"
    ninja py_nvblox
fi

echo "=== [5/5] Install nvblox_torch wheel (+ runtime deps) ==="
# nvblox_torch's `pyproject.toml` lists heavyweight runtime deps
# (open3d, timm, matplotlib, ...) most of which aren't needed for
# just TSDF fusion. Install with --no-deps, then add back only the
# handful that the `Mapper` import chain actually requires.
if ! "${NVBLOX_ENV}/bin/python" -c "import nvblox_torch" 2>/dev/null; then
    export CUDA_VERSION=12
    "${NVBLOX_ENV}/bin/pip" install --no-cache-dir --no-deps \
        transforms3d imageio opencv-python einops nvtx scipy scikit-learn \
        plotly dash flask werkzeug jinja2 markupsafe itsdangerous click \
        blinker retrying importlib-metadata jupyter_dash nbformat narwhals \
        threadpoolctl joblib pillow
    "${NVBLOX_ENV}/bin/pip" install --no-deps --no-cache-dir \
        "${NVBLOX_SRC}/nvblox_torch/"
    # open3d is the biggest single dep and needs its own transitive
    # set (plotly already above; open3d's _ml3d module imports
    # sklearn which we got, addict, ...). Resolving normally is
    # expensive; pin to the 0.18.x wheel.
    "${NVBLOX_ENV}/bin/pip" install --no-cache-dir 'open3d==0.18.*'
fi

echo "=== Verifying import ==="
"${NVBLOX_ENV}/bin/python" -c "
import nvblox_torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.sensor import Sensor
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
m = Mapper(voxel_sizes_m=[0.2], integrator_types=[ProjectiveIntegratorType.TSDF])
s = Sensor.from_lidar(1800, 64, 0.4712, 1.0)
print(f'OK -- nvblox_torch {nvblox_torch.__version__} with {m} and {s}')
"
echo "=== Done ==="
