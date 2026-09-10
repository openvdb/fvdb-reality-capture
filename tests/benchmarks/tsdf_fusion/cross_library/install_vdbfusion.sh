#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible VDBFusion install into the `fvdb` conda env.
#
# Why this script exists: `pip install vdbfusion` fails on any modern
# Linux toolchain because VDBFusion vendors c-blosc 1.5.0 + TBB 2018
# as ExternalProject_Add trees and those trees do not build under
# gcc-13+ or CMake >= 4 without extensive patches. This script
# sidesteps the vendored stack entirely by:
#
#   1. Cloning OpenVDB 12 and compiling it against the `fvdb` conda
#      env's TBB / Blosc / Boost (all of which are already modern).
#   2. Cloning Eigen 3.4 (headers only).
#   3. Cloning VDBFusion and applying a minimal CMake patch to make
#      it find our freshly-built OpenVDB and explicitly link the
#      transitive TBB / Blosc / Boost dependencies that OpenVDB's
#      FindOpenVDB.cmake does not propagate through OpenVDB::openvdb.
#   4. `pip install`-ing the patched VDBFusion into `fvdb`.
#
# Result: `import vdbfusion` works inside the `fvdb` env with no
# toolchain conflicts. Works on Ubuntu 24.04 / gcc-13 / CMake 4.3 /
# Python 3.12 as of 2026-04-22.
#
# Usage:
#   bash tests/benchmarks/tsdf_fusion/cross_library/install_vdbfusion.sh
#
# Idempotent: re-running with existing /tmp state just rebuilds.

set -euo pipefail

FVDB_ENV="${FVDB_ENV:-/home/fwilliams/bin/miniconda3/envs/fvdb}"
OPENVDB_SRC="${OPENVDB_SRC:-/tmp/openvdb-build/src}"
OPENVDB_BUILD="${OPENVDB_BUILD:-/tmp/openvdb-build/build}"
OPENVDB_PREFIX="${OPENVDB_PREFIX:-/tmp/openvdb-prefix}"
EIGEN_SRC="${EIGEN_SRC:-/tmp/openvdb-build/eigen-src}"
EIGEN_PREFIX="${EIGEN_PREFIX:-/tmp/eigen-prefix}"
VDBFUSION_SRC="${VDBFUSION_SRC:-/tmp/vdbfusion_src}"

export PATH="${FVDB_ENV}/bin:${PATH}"

echo "=== [1/4] Clone OpenVDB 12 + Eigen ==="
if [[ ! -d "${OPENVDB_SRC}" ]]; then
    mkdir -p "$(dirname "${OPENVDB_SRC}")"
    git clone --depth 1 --branch v12.0.1 \
        https://github.com/AcademySoftwareFoundation/openvdb.git "${OPENVDB_SRC}"
fi
if [[ ! -d "${EIGEN_SRC}" ]]; then
    git clone --depth 1 --branch 3.4 \
        https://gitlab.com/libeigen/eigen.git "${EIGEN_SRC}"
fi

echo "=== [2/4] Build OpenVDB 12 against ${FVDB_ENV} ==="
if [[ ! -f "${OPENVDB_PREFIX}/lib/libopenvdb.a" ]]; then
    rm -rf "${OPENVDB_BUILD}"
    mkdir -p "${OPENVDB_BUILD}"
    cd "${OPENVDB_BUILD}"
    cmake "${OPENVDB_SRC}" \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${OPENVDB_PREFIX}" \
        -DCMAKE_PREFIX_PATH="${FVDB_ENV}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_STANDARD=17 \
        -DOPENVDB_BUILD_CORE=ON \
        -DOPENVDB_BUILD_BINARIES=OFF \
        -DOPENVDB_BUILD_DOCS=OFF \
        -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
        -DOPENVDB_BUILD_NANOVDB=OFF \
        -DUSE_BLOSC=ON \
        -DUSE_ZLIB=ON \
        -DBUILD_SHARED_LIBS=OFF
    ninja -j "$(nproc)"
    ninja install
fi

echo "=== [3/4] Install Eigen 3.4 headers ==="
if [[ ! -d "${EIGEN_PREFIX}/include/eigen3" ]]; then
    cd "${EIGEN_SRC}"
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX="${EIGEN_PREFIX}"
    cmake --install build
fi

echo "=== [4/4] Clone + patch + install VDBFusion ==="
if [[ ! -d "${VDBFUSION_SRC}" ]]; then
    git clone --depth 1 https://github.com/PRBonn/vdbfusion.git "${VDBFUSION_SRC}"
fi

# Apply two minimal patches.
#
# Patch 1: the main vdbfusion lib compiles fine against OpenVDB 12 but
# hits the gcc-13 `-Werror` wall against the ancient TBB warning style
# that OpenVDB headers still emit. Relax the per-target options.
#
# Patch 2: OpenVDB's `FindOpenVDB.cmake` does not propagate TBB /
# Blosc / Boost.iostreams / ZLIB through `OpenVDB::openvdb`'s public
# interface (they are compile-time-only deps there), so static-built
# libopenvdb.a leaves unresolved TBB symbols in the pybind .so,
# causing `ImportError: undefined symbol _ZN3tbb6detail2r15spawn...`
# at import time. Explicitly list them as PUBLIC deps, guarded by
# `if(TARGET ...)` so we don't double-define TBB::tbb when CMake
# config-mode loading also brings it in.
cd "${VDBFUSION_SRC}"
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/vdbfusion/vdbfusion/CMakeLists.txt")
text = p.read_text()
if "-Wno-changes-meaning" not in text:
    text = text.replace(
        "target_compile_options(vdbfusion PRIVATE -Wall -Wextra)",
        "target_compile_options(vdbfusion PRIVATE\n"
        "    -Wall -Wextra\n"
        "    -Wno-error -Wno-changes-meaning -Wno-template-id-cdtor\n"
        "    -Wno-deprecated-declarations -Wno-deprecated\n"
        "    -Wno-class-memaccess -Wno-cast-user-defined\n"
        "    -Wno-missing-template-keyword -Wno-narrowing)",
    )
if "PATCH: TBB transitive link" not in text:
    text = text.replace(
        "target_link_libraries(vdbfusion PUBLIC Eigen3::Eigen OpenVDB::openvdb)",
        "target_link_libraries(vdbfusion PUBLIC Eigen3::Eigen OpenVDB::openvdb)\n\n"
        "# PATCH: TBB transitive link -- see install_vdbfusion.sh for why.\n"
        "foreach(_dep IN ITEMS TBB::tbb TBB::tbbmalloc Blosc::blosc Boost::iostreams ZLIB::ZLIB)\n"
        "  if(TARGET ${_dep})\n"
        "    target_link_libraries(vdbfusion PUBLIC ${_dep})\n"
        "  endif()\n"
        "endforeach()",
    )
p.write_text(text)

q = pathlib.Path("src/vdbfusion/pybind/CMakeLists.txt")
text = q.read_text()
if "-Wno-changes-meaning" not in text:
    text = text.replace(
        "target_compile_options(vdbfusion_pybind PRIVATE -Werror -Wall -Wextra)",
        "target_compile_options(vdbfusion_pybind PRIVATE\n"
        "    -Wall -Wextra\n"
        "    -Wno-error -Wno-changes-meaning -Wno-template-id-cdtor\n"
        "    -Wno-deprecated-declarations -Wno-deprecated\n"
        "    -Wno-class-memaccess -Wno-cast-user-defined\n"
        "    -Wno-missing-template-keyword -Wno-narrowing)",
    )
q.write_text(text)
print("Patches applied.")
PY

rm -rf "${VDBFUSION_SRC}/build"
CMAKE_ARGS="-DUSE_SYSTEM_OPENVDB=ON -DUSE_SYSTEM_EIGEN3=ON -DUSE_SYSTEM_PYBIND11=ON -DCMAKE_PREFIX_PATH=${OPENVDB_PREFIX};${EIGEN_PREFIX};${FVDB_ENV} -DCMAKE_MODULE_PATH=${OPENVDB_PREFIX}/lib/cmake/OpenVDB" \
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
"${FVDB_ENV}/bin/pip" install "${VDBFUSION_SRC}"

echo "=== Verifying import ==="
"${FVDB_ENV}/bin/python" -c "
import vdbfusion, numpy as np
v = vdbfusion.VDBVolume(voxel_size=0.2, sdf_trunc=0.6, space_carving=True)
v.integrate(np.random.randn(1000, 3).astype(np.float64), np.zeros(3))
verts, tris = v.extract_triangle_mesh(min_weight=0.1)
print(f'OK -- VDBFusion integrates and meshes ({verts.shape[0]} verts, {tris.shape[0]} tris).')
"
echo "=== Done ==="
