#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Run the nightly comparative benchmark locally using the same container and
# environment setup as CI. This mirrors the GitHub Actions nightly workflow
# step-for-step so that CI failures can be reproduced locally.
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit (nvidia-docker2 / nvidia-container-toolkit)
#   - fvdb-core source tree (auto-detected or set FVDB_CORE_DIR)
#   - mipnerf360 dataset (auto-detected or set DATA_DIR; downloaded if missing)
#
# Usage:
#   cd tests/benchmarks/comparative
#   ./run_nightly_local.sh                              # full nightly benchmark
#   ./run_nightly_local.sh --matrix smoke_test_matrix.yml  # quick smoke test
#   ./run_nightly_local.sh --plot-only                  # regenerate plots from existing results
#
# Environment variables:
#   FVDB_CORE_DIR   - path to fvdb-core checkout (default: auto-detected)
#   DATA_DIR        - path to dataset directory (default: auto-detected)
#   GSPLAT_DIR      - path to gsplat checkout (default: auto-detected; cloned in container if unset)
#   MATRIX          - matrix YAML to use (default: nightly_matrix.yml)
#   CUDA_ARCH_LIST  - override TORCH_CUDA_ARCH_LIST (default: auto-detected from GPU)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Auto-detect fvdb-core
# ---------------------------------------------------------------------------
FVDB_CORE_DIR="${FVDB_CORE_DIR:-}"
if [[ -z "${FVDB_CORE_DIR}" ]]; then
    for candidate in \
        "${REPO_ROOT}/../fvdb-core" \
        "${REPO_ROOT}/../../fvdb-core" \
        ; do
        if [[ -f "${candidate}/build.sh" ]]; then
            FVDB_CORE_DIR="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [[ -z "${FVDB_CORE_DIR}" || ! -f "${FVDB_CORE_DIR}/build.sh" ]]; then
    echo "ERROR: Could not find fvdb-core (looked for build.sh). Set FVDB_CORE_DIR." >&2
    exit 1
fi
echo "fvdb-core:            ${FVDB_CORE_DIR}"

# ---------------------------------------------------------------------------
# Auto-detect data directory (needs 360_v2/ inside it)
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" ]]; then
    for candidate in \
        "${REPO_ROOT}/data" \
        "${REPO_ROOT}/../fvdb-reality-capture/data" \
        "${REPO_ROOT}/../data" \
        "${REPO_ROOT}/../../data" \
        ; do
        if [[ -d "${candidate}/360_v2" ]]; then
            DATA_DIR="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [[ -z "${DATA_DIR}" ]]; then
    DATA_DIR="${REPO_ROOT}/data"
    echo "No existing data directory found. Will download to: ${DATA_DIR}"
fi
DATA_DIR="$(cd "${DATA_DIR}" 2>/dev/null && pwd || echo "${DATA_DIR}")"
mkdir -p "${DATA_DIR}"
echo "Data directory:       ${DATA_DIR}"

# ---------------------------------------------------------------------------
# Auto-detect GSplat (optional -- cloned inside container if not found)
# ---------------------------------------------------------------------------
GSPLAT_DIR="${GSPLAT_DIR:-}"
if [[ -z "${GSPLAT_DIR}" ]]; then
    for candidate in \
        "${REPO_ROOT}/../gsplat" \
        "${REPO_ROOT}/../../gsplat" \
        ; do
        if [[ -d "${candidate}/gsplat" && -f "${candidate}/setup.py" ]] || \
           [[ -d "${candidate}/gsplat" && -f "${candidate}/pyproject.toml" ]]; then
            GSPLAT_DIR="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
GSPLAT_MOUNT=""
GSPLAT_CLONE_CMD=""
if [[ -n "${GSPLAT_DIR}" && -d "${GSPLAT_DIR}" ]]; then
    echo "GSplat (host mount):  ${GSPLAT_DIR}"
    GSPLAT_MOUNT="-v ${GSPLAT_DIR}:/workspace/gsplat:rw"
else
    echo "GSplat:               (will clone inside container)"
    GSPLAT_CLONE_CMD=$(cat <<'CLONE'
echo '=== Cloning GSplat ==='
GSPLAT_COMMIT="b60e917c95afc449c5be33a634f1f457e116ff5e"
git clone https://github.com/nerfstudio-project/gsplat.git /workspace/gsplat
cd /workspace/gsplat && git checkout "${GSPLAT_COMMIT}" && git submodule update --init --recursive
echo "GSplat pinned to commit: $(git rev-parse HEAD)"
CLONE
)
fi

# Parse CLI arguments: pull --matrix out, pass the rest through
MATRIX="${MATRIX:-nightly_matrix.yml}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
echo "Matrix:               ${MATRIX}"
echo "Extra args:           ${EXTRA_ARGS[*]:-<none>}"

CONTAINER_IMAGE="aswf/ci-openvdb:2024-clang17.2"
CONTAINER_NAME="fvdb-nightly-benchmark-$$"
ENV_FILE="/workspace/fvdb-reality-capture/tests/benchmarks/comparative/docker/benchmark_environment.yml"

echo "Container image:      ${CONTAINER_IMAGE}"
echo ""

# Write the inner script to a temp file to avoid quoting hell
INNER_SCRIPT=$(mktemp /tmp/fvdb-benchmark-inner.XXXXXX.sh)
trap 'rm -f "${INNER_SCRIPT}"' EXIT

cat > "${INNER_SCRIPT}" <<'INNER'
#!/usr/bin/env bash
set -e

ENV_FILE="$1"; shift
MATRIX="$1"; shift
GSPLAT_CLONE_CMD="$1"; shift
EXTRA_ARGS="$*"

echo "=== Setting up micromamba environment ==="
MICROMAMBA_VERSION="2.5.0-2"
MICROMAMBA_SHA256="c04571cfb0750e5432d530a3068b8fcd232ebed3133358e056e59a90b9852b00"
if ! command -v micromamba &>/dev/null; then
    curl -fsSL -o /tmp/micromamba \
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64"
    echo "${MICROMAMBA_SHA256}  /tmp/micromamba" | sha256sum -c -
    install -m 0755 /tmp/micromamba /usr/local/bin/micromamba
    rm -f /tmp/micromamba
fi
eval "$(micromamba shell hook -s bash)"
export CONDA_OVERRIDE_CUDA=12.9
micromamba create -n benchmark -f "${ENV_FILE}" -y
micromamba activate benchmark

echo "=== Detecting GPU architecture ==="
archs=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | awk NF | sort -u | sed 's/$/+PTX/')}
[ -z "$archs" ] && archs="8.9+PTX"
export TORCH_CUDA_ARCH_LIST="$archs"
export CUDAARCHS=$(echo "$archs" | tr ';' '\n' | sed 's/+PTX//' | tr -d . | paste -sd',' -)
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CUDAARCHS=$CUDAARCHS"

echo "=== Building fVDB ==="
cd /workspace/fvdb-core
./build.sh install verbose

if [[ -n "${GSPLAT_CLONE_CMD}" ]]; then
    eval "${GSPLAT_CLONE_CMD}"
fi

echo "=== Building GSplat ==="
pip install -v --no-build-isolation --no-cache-dir /workspace/gsplat

echo "=== Installing fused-ssim ==="
pip install -v --no-build-isolation --no-cache-dir \
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"

echo "=== Installing fvdb-reality-capture ==="
cd /workspace/fvdb-reality-capture
pip install -e .

echo "=== Patching pycolmap ==="
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
sed -i 's/INVALID_POINT3D = np.uint64(-1)/INVALID_POINT3D = np.uint64(2**64-1)/' \
    "$SITE/pycolmap/scene_manager.py" || true

echo "=== Downloading mipnerf360 dataset (if needed) ==="
frgs download mipnerf360 --download-path /workspace/data

echo "=== Running comparative benchmark ==="
cd /workspace/fvdb-reality-capture/tests/benchmarks/comparative
# EXTRA_ARGS is a flat space-separated string from "$*"; word splitting is intentional.
python comparison_benchmark.py --matrix "${MATRIX}" ${EXTRA_ARGS}

echo "=== Converting results to benchmark format ==="
RESULTS_NAME=$(python - "${MATRIX}" <<'PY'
import sys
import yaml

matrix_path = sys.argv[1]
with open(matrix_path, "r", encoding="utf-8") as f:
    m = yaml.safe_load(f)
print(m.get("name", "benchmark"))
PY
)
python format_for_gh_benchmark.py \
    "results/${RESULTS_NAME}/summary/summary_data.json" \
    --output-dir "results/${RESULTS_NAME}/summary"

echo "=== Fixing output file ownership ==="
RESULTS_DIR="/workspace/fvdb-reality-capture/tests/benchmarks/comparative/results/${RESULTS_NAME}"
if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${RESULTS_DIR}" 2>/dev/null || true
fi

echo "=== Done ==="
echo "Results at: tests/benchmarks/comparative/results/${RESULTS_NAME}/summary/"
INNER

chmod +x "${INNER_SCRIPT}"

# shellcheck disable=SC2086
docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --shm-size 16gb \
    --ipc host \
    -e PYTHONPATH="" \
    -e CPM_SOURCE_CACHE="/workspace/.cache/CPM" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    ${CUDA_ARCH_LIST:+-e TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"} \
    -v "${FVDB_CORE_DIR}:/workspace/fvdb-core:rw" \
    -v "${REPO_ROOT}:/workspace/fvdb-reality-capture:rw" \
    -v "${DATA_DIR}:/workspace/data:rw" \
    ${GSPLAT_MOUNT} \
    -v "${INNER_SCRIPT}:/workspace/run_inner.sh:ro" \
    -w /workspace/fvdb-reality-capture \
    "${CONTAINER_IMAGE}" \
    bash -l /workspace/run_inner.sh \
        "${ENV_FILE}" \
        "${MATRIX}" \
        "${GSPLAT_CLONE_CMD}" \
        "${EXTRA_ARGS[@]}"
