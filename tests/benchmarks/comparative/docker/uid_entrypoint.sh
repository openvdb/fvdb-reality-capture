# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash
set -euo pipefail

# Map container runtime user to the host user so files created on bind mounts
# are owned by the user who launched the container.
#
# Configure via env:
# - LOCAL_UID: host UID (optional; auto-detected from bind mounts when unset)
# - LOCAL_GID: host GID (optional; auto-detected from bind mounts when unset)

LOCAL_UID="${LOCAL_UID-}"
LOCAL_GID="${LOCAL_GID-}"
LOCAL_USER="${LOCAL_USER:-benchmark}"
LOCAL_GROUP="${LOCAL_GROUP:-benchmark}"

if [[ "$(id -u)" != "0" ]]; then
  echo "uid_entrypoint: must start as root (current uid=$(id -u))." >&2
  exec "$@"
fi

# If the caller didn't specify LOCAL_UID/GID, infer them from bind-mounted paths.
# Prefer the repo bind mount (openvdb/benchmark) since that's where we need to write build outputs.
if [[ -z "${LOCAL_UID}" || -z "${LOCAL_GID}" ]]; then
  for probe in /workspace/openvdb /workspace/benchmark /workspace/.cache/CPM /workspace/results; do
    if [[ -e "${probe}" ]]; then
      inferred_uid="$(stat -c '%u' "${probe}" 2>/dev/null || true)"
      inferred_gid="$(stat -c '%g' "${probe}" 2>/dev/null || true)"
      if [[ -n "${inferred_uid}" && -n "${inferred_gid}" ]]; then
        LOCAL_UID="${LOCAL_UID:-${inferred_uid}}"
        LOCAL_GID="${LOCAL_GID:-${inferred_gid}}"
        break
      fi
    fi
  done
fi

# Final fallback (works for many images, but is wrong on some hosts, hence detection above).
LOCAL_UID="${LOCAL_UID:-1000}"
LOCAL_GID="${LOCAL_GID:-1000}"

echo "uid_entrypoint: using LOCAL_UID=${LOCAL_UID} LOCAL_GID=${LOCAL_GID}" >&2

if ! getent group "${LOCAL_GID}" >/dev/null 2>&1; then
  groupadd -g "${LOCAL_GID}" "${LOCAL_GROUP}"
else
  LOCAL_GROUP="$(getent group "${LOCAL_GID}" | cut -d: -f1)"
fi

if ! getent passwd "${LOCAL_UID}" >/dev/null 2>&1; then
  useradd -m -u "${LOCAL_UID}" -g "${LOCAL_GID}" -s /bin/bash "${LOCAL_USER}"
else
  LOCAL_USER="$(getent passwd "${LOCAL_UID}" | cut -d: -f1)"
fi

export HOME="/home/${LOCAL_USER}"

# Ensure common writable dirs exist (avoids tools writing into /root or /).
mkdir -p /workspace/results /workspace/.cache/CPM "${HOME}"
chown -R "${LOCAL_UID}:${LOCAL_GID}" /workspace/results /workspace/.cache "${HOME}" >/dev/null 2>&1 || true

# If previous runs created build artifacts as root (or another UID), fix ownership so CMake can write.
# Keep this narrowly targeted to avoid expensive recursive chowns on the whole repo.
for p in \
  /workspace/openvdb/fvdb-core/build \
  /workspace/openvdb/fvdb-core/dist \
  /workspace/openvdb/fvdb-core/_skbuild \
  /workspace/openvdb/fvdb-core/.cache \
  ; do
  if [[ -e "${p}" ]]; then
    chown -R "${LOCAL_UID}:${LOCAL_GID}" "${p}" >/dev/null 2>&1 || true
  fi
done

exec setpriv --reuid="${LOCAL_UID}" --regid="${LOCAL_GID}" --init-groups "$@"
