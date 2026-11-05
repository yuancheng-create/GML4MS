#!/usr/bin/env bash
set -euo pipefail
FE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${FE_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "${SAM_PYTHON:-python}" -u -B "$@"
