#!/bin/bash
# Visualise BEV memories generated from a dataset (debug helper).
#
# Override the environment if needed:
#   CONDA_ENV=/path/to/env ./scripts/run_bev_gen.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

python scripts/data_pipeline/generate_bev_from_dataset.py "$@"
