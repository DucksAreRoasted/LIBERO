#!/usr/bin/env bash
# Download (on first run) lerobot/libero and print one training sample.
set -euo pipefail
source "$(dirname "$0")/common.sh"

export DATASET_REPO="${DATASET_REPO:-lerobot/libero}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
export SAMPLE_INDEX="${SAMPLE_INDEX:-0}"

python scripts/inspect_dataset.py
