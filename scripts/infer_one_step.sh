#!/usr/bin/env bash
# Open-loop: one LIBERO frame in, one 7-DoF action out.
set -euo pipefail
source "$(dirname "$0")/common.sh"

export DATASET_REPO="${DATASET_REPO:-lerobot/libero}"
export POLICY="${POLICY:-HuggingFaceVLA/smolvla_libero}"
export EPISODE="${EPISODE:-0}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

python scripts/infer_one_step.py
