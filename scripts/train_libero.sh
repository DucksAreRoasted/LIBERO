#!/usr/bin/env bash
# Fine-tune SmolVLA on lerobot/libero for this RTX 5070 Laptop (8 GB).
#
# The policy shape matches HuggingFaceVLA/smolvla_libero: full SmolVLM2-500M-Instruct,
# expert width 0.5, and n_action_steps=1. Defaults are 100k steps at batch 4.
# If nvidia-smi stays under about 6.5 GB, rerun with BATCH=8. If it runs out of
# memory, rerun with BATCH=2. A shorter first look is STEPS=30000, which is when
# the learning rate has finished decaying.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STEPS="${STEPS:-100000}"
BATCH="${BATCH:-4}"
OUT="${OUT:-outputs/train/smolvla_libero}"

lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct \
  --policy.num_vlm_layers=0 \
  --policy.expert_width_multiplier=0.5 \
  --policy.n_action_steps=1 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.device="${DEVICE:-cuda}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO:-lerobot/libero}" \
  --dataset.video_backend="${VIDEO_BACKEND:-pyav}" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="${SAVE_FREQ:-10000}" \
  --log_freq="${LOG_FREQ:-100}" \
  --num_workers="${NUM_WORKERS:-4}" \
  --wandb.enable=false \
  --output_dir="$OUT" \
  --job_name=smolvla_libero \
  --seed="${SEED:-42}"

echo
echo "checkpoint: $OUT/checkpoints/last/pretrained_model"
echo "评测: POLICY=$OUT/checkpoints/last/pretrained_model SUITE=libero_spatial N_EPISODES=10 bash scripts/eval_pretrained.sh"
