#!/usr/bin/env bash
# Closed-loop eval of the official SmolVLA checkpoint on one LIBERO-Spatial task.
# One episode is enough to see language -> action -> MuJoCo success check.
set -euo pipefail
source "$(dirname "$0")/common.sh"

POLICY="${POLICY:-HuggingFaceVLA/smolvla_libero}"
SUITE="${SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
OUT="${OUT:-outputs/eval/pretrained_${SUITE}}"

if [[ -n "${PROMPT:-}" ]]; then
  echo "只更换模型听到的英文提示词。日志里的成功率仍按原任务的目标计算，泛化与否请看视频。"
  python scripts/eval_with_prompt.py \
else
  lerobot-eval \
fi
  --policy.path="$POLICY" \
  --policy.path="$POLICY" \
  --policy.device="${DEVICE:-cuda}" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="$TASK_IDS" \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="$N_EPISODES" \
  --eval.use_async_envs=false \
  --output_dir="$OUT" \
  --seed="${SEED:-42}"

echo
echo "结果写在 $OUT"
echo "视频在 $OUT/videos （若该版本会录屏）"
