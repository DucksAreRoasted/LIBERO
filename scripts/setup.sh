#!/usr/bin/env bash
# Create a Python 3.12 env and install LeRobot with SmolVLA + LIBERO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "需要 uv。安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'lerobot[smolvla,libero,training]' 'socksio>=1.0.0'

# RTX 5070 is Blackwell. If the default wheel cannot see CUDA, switch to the CUDA 12.8 build.
if ! .venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "默认 PyTorch 看不到 GPU，改装 CUDA 12.8 轮子。"
  uv pip install --python .venv/bin/python --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
fi

echo
echo "环境已就绪。下一步:"
echo "  bash scripts/inspect_dataset.sh    # 看一条示教"
echo "  bash scripts/infer_one_step.sh     # 同一帧上让 SmolVLA 预测动作"
echo "  bash scripts/eval_pretrained.sh    # 官方权重在仿真里做 1 个任务"
echo "  bash scripts/train_libero.sh       # 在 8GB 显卡上微调"
