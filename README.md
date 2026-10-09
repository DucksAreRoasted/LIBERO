# LIBERO + SmolVLA 上手

这条仓库按官方 LeRobot 链路走一遍。默认每一步都很小，用来看清数据、模型和仿真是怎么接上的，而不是刷榜。

机器上的 RTX 5070 可以跑推理和短训练。完整实验大约是 10 万步微调，再在四个套件上各评 10 条。

## 这条链路

1. **数据。** `lerobot/libero` 把 LIBERO 的四套演示收成一个数据集：1693 条轨迹、40 个语言任务。每一帧是两路 256×256 相机、8 维本体状态，监督信号是 7 维动作（末端 6 维增量 + 夹爪）。
2. **模型。** SmolVLA 用 SmolVLM2-500M 读图像和语言，再用一个动作专家做流匹配，一次生成一串动作。`lerobot/smolvla_base` 是基座；`HuggingFaceVLA/smolvla_libero` 是已经在 LIBERO 上微调过的检查点。
3. **训练。** `lerobot-train` 从基座出发，在演示上拟合动作。
4. **评测。** `lerobot-eval` 在 MuJoCo 里重置场景、把观测送给策略、执行动作，用环境自带的成功判定算成功率。

## 准备

```bash
bash scripts/setup.sh
```

这会用 `uv` 建一个 Python 3.12 环境，并装上 LeRobot 的 SmolVLA、LIBERO 和训练依赖。如果默认 PyTorch 看不到 GPU，脚本会改装 CUDA 12.8 轮子（RTX 5070 需要它）。

## 四步

看一条示教，主相机和腕相机会写到 `outputs/inspect/`：

```bash
bash scripts/inspect_dataset.sh
```

用微调过的 SmolVLA 对同一帧做一次开环预测，结果在 `outputs/infer/action.json`：

```bash
bash scripts/infer_one_step.sh
```

在仿真里让这个检查点自己做完 1 条 LIBERO-Spatial。无显示器时脚本会用 EGL：

```bash
bash scripts/eval_pretrained.sh
```

在这台 8GB 显卡上训练。第一次会下载约 1.9GB 的 `lerobot/libero`：

```bash
bash scripts/train_libero.sh
```

默认 10 万步、batch 4，结构对齐 `HuggingFaceVLA/smolvla_libero`。显存长期低于约 6.5GB 时用 `BATCH=8`，显存不足时用 `BATCH=2`。想先看到学习率降完，用 `STEPS=30000`。评测自己的检查点：

```bash
POLICY=outputs/train/smolvla_libero/checkpoints/last/pretrained_model \
  SUITE=libero_spatial N_EPISODES=10 \
  bash scripts/eval_pretrained.sh
```
