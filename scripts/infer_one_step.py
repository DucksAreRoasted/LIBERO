"""Run the LIBERO-finetuned SmolVLA on one demonstration frame.

This is open-loop: the model sees the cameras, proprioception, and the
language instruction, then predicts the 7-DoF action. It does not step
the simulator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DATASET_ID = os.environ.get("DATASET_REPO", "lerobot/libero")
POLICY_ID = os.environ.get("POLICY", "HuggingFaceVLA/smolvla_libero")
EPISODE = int(os.environ.get("EPISODE", "0"))
OUT = Path(os.environ.get("INFER_DIR", "outputs/infer"))


def task_text(item: dict, dataset) -> str:
    task = item.get("task")
    if isinstance(task, str) and task.strip():
        return task
    task_index = item.get("task_index")
    tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
    if task_index is None or tasks is None:
        return ""
    index = int(task_index)
    try:
        return str(tasks.iloc[index, 0])
    except Exception:
        return str(tasks[index]) if not isinstance(tasks, dict) else str(tasks.get(index, ""))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"policy: {POLICY_ID}")
    print(f"device: {device}")

    policy = SmolVLAPolicy.from_pretrained(POLICY_ID)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        POLICY_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    dataset = LeRobotDataset(
        DATASET_ID,
        episodes=[EPISODE],
        video_backend=os.environ.get("VIDEO_BACKEND", "pyav"),
    )
    item = dataset[0]
    task = task_text(item, dataset)
    frame = {
        "observation.images.image": item["observation.images.image"],
        "observation.images.image2": item["observation.images.image2"],
        "observation.state": item["observation.state"],
        "task": task,
    }

    policy.reset()
    action = postprocess(policy.select_action(preprocess(frame))).detach().cpu().flatten()
    expert = item["action"].detach().cpu().flatten()
    distance = torch.linalg.vector_norm(action.float() - expert.float()).item()

    def listed(tensor: torch.Tensor) -> list[float]:
        return [round(float(value), 5) for value in tensor]

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": POLICY_ID,
        "dataset": DATASET_ID,
        "episode": EPISODE,
        "task": task,
        "device": str(device),
        "chunk_size": int(policy.config.chunk_size),
        "n_action_steps": int(policy.config.n_action_steps),
        "num_steps": int(policy.config.num_steps),
        "predicted_action_7d": listed(action),
        "expert_action_7d": listed(expert),
        "l2_distance": round(distance, 5),
    }
    (OUT / "action.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"task: {task}")
    print(
        f"chunk_size={policy.config.chunk_size}  "
        f"n_action_steps={policy.config.n_action_steps}  "
        f"num_steps={policy.config.num_steps}"
    )
    print(f"predicted: {listed(action)}")
    print(f"expert:    {listed(expert)}")
    print(f"l2:        {distance:.4f}")
    print(f"wrote {OUT / 'action.json'}")


if __name__ == "__main__":
    main()
