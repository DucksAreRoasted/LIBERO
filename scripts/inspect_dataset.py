"""Print one LIBERO demonstration and save its two camera frames.

This is the data side of the loop: language instruction, proprioception,
and the 7-DoF action that SmolVLA is trained to predict.
"""

from __future__ import annotations

import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

REPO_ID = os.environ.get("DATASET_REPO", "lerobot/libero")
INDEX = int(os.environ.get("SAMPLE_INDEX", "0"))
EPISODE = os.environ.get("EPISODE", "0")
OUT = Path(os.environ.get("INSPECT_DIR", "outputs/inspect"))


def to_pil(tensor) -> Image.Image:
    array = tensor.detach().cpu()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)
    array = array.numpy()
    if array.max() <= 1.0:
        array = (array * 255).clip(0, 255)
    return Image.fromarray(array.astype("uint8"))


def main() -> None:
    episodes = None if EPISODE == "all" else [int(part) for part in EPISODE.split(",")]
    dataset = LeRobotDataset(
        REPO_ID,
        episodes=episodes,
        video_backend=os.environ.get("VIDEO_BACKEND", "pyav"),
    )
    sample = dataset[INDEX]
    meta = dataset.meta
    info = getattr(meta, "info", {}) or {}
    n_episodes = info.get("total_episodes", dataset.num_episodes)
    n_frames = info.get("total_frames", dataset.num_frames)
    n_tasks = info.get("total_tasks", getattr(meta, "total_tasks", None))
    if n_tasks is None and getattr(meta, "tasks", None) is not None:
        n_tasks = len(meta.tasks)

    print(f"dataset: {REPO_ID}")
    print(f"episodes: {n_episodes}  frames: {n_frames}  tasks: {n_tasks}")
    print(f"loaded episode(s): {EPISODE}")
    print(f"sample index: {INDEX}")
    task = sample.get("task")
    if task is None and "task_index" in sample and getattr(meta, "tasks", None) is not None:
        task_index = int(sample["task_index"])
        tasks = meta.tasks
        task = tasks[task_index] if isinstance(tasks, (list, tuple)) else tasks.iloc[task_index]
    if task is not None:
        print(f"task: {task}")
    print("tensors:")
    for key, value in sample.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value).__name__)
        print(f"  {key}: shape={tuple(shape) if shape is not None else '-'} dtype={dtype}")

    OUT.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("observation.images.image", "agentview.png"),
        ("observation.images.image2", "wrist.png"),
    ):
        if key not in sample:
            continue
        path = OUT / filename
        to_pil(sample[key]).save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
