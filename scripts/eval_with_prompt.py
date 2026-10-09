"""Run lerobot-eval with a replacement language prompt.

The LIBERO scene, initial object poses, and success checker stay on the
selected task. Only the sentence passed to SmolVLA changes.
"""

from __future__ import annotations

import os

from lerobot.envs.libero import LiberoEnv

PROMPT = os.environ.get("PROMPT", "").strip()
if not PROMPT:
    raise SystemExit("PROMPT is empty.")

_original_init = LiberoEnv.__init__


def _init_with_prompt(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    original = self.task_description
    self.task_description = PROMPT
    print(f"模型听到的提示词: {PROMPT}")
    print(f"场景和成功判定仍属于原任务: {original}")


LiberoEnv.__init__ = _init_with_prompt

from lerobot.scripts.lerobot_eval import main  # noqa: E402

main()
