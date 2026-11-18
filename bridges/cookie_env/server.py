"""cookie-env side of the bridge (runs under this sub-project's Python >= 3.11).

Spawned by examples/cookie_grid.py. Protocol: one JSON object per line on
stdin, one JSON reply per line on the original stdout. Anything else that gets
printed (pygame banner, warnings, stray prints) is redirected to stderr so it
can't corrupt the protocol. The process exits when stdin closes.

Commands (all replies carry "ok"; failures carry "error" with a traceback):
  make   {env_id, kwargs}          -> {n_actions, size, max_steps}
  reset  {seed}                    -> state
  step   {action}                  -> state + {reward, terminated, truncated}
  render {states, tile_size}       -> {frames (base64 uint8), shape}
         Renders arbitrary (agent_pos, agent_dir, goal_pos) states on a
         separate env instance, so the running episode (and its RNG) is
         never touched.
where state = {agent_pos: [x, y], agent_dir, goal_pos: [x, y], step_count}.
"""
import base64
import json
import os
import sys
import traceback
import warnings

_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message=".*not within the observation space.*")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import cookie_env  # noqa: E402,F401  (registers the env ids)
from minigrid.core.world_object import Goal  # noqa: E402

env = None
base = None
render_base = None


def _state():
    return {
        "agent_pos": [int(v) for v in base.agent_pos],
        "agent_dir": int(base.agent_dir),
        "goal_pos": [int(v) for v in base.goal_position],
        "step_count": int(base.step_count),
    }


def make(env_id, kwargs):
    global env, base, render_base
    if "agent_start_pos" in kwargs and kwargs["agent_start_pos"] is not None:
        kwargs["agent_start_pos"] = tuple(kwargs["agent_start_pos"])
    env = gym.make(env_id, **kwargs)
    base = env.unwrapped
    render_env = gym.make(env_id, **kwargs)
    render_env.reset(seed=0)
    render_base = render_env.unwrapped
    return {"n_actions": int(env.action_space.n), "size": int(base.size),
            "max_steps": int(base.max_steps)}


def reset(seed=None):
    env.reset(seed=seed)
    return _state()


def step(action):
    _, reward, terminated, truncated, _ = env.step(int(action))
    return {**_state(), "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated)}


def render(states, tile_size=8):
    rb = render_base
    frames = []
    for s in states:
        goal = tuple(int(v) for v in s["goal_pos"])
        old_goal = tuple(int(v) for v in rb.goal_position)
        rb.grid.set(*old_goal, None)
        rb.grid.set(*goal, Goal())
        rb._goal_pos = goal
        rb.agent_pos = tuple(int(v) for v in s["agent_pos"])
        rb.agent_dir = int(s.get("agent_dir", 0))
        frames.append(rb.get_frame(highlight=False, tile_size=tile_size))
    arr = np.ascontiguousarray(np.stack(frames).astype(np.uint8))
    return {"frames": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape)}


COMMANDS = {"make": make, "reset": reset, "step": step, "render": render}

for line in sys.stdin:
    if not line.strip():
        continue
    try:
        msg = json.loads(line)
        cmd = msg.pop("cmd")
        reply = {"ok": True, **COMMANDS[cmd](**msg)}
    except Exception:
        reply = {"ok": False, "error": traceback.format_exc()}
    _proto.write(json.dumps(reply) + "\n")
