"""PEG-side (Python 3.8) client for cookie-env's GoalGrid environments.

cookie-env needs Python >= 3.11, so it runs in its own uv venv
(bridges/cookie_env, set up with `uv sync --project bridges/cookie_env`) and
bridges/cookie_env/server.py drives it. This module spawns that server and
exposes the env through the old gym API used by the rest of PEG.

Goal space = observation space = the agent's (x, y) grid cell, as in Point
Maze; the env's green square is the evaluation goal. The agent's direction is
not part of the observation (PEG encodes goals with the state encoder, so the
goal must have the observation's shape, and env goals have no direction); the
recurrent world model can infer it from the action history. The true direction
is still recorded as `log_agent_dir` for rendering videos; `log_` keys are
skipped by the world model's preprocessing and by replay sampling, so the
agent never sees it.
"""
import atexit
import base64
import json
import os
import pathlib
import subprocess

import gym
import numpy as np

BRIDGE_DIR = pathlib.Path(__file__).resolve().parent.parent / 'bridges' / 'cookie_env'


class CookieEnvBridge:
  """Owns one cookie-env server process holding one environment."""

  def __init__(self, env_id, env_kwargs):
    python = os.environ.get('COOKIE_ENV_PYTHON', str(BRIDGE_DIR / '.venv' / 'bin' / 'python'))
    if not pathlib.Path(python).exists():
      raise RuntimeError(
          f'cookie-env bridge interpreter not found at {python}. '
          f'Run: uv sync --project {BRIDGE_DIR}')
    # -I: ignore PYTHONPATH (it points at PEG's py3.8 env repos) and user site.
    self._proc = subprocess.Popen(
        [python, '-I', str(BRIDGE_DIR / 'server.py')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
    atexit.register(self.close)
    self.info = self.call('make', env_id=env_id, kwargs=env_kwargs)

  def call(self, cmd, **kwargs):
    if self._proc.poll() is not None:
      raise RuntimeError(f'cookie-env server exited with code {self._proc.returncode}')
    self._proc.stdin.write(json.dumps({'cmd': cmd, **kwargs}) + '\n')
    self._proc.stdin.flush()
    line = self._proc.stdout.readline()
    if not line:
      raise RuntimeError(f'cookie-env server died during {cmd!r} (see stderr above)')
    reply = json.loads(line)
    if not reply.pop('ok'):
      raise RuntimeError(f'cookie-env server error in {cmd!r}:\n{reply["error"]}')
    return reply

  def close(self):
    if self._proc.poll() is None:
      try:
        self._proc.stdin.close()
        self._proc.wait(timeout=5)
      except Exception:
        self._proc.kill()


class GoalGridEnv(gym.Env):
  """GoalGrid (cookie-env) with obs {'observation': agent (x, y), 'goal': square (x, y)}.

  Never terminates; the episode is truncated after `max_steps`. info carries
  `metric_success` (agent on the goal cell) and `is_terminal` (always False,
  so the time limit isn't treated as a terminal state).
  """

  def __init__(self, env_id='GoalGrid-random-v0', max_steps=50, seed=None, **env_kwargs):
    self._bridge = CookieEnvBridge(env_id, {'max_steps': max_steps, **env_kwargs})
    self.size = self._bridge.info['size']
    self.max_steps = self._bridge.info['max_steps']
    self._seed = seed
    self._state = None
    lo, hi = 1.0, float(self.size - 2)  # interior cells; the border is walls.
    self.goal_min = np.array([lo, lo], np.float32)
    self.goal_max = np.array([hi, hi], np.float32)
    box = gym.spaces.Box(lo, hi, (2,), dtype=np.float32)
    # log_agent_dir is returned in every obs but deliberately left out of the
    # space: the model's encoder is built from the space and would expect it in
    # training batches, which replay strips `log_` keys from.
    self.observation_space = gym.spaces.Dict({'observation': box, 'goal': box})
    self.action_space = gym.spaces.Discrete(self._bridge.info['n_actions'])

  def _obs(self):
    return {
        'observation': np.asarray(self._state['agent_pos'], np.float32),
        'goal': np.asarray(self._state['goal_pos'], np.float32),
        'log_agent_dir': np.int32(self._state['agent_dir']),
    }

  def get_metrics_dict(self):
    reached = self._state['agent_pos'] == self._state['goal_pos']
    return {'metric_success': float(reached)}

  def reset(self):
    # Seed only the first reset; later resets continue the env's RNG stream.
    self._state = self._bridge.call('reset', seed=self._seed)
    self._seed = None
    return self._obs()

  def step(self, action):
    self._state = self._bridge.call('step', action=int(action))
    done = self._state['terminated'] or self._state['truncated']
    info = {'is_terminal': self._state['terminated'], **self.get_metrics_dict()}
    return self._obs(), self._state['reward'], done, info

  def render_states(self, agent_xy, goal_xy, agent_dir=None, tile_size=8):
    """Render (possibly imagined) states -> uint8 array (N, H, W, 3).

    Positions are rounded and clipped to the interior, so model predictions
    can be passed directly. Without agent_dir (e.g. imagined states, which
    have no direction) the agent is drawn facing east.
    """
    agent_xy = np.clip(np.rint(np.asarray(agent_xy, np.float64)), self.goal_min, self.goal_max).astype(int)
    goal_xy = np.clip(np.rint(np.asarray(goal_xy, np.float64)), self.goal_min, self.goal_max).astype(int)
    dirs = np.zeros(len(agent_xy), int) if agent_dir is None else np.asarray(agent_dir, int)
    states = [{'agent_pos': a.tolist(), 'goal_pos': g.tolist(), 'agent_dir': int(d)}
              for a, g, d in zip(agent_xy, goal_xy, dirs)]
    reply = self._bridge.call('render', states=states, tile_size=tile_size)
    return np.frombuffer(base64.b64decode(reply['frames']), np.uint8).reshape(reply['shape'])

  def close(self):
    self._bridge.close()
