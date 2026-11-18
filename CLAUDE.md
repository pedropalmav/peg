# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Official implementation of PEG (Planning Exploratory Goals, ICLR 2023): an unsupervised goal-conditioned model-based RL agent built on a fork of DreamerV2/LEXA (TensorFlow 2). Also includes model-based versions of the MEGA, SkewFit, LEXA and P2E baselines. The `docs/` folder is the static project website (GitHub Pages) and is unrelated to the code.

## Environment / setup

- Python 3.6 + TensorFlow 2.4 (GPU). Pinned conda env: `conda env create -f environment.yml && conda activate peg`, then `pip install -e .` (the package installs as `dreamerv2`).
- Environments live in **external repos** that must be on `PYTHONPATH` (only one at a time):
  - Point Maze, Ant Maze, 3-block Stack → `mrl` (github.com/hueds/mrl)
  - Walker → `lexa-benchmark` (+ `dm_control` installed with `pip install .`, no `-e`)
- Ant Maze, Walker and Stack need MuJoCo 2.0. Rendering often needs `MUJOCO_GL=egl MUJOCO_RENDERER=egl` and `LD_PRELOAD` set to libGLEW/libGL (or `unset LD_PRELOAD` if GLEW errors).
- `main()` asserts that a GPU is present; there is no CPU path without removing that assert.

## Commands

There are no tests or linters. The single entry point is `examples/run_goal_cond.py`:

```bash
# Quick smoke test (~2-3 min, Point Maze, needs mrl on PYTHONPATH)
python examples/run_goal_cond.py --configs peg_point_maze test_maze --jit True --logdir ~/logdir/test_peg_maze
# Walker smoke test (needs lexa-benchmark on PYTHONPATH)
python examples/run_goal_cond.py --configs peg_walker test_walker --jit True --logdir ~/logdir/test_peg_walker

# Full runs: peg_point_maze | peg_ant_maze | peg_walker | peg_three_stack
python examples/run_goal_cond.py --configs peg_ant_maze --logdir ~/logdir/peg_ant_maze
# Baselines via overrides
--goal_strategy MEGA | --goal_strategy Skewfit
--goal_strategy SampleReplay --two_policy_rollout_every 0 --gcp_rollout_every 1 --exp_rollout_every 1   # LEXA
--goal_strategy SampleReplay --two_policy_rollout_every 0 --gcp_rollout_every 0 --exp_rollout_every 1   # P2E

tensorboard --logdir ~/logdir/<run>
```

- `--jit False` runs TF eagerly (no `@tf.function` graphs) so `ipdb` breakpoints work inside model code. The `debug` config shortens timing to reach breakpoints fast.

## Configuration

- All hyperparameters are in `dreamerv2/configs.yaml`. `defaults` is always loaded; `--configs a b c` applies named sections in order (later overrides earlier). Any key can then be overridden with `--key value` (nested keys like `planner` are dicts in the yaml).
- **Resume behavior:** if `--logdir` already exists, `main()` loads `logdir/config.yaml` instead of rebuilding from `configs.yaml` (CLI `--key value` overrides still apply), and training resumes from `variables.pkl`. Edits to `configs.yaml` therefore do not affect an existing logdir — use a fresh logdir when changing config.
- `config.task` is a string that selects the environment and all env-specific helpers via substring/equality checks (e.g. `pointmaze`, `hardumazefulldownscale` = Ant Maze, `dmc_walker_walk_proprio` = Walker, `demofetchpnp`/`wallsdemofetchpnp*` = Stack).

## Architecture

Flow: `examples/run_goal_cond.py:main` → builds config, then env + per-task factory functions (`make_env`, `make_eval_fn`, `make_report_render_function`, `make_ep_render_fn`, `make_plot_fn`, `make_cem_vis_fn`, `make_obs2goal_fn`, `make_sample_env_goals`) → passes them to `dreamerv2/api.py:train`. Adding a new environment means adding a branch keyed on `config.task` in each of these factories (the file is ~2500 lines of such branches) plus a config section in `configs.yaml`.

`dreamerv2/api.py:train` is the training/eval loop:
- Builds replay (`common/replay.py`), `GCAgent`, and a `common.GCDriver` that performs rollouts.
- Rollout type per episode is chosen by `gcp_rollout_every` (goal-conditioned policy only), `exp_rollout_every` (exploration policy only) and `two_policy_rollout_every` (Go-Explore: GC policy toward a picked goal, then exploration policy after the goal time limit) — see `common/driver.py:GCDriver`.
- Goal selection: `config.goal_strategy` is the **class name** looked up via `getattr(goal_picker, ...)` in `dreamerv2/goal_picker.py`. `SubgoalPlanner` is PEG (CEM/MPPI over goals in the world model, scored by the P2E exploration value; params under `config.planner`); `MEGA`/`Skewfit` use a KDE over replay; `SampleReplay` (LEXA/P2E) samples goals from replay; `Greedy` and `SubgoalPlannerKDE` are additional variants. Construction and per-episode update logic for each strategy is special-cased in `api.py`, so a new strategy needs handling there too.
- Periodic evaluation, Tensorboard/`metrics.jsonl` logging and `variables.pkl` checkpointing.

`dreamerv2/gc_agent.py`: `GCAgent` = `GCWorldModel` (extends `agent.WorldModel`, RSSM) + `GCActorCritic` (goal-conditioned policy trained in imagination, with temporal-distance reward) + exploration behavior from `expl.py` (`Plan2Explore` ensemble disagreement by default; also `RND`, `ModelLoss`, `Random`). `common/` holds DreamerV2 building blocks (nets, dists, env wrappers in `envs.py`, logger, `Config`/`Flags`).

Imports are path-based, not package-relative: `dreamerv2/api.py` appends its own directory to `sys.path`, so modules do `import common`, `import gc_agent`, `import goal_picker`, and `run_goal_cond.py` does `import common` / `import envs` after importing `dreamerv2.api`. Keep this style when adding modules. `import envs` is not `dreamerv2/common/envs.py` (that is `common.envs`); it must come from the external env repo on `PYTHONPATH`.

`dreamerv2/train.py`, `agent.py`'s `Agent`, and `scores/` are legacy upstream DreamerV2 (non-goal-conditioned) code, not used by the PEG pipeline.
