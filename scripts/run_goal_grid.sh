#!/usr/bin/env bash
# Train PEG and the baselines on cookie-env's GoalGrid.
#
#   scripts/run_goal_grid.sh                       # all methods, seed 0, defaults below
#   scripts/run_goal_grid.sh --test                # ~1 min per method smoke test
#   scripts/run_goal_grid.sh --methods "peg mega" --seeds "0 1 2" --steps 200000
#   scripts/run_goal_grid.sh --env GoalGrid-v0 --fixed-start --time-limit 30
#   scripts/run_goal_grid.sh --methods lexa -- --eval_eps 50   # args after -- go to every run
#
# Options (defaults come from peg_goal_grid / cookie_env in dreamerv2/configs.yaml):
#   --methods "..."    any of: peg mega skewfit lexa p2e   (default: all five)
#   --seeds "..."      (default: 0)
#   --env ID           GoalGrid-random-v0 (goal resampled each episode, default)
#                      or GoalGrid-v0 (goal fixed at (8, 1))
#   --fixed-start      use the env's registered start instead of a random start
#                      cell + direction each episode
#   --steps N          total env steps per run (default 300000)
#   --time-limit N     episode length in steps (default 50)
#   --logroot DIR      default <repo>/logdir/goal_grid/<env>[_fixedstart]
#   --test             short smoke-test timings; logroot gets a _test suffix
#
# Runs are sequential (one GPU). Logdirs are <logroot>/<method>/seed<seed>
# (stdout in seed<seed>.stdout.log beside it). Re-running with an existing
# logdir resumes it and reuses its saved config.yaml (see CLAUDE.md), so use a
# new --logroot when changing settings.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

METHODS="peg mega skewfit lexa p2e"
SEEDS="0"
STEPS=""
TIME_LIMIT=""
ENV_ID="GoalGrid-random-v0"
RANDOM_START=True
LOGROOT=""
TEST=0
EXTRA=()

usage() { sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --methods)     METHODS="$2"; shift 2 ;;
    --seeds)       SEEDS="$2"; shift 2 ;;
    --env)         ENV_ID="$2"; shift 2 ;;
    --fixed-start) RANDOM_START=False; shift ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --time-limit)  TIME_LIMIT="$2"; shift 2 ;;
    --logroot)     LOGROOT="$2"; shift 2 ;;
    --test)        TEST=1; shift ;;
    -h|--help)     usage ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done
case "$ENV_ID" in
  GoalGrid-v0|GoalGrid-random-v0) ;;
  *) echo "unsupported --env $ENV_ID (expected GoalGrid-v0 or GoalGrid-random-v0)" >&2; exit 1 ;;
esac
if [[ -z "$LOGROOT" ]]; then
  LOGROOT="$ROOT/logdir/goal_grid/$ENV_ID"
  [[ $RANDOM_START == False ]] && LOGROOT+="_fixedstart"
fi

method_args() {
  case "$1" in
    peg)     echo "--goal_strategy SubgoalPlanner" ;;
    mega)    echo "--goal_strategy MEGA" ;;
    skewfit) echo "--goal_strategy Skewfit" ;;
    lexa)    echo "--goal_strategy SampleReplay --two_policy_rollout_every 0 --gcp_rollout_every 1 --exp_rollout_every 1" ;;
    p2e)     echo "--goal_strategy SampleReplay --two_policy_rollout_every 0 --gcp_rollout_every 0 --exp_rollout_every 1" ;;
    *) echo "unknown method: $1 (expected: peg mega skewfit lexa p2e)" >&2; return 1 ;;
  esac
}
for m in $METHODS; do method_args "$m" > /dev/null; done  # validate before starting

CONFIGS=(peg_goal_grid)
if [[ $TEST == 1 ]]; then
  CONFIGS+=(test_maze)  # generic short-timing overrides despite the name
  STEPS="${STEPS:-2000}"
  LOGROOT="${LOGROOT%/}_test"
  EXTRA=(--eval_eps 5 "${EXTRA[@]}")
fi
[[ -n "$STEPS" ]] && EXTRA=(--steps "$STEPS" "${EXTRA[@]}")
if [[ -n "$TIME_LIMIT" ]]; then
  # Training sequences are 50 steps (replay.minlen/maxlen, dataset.length).
  # Replay drops episodes shorter than replay.minlen, so for limits below 50
  # these shrink to the limit; longer episodes are just cut into 50-step chunks.
  CHUNK=$(( TIME_LIMIT < 50 ? TIME_LIMIT : 50 ))
  EXTRA=(--time_limit "$TIME_LIMIT" --replay.minlen "$CHUNK" --replay.maxlen "$CHUNK"
         --dataset.length "$CHUNK" "${EXTRA[@]}")
fi
EXTRA=(--cookie_env.id "$ENV_ID" --cookie_env.random_start "$RANDOM_START" "${EXTRA[@]}")

# Make sure cookie-env's python>=3.11 venv exists and matches its lockfile.
uv sync --project "$ROOT/bridges/cookie_env" --frozen --quiet

summary=()
for seed in $SEEDS; do
  for m in $METHODS; do
    logdir="$LOGROOT/$m/seed$seed"
    # stdout goes next to the logdir: creating the logdir itself would make
    # run_goal_cond.py think it's resuming and look for config.yaml.
    out="$logdir.stdout.log"
    mkdir -p "$(dirname "$logdir")"
    echo "=== $m seed $seed -> $logdir"
    # shellcheck disable=SC2046
    if "$ROOT/scripts/run.sh" python "$ROOT/examples/run_goal_cond.py" \
        --configs "${CONFIGS[@]}" --jit True --seed "$seed" --logdir "$logdir" \
        $(method_args "$m") "${EXTRA[@]}" >> "$out" 2>&1; then
      status=ok
    else
      status="FAILED (see $out)"
    fi
    # Last evaluation's success rates (reached goal at any step / on it at the end).
    last=$("$ROOT/.venv/bin/python" - "$logdir/metrics.jsonl" <<'EOF' 2>/dev/null || true
import json, sys
anyk, fink = 'max_eval_metric_success/goal_all', 'final_eval_metric_success/goal_all'
last = None
for line in open(sys.argv[1]):
    r = json.loads(line)
    if anyk in r:
        last = r
    elif last is not None and fink in r and r['step'] == last['step']:
        last.update(r)
if last:
    print(f"step {last['step']}: success_any {last[anyk]:.2f}, success_final {last.get(fink, float('nan')):.2f}")
EOF
)
    summary+=("$m seed$seed: $status ${last:+| $last}")
    echo "    $status ${last:+| $last}"
  done
done

echo; echo "=== Summary"; printf '%s\n' "${summary[@]}"
echo "TensorBoard: scripts/run.sh tensorboard --logdir $LOGROOT"
