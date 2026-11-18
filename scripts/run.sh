#!/usr/bin/env bash
# Run a command inside the uv venv with the CUDA libs TF 2.4 needs.
#   scripts/run.sh python examples/run_goal_cond.py --configs peg_point_maze test_maze ...
# Env vars honoured: MRL_PATH (default ../mrl), LEXA_PATH (if set, used instead of mrl).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
SITE="$VENV/lib/python3.8/site-packages"

# pip-installed CUDA 11 / cuDNN 8 libraries (from the `gpu` extra).
LIBS=""
for d in "$SITE"/nvidia/*/lib; do LIBS="$LIBS:$d"; done

# TF 2.4 dlopens libcusolver.so.10 (CUDA 11.0 naming); the API-compatible
# CUDA 11.x library is libcusolver.so.11.
SHIM="$VENV/cuda-shim"
if [[ -f "$SITE/nvidia/cusolver/lib/libcusolver.so.11" && ! -e "$SHIM/libcusolver.so.10" ]]; then
  mkdir -p "$SHIM"
  ln -s "$SITE/nvidia/cusolver/lib/libcusolver.so.11" "$SHIM/libcusolver.so.10"
fi
export LD_LIBRARY_PATH="$SHIM$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Ada GPUs have no SASS in TF 2.4: kernels are JIT-compiled from PTX once and cached.
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
# Headless plotting (uv's standalone CPython has a broken Tk backend).
export MPLBACKEND="${MPLBACKEND:-Agg}"

ENV_REPO="${LEXA_PATH:-${MRL_PATH:-$ROOT/../mrl}}"
export PYTHONPATH="$ENV_REPO${PYTHONPATH:+:$PYTHONPATH}"

# Old torch (1.10, imported by mrl) ships libs flagged with an executable stack,
# which glibc >= 2.41 refuses to load by default.
export GLIBC_TUNABLES="glibc.rtld.execstack=2${GLIBC_TUNABLES:+:$GLIBC_TUNABLES}"

export PATH="$VENV/bin:$PATH"
exec "$@"
