#!/usr/bin/env bash
set -euo pipefail

# EXP-12: unified efficiency benchmark (Params / FLOPs / FPS) for every model
# row in the paper, measured in a single environment within one run. This
# replaces the per-run FPS values that were logged incidentally during
# validation on different hardware. No checkpoints or datasets are required;
# models are built from the c2seg configs and timed on synthetic inputs.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

preset="all"
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      preset="$2"
      shift 2
      ;;
    *)
      passthrough+=("$1")
      shift
      ;;
  esac
done

python PaddleCD/benchmark_efficiency.py \
  --preset "${preset}" \
  "${passthrough[@]+"${passthrough[@]}"}"
