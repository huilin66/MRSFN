#!/usr/bin/env bash
set -euo pipefail

# EXP-02: supplement the existing ordinary BW experiments with larger-backbone
# 1B models. ConvNeXt-Small approaches the existing 2B-Tiny parameter scale;
# ConvNeXt-Base approaches 3B-Tiny and is the closest standard 1B comparison
# for 4B-Tiny. The split, loss, crop, budget, and seed match the existing BW
# experiments; this is not part of the EXP-01 city-split run.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

case "${1:-}" in
  "") smoke_mode=false ;;
  --smoke) smoke_mode=true ;;
  *) echo "Usage: $0 [--smoke]" >&2; exit 2 ;;
esac

exp02_seed=1919810

if $smoke_mode; then
  exp02_output_root="smoke_test/exp02/output"
  exp02_model_suffix=""
  exp02_log_args=(--log_dir "smoke_test/exp02/log")
  exp02_train_args=(--iters 100)
else
  exp02_output_root="output"
  exp02_model_suffix="_exp02"
  exp02_log_args=()
  exp02_train_args=()
fi

for model in cxup_1b_BW_small cxup_1b_BW_base; do
  python PaddleCD/train.py \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --save_dir "${exp02_output_root}/${model}${exp02_model_suffix}" \
    --seed "$exp02_seed" \
    "${exp02_train_args[@]}" \
    "${exp02_log_args[@]}" \
    --do_eval
done
