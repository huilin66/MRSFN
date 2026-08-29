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

exp02_seed=1919810

for model in cxup_1b_BW_small cxup_1b_BW_base; do
  python PaddleCD/train.py \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --save_dir "output/${model}_exp02" \
    --seed "$exp02_seed" \
    --do_eval
done
