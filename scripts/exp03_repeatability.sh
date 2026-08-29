#!/usr/bin/env bash
set -euo pipefail

# EXP-03: repeatability of the branch-count trend and the 4B component gains.
# All runs use the existing ordinary BW protocol from C2Seg_BW.yml. Only the
# seed and the experimental condition vary. The 4B baseline is shared by both
# evidence chains, so the per-seed deltas remain directly comparable.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

exp03_seeds=(1919810 1919811 1919812)
exp03_models=(
  cxup_1b_BW
  cxup_2b_BW
  cxup_3b_BW
  cxup_4b_BW
  cxup_4b_BW_PMRG
  cxup_4b_BW_loss
  cxup_4b_BW_PMRG_v2_lossV2
)

for seed in "${exp03_seeds[@]}"; do
  for model in "${exp03_models[@]}"; do
    echo "[EXP-03] condition=${model}, seed=${seed}"
    python PaddleCD/train.py \
      --config "PaddleCD/c2seg_config/${model}.yml" \
      --save_dir "output/exp03_${model}_seed${seed}" \
      --log_dir "log/exp03/${model}_seed${seed}" \
      --seed "${seed}" \
      --do_eval
  done
done
