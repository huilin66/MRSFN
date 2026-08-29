#!/usr/bin/env bash
set -euo pipefail

# EXP-03: repeatability of the branch-count trend and the 4B component gains.
# All runs use the existing ordinary BW protocol from C2Seg_BW.yml. Only the
# seed and the experimental condition vary. The 4B baseline is shared by both
# evidence chains, so the per-seed deltas remain directly comparable.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

case "${1:-}" in
  "") smoke_mode=false ;;
  --smoke) smoke_mode=true ;;
  *) echo "Usage: $0 [--smoke]" >&2; exit 2 ;;
esac

if $smoke_mode; then
  exp03_output_root="smoke_test/exp03/output"
  exp03_log_prefix="smoke_test/exp03/log"
  exp03_train_args=(--iters 100)
else
  exp03_output_root="output"
  exp03_log_prefix="log/exp03"
  exp03_train_args=()
fi

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
      --save_dir "${exp03_output_root}/exp03_${model}_seed${seed}" \
      --log_dir "${exp03_log_prefix}/${model}_seed${seed}" \
      --seed "${seed}" \
      "${exp03_train_args[@]}" \
      --do_eval
  done
done
