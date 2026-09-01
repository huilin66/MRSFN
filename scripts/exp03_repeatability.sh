#!/usr/bin/env bash
set -euo pipefail

# EXP-03: repeatability of the branch-count trend and the 4B component gains.
# All runs use the existing ordinary BW protocol from C2Seg_BW.yml. Only the
# seed and the experimental condition vary. The 4B baseline is shared by both
# evidence chains, so the per-seed deltas remain directly comparable.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

source "${script_dir}/resume_helpers.sh"

smoke_mode=false
resume_mode=false
EXPO3_GROUP=""   # empty = run all seeds sequentially; 0|1|2 = this seed-parallel group only
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke_mode=true; shift ;;
    --resume) resume_mode=true; shift ;;
    --group) EXPO3_GROUP="$2"; shift 2 ;;
    --group=*) EXPO3_GROUP="${1#*=}"; shift ;;
    *) echo "Usage: $0 [--smoke] [--resume] [--group {0|1|2}]" >&2; exit 2 ;;
  esac
done

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

# Parallel-group mode: --group {0|1|2} selects ONE seed's 7-model block so 3
# terminals (one per group) train concurrently on the same GPU. Without it, all
# 21 runs execute sequentially (EXP-03's original behaviour, used by exp_add.sh).
if [[ -n "$EXPO3_GROUP" ]]; then
  case "$EXPO3_GROUP" in
    0|1|2) ;;
    *) echo "Usage: --group must be 0, 1, or 2" >&2; exit 2 ;;
  esac
  exp03_seeds=("${exp03_seeds[$EXPO3_GROUP]}")
  echo "[EXP-03] parallel group ${EXPO3_GROUP}: seed=${exp03_seeds[0]}, 7 models"
fi

for seed in "${exp03_seeds[@]}"; do
  for model in "${exp03_models[@]}"; do
    exp03_save_dir="${exp03_output_root}/exp03_${model}_seed${seed}"
    echo "[EXP-03] condition=${model}, seed=${seed}"
    resume_args=()
    if $resume_mode; then
      ckpt="$(latest_iter_ckpt "${exp03_save_dir}")"
      if [[ -n "${ckpt}" ]]; then
        echo "[EXP-03] resuming ${model} (seed ${seed}) from ${ckpt}"
        resume_args=(--resume_model "${ckpt}")
      else
        echo "[EXP-03] no checkpoint to resume for ${model} (seed ${seed}); training from scratch"
      fi
    fi
    python PaddleCD/train.py \
      --config "PaddleCD/c2seg_config/${model}.yml" \
      --save_dir "${exp03_save_dir}" \
      --log_dir "${exp03_log_prefix}/${model}_seed${seed}" \
      --seed "${seed}" \
      "${exp03_train_args[@]}" \
      "${resume_args[@]}" \
      --do_eval
  done
done
