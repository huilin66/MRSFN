#!/usr/bin/env bash
set -euo pipefail

# EXP-06: control the overlapping MSI branch design with four non-overlapping
# streams: RGB | NIR | SAR | HSI. Run the control on BW and AB with the same
# three seeds used by EXP-03. The existing BW seed 1919810 is preserved in
# its original output directory and is skipped when --resume is used.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

source "${script_dir}/resume_helpers.sh"

smoke_mode=false
resume_mode=false
for arg in "$@"; do
  case "$arg" in
    --smoke) smoke_mode=true ;;
    --resume) resume_mode=true ;;
    *) echo "Usage: $0 [--smoke] [--resume]" >&2; exit 2 ;;
  esac
done
exp06_conditions=(BW AB)
exp06_seeds=(1919810 1919811 1919812)

for condition in "${exp06_conditions[@]}"; do
  if [[ "${condition}" == "BW" ]]; then
    exp06_model="cxup_4b_BW_RGB_NIR"
    exp06_config="PaddleCD/c2seg_config/cxup_4b_BW_RGB_NIR.yml"
    exp06_target_iters=40000
  else
    exp06_model="cxup_4b_AB_RGB_NIR"
    exp06_config="PaddleCD/c2seg_config/cxup_4b_AB_RGB_NIR.yml"
    exp06_target_iters=1600
  fi

  for exp06_seed in "${exp06_seeds[@]}"; do
    if $smoke_mode; then
      exp06_save_dir="smoke_test/exp06/output/${exp06_model}_seed${exp06_seed}"
      exp06_log_dir="smoke_test/exp06/log/${condition}_seed${exp06_seed}"
      exp06_train_args=(--iters 100)
      exp06_target_iters=100
    elif [[ "${condition}" == "BW" && "${exp06_seed}" == "1919810" ]]; then
      # Keep the completed run and its original log/checkpoint locations.
      exp06_save_dir="output/cxup_4b_BW_RGB_NIR_exp06"
      exp06_log_dir="log/exp06"
      exp06_train_args=()
    else
      exp06_save_dir="output/exp06_${exp06_model}_seed${exp06_seed}"
      exp06_log_dir="log/exp06/${condition}_seed${exp06_seed}"
      exp06_train_args=()
    fi

    resume_args=()
    if $resume_mode; then
      ckpt="$(latest_iter_ckpt "${exp06_save_dir}")"
      if [[ -n "${ckpt}" ]]; then
        ckpt_iter="$(checkpoint_iter "${ckpt}")"
        if checkpoint_reaches_target "${ckpt}" "${exp06_target_iters}"; then
          echo "[EXP-06] checking ${condition} seed ${exp06_seed}: complete at iter_${ckpt_iter}; skip"
          continue
        fi
        echo "[EXP-06] checking ${condition} seed ${exp06_seed}: resume from iter_${ckpt_iter}"
        resume_args=(--resume_model "${ckpt}")
      else
        echo "[EXP-06] checking ${condition} seed ${exp06_seed}: no resumable checkpoint; start"
      fi
    else
      echo "[EXP-06] condition=${condition}, seed=${exp06_seed}"
    fi

    python PaddleCD/train.py \
      --config "${exp06_config}" \
      --save_dir "${exp06_save_dir}" \
      --seed "${exp06_seed}" \
      "${exp06_train_args[@]}" \
      --log_dir "${exp06_log_dir}" \
      "${resume_args[@]}" \
      --do_eval
  done
done
