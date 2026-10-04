#!/usr/bin/env bash
set -euo pipefail

# EXP-11: serial grid-split training for U-Net, 1B/UpderNet, and MRSFN.
# Dataset: C2SEG_BW_GRID_10X10_SEED15859, generated with grid_seed=15859.
# The three models intentionally use independent output/log directories.
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
    *)
      echo "Usage: $0 [--smoke] [--resume]" >&2
      exit 2
      ;;
  esac
done

if $smoke_mode; then
  exp11_train_args=(--iters 100)
  exp11_target_iters=100
else
  exp11_train_args=()
  exp11_target_iters=20200
fi

exp11_models=(
  "unet_BW_grid_10x10_seed15859|unet"
  "cxup_1b_BW_grid_10x10_seed15859|1b"
  "cxup_4b_BW_PMRG_ML_grid_10x10_seed15859|mrsfn"
)

for model_spec in "${exp11_models[@]}"; do
  IFS='|' read -r exp11_config_stem exp11_model_tag <<< "${model_spec}"
  exp11_config="PaddleCD/c2seg_config/${exp11_config_stem}.yml"
  exp11_save_dir="output/exp11_bw_grid_${exp11_model_tag}/${exp11_config_stem}"
  exp11_log_dir="log/exp11_bw_grid_${exp11_model_tag}"
  if $smoke_mode; then
    exp11_save_dir="smoke_test/exp11_bw_grid_${exp11_model_tag}/output/${exp11_config_stem}"
    exp11_log_dir="smoke_test/exp11_bw_grid_${exp11_model_tag}/log"
  fi

  resume_args=()
  if $resume_mode; then
    ckpt="$(latest_iter_ckpt "${exp11_save_dir}")"
    if [[ -n "${ckpt}" ]]; then
      ckpt_iter="$(checkpoint_iter "${ckpt}")"
      if checkpoint_reaches_target "${ckpt}" "${exp11_target_iters}"; then
        echo "[EXP-11] checking ${exp11_model_tag}: complete at iter_${ckpt_iter}; skip"
        continue
      fi
      echo "[EXP-11] checking ${exp11_model_tag}: resume from iter_${ckpt_iter}"
      resume_args=(--resume_model "${ckpt}")
    else
      echo "[EXP-11] checking ${exp11_model_tag}: no resumable checkpoint; start"
    fi
  else
    echo "[EXP-11] start ${exp11_model_tag}: target iter ${exp11_target_iters}"
  fi

  python PaddleCD/train.py \
    --config "${exp11_config}" \
    --save_dir "${exp11_save_dir}" \
    --log_dir "${exp11_log_dir}" \
    "${exp11_train_args[@]}" \
    "${resume_args[@]}" \
    --do_eval
done
