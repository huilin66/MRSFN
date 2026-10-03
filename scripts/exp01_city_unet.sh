#!/usr/bin/env bash
set -euo pipefail

# EXP-01 (U-Net only): train on the corrected Beijing-train / Wuhan-validation
# city split. This is intentionally a standalone entry point while the set of
# city-split models is still being finalized.
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

if $smoke_mode; then
  city_unet_save_dir="smoke_test/exp01_city_unet/output/unet_BW_city"
  city_unet_log_dir="smoke_test/exp01_city_unet/log"
  city_unet_train_args=(--iters 100)
  city_unet_target_iters=100
else
  # Keep this separate from the historical EXP-01 all-model output, which may
  # contain checkpoints trained before the SAR preprocessing fix.
  city_unet_save_dir="output/exp01_city_unet_train_B_val_W"
  city_unet_log_dir="log/exp01_city_unet"
  city_unet_train_args=()
  city_unet_target_iters=11500
fi

resume_args=()
if $resume_mode; then
  ckpt="$(latest_iter_ckpt "${city_unet_save_dir}")"
  if [[ -n "${ckpt}" ]]; then
    ckpt_iter="$(checkpoint_iter "${ckpt}")"
    if checkpoint_reaches_target "${ckpt}" "${city_unet_target_iters}"; then
      echo "[EXP-01-U-Net] checking: complete at iter_${ckpt_iter}; skip"
      exit 0
    fi
    echo "[EXP-01-U-Net] checking: resume from iter_${ckpt_iter}"
    resume_args=(--resume_model "${ckpt}")
  else
    echo "[EXP-01-U-Net] checking: no resumable checkpoint; start"
  fi
else
  echo "[EXP-01-U-Net] train corrected city split: target iter ${city_unet_target_iters}"
fi

python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/unet_BW_city.yml \
  --save_dir "${city_unet_save_dir}" \
  --log_dir "${city_unet_log_dir}" \
  "${city_unet_train_args[@]}" \
  "${resume_args[@]}" \
  --do_eval
