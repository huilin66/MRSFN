#!/usr/bin/env bash
set -euo pipefail

# EXP-11: U-Net on the 10x10 BW geographic-grid split generated with
# grid_seed=15859. The split contains both Beijing and Wuhan scenes in train
# and validation; validation cells are spatially disjoint from training cells.
# This standalone script intentionally does not modify .env or dispatch other
# models.
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

exp11_config="PaddleCD/c2seg_config/unet_BW_grid_10x10_seed15859.yml"

if $smoke_mode; then
  exp11_save_dir="smoke_test/exp11_bw_grid_unet/output/unet_BW_grid_10x10_seed15859"
  exp11_log_dir="smoke_test/exp11_bw_grid_unet/log"
  exp11_train_args=(--iters 100)
  exp11_target_iters=100
else
  exp11_save_dir="output/exp11_bw_grid_unet/unet_BW_grid_10x10_seed15859"
  exp11_log_dir="log/exp11_bw_grid_unet"
  exp11_train_args=()
  exp11_target_iters=20200
fi

resume_args=()
if $resume_mode; then
  ckpt="$(latest_iter_ckpt "${exp11_save_dir}")"
  if [[ -n "${ckpt}" ]]; then
    ckpt_iter="$(checkpoint_iter "${ckpt}")"
    if checkpoint_reaches_target "${ckpt}" "${exp11_target_iters}"; then
      echo "[EXP-11] checking: complete at iter_${ckpt_iter}; skip"
      exit 0
    fi
    echo "[EXP-11] checking: resume from iter_${ckpt_iter}"
    resume_args=(--resume_model "${ckpt}")
  else
    echo "[EXP-11] checking: no resumable checkpoint; start"
  fi
else
  echo "[EXP-11] train U-Net on BW 10x10 grid split seed 15859: target iter ${exp11_target_iters}"
fi

python PaddleCD/train.py \
  --config "${exp11_config}" \
  --save_dir "${exp11_save_dir}" \
  --log_dir "${exp11_log_dir}" \
  "${exp11_train_args[@]}" \
  "${resume_args[@]}" \
  --do_eval
