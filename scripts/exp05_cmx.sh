#!/usr/bin/env bash
set -euo pipefail

# EXP-05: CMX-adapted two-stream baseline using MiT-B4 in both streams.
# The input tensors remain identical to the ordinary BW/AB two-stream
# protocols; only the CMX backbone is fixed to MiT-B4.
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

exp05_seed=1919810
exp05_models=(cmx_4b_BW cmx_4b_AB)

if $smoke_mode; then
  exp05_output_root="smoke_test/exp05/output"
  exp05_log_root="smoke_test/exp05/log"
  exp05_model_suffix=""
  exp05_train_args=(--iters 100)
  exp05_target_iters=100
else
  exp05_output_root="output"
  exp05_log_root="log/exp05"
  exp05_model_suffix="_exp05"
  exp05_train_args=()
  exp05_target_iters=40000
fi

for model in "${exp05_models[@]}"; do
  exp05_save_dir="${exp05_output_root}/${model}${exp05_model_suffix}"
  exp05_log_dir="${exp05_log_root}/${model}"
  resume_args=()

  if $resume_mode; then
    ckpt="$(latest_iter_ckpt "${exp05_save_dir}")"
    if [[ -n "${ckpt}" ]]; then
      ckpt_iter="$(checkpoint_iter "${ckpt}")"
      if checkpoint_reaches_target "${ckpt}" "${exp05_target_iters}"; then
        echo "[EXP-05] checking ${model}: complete at iter_${ckpt_iter}; skip"
        continue
      fi
      echo "[EXP-05] checking ${model}: resume from iter_${ckpt_iter}"
      resume_args=(--resume_model "${ckpt}")
    else
      echo "[EXP-05] checking ${model}: no resumable checkpoint; start"
    fi
  else
    echo "[EXP-05] condition=${model}"
  fi

  python PaddleCD/train.py \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --save_dir "${exp05_save_dir}" \
    --log_dir "${exp05_log_dir}" \
    --seed "${exp05_seed}" \
    "${exp05_train_args[@]}" \
    "${resume_args[@]}" \
    --do_eval
done
