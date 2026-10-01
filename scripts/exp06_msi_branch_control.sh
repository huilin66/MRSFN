#!/usr/bin/env bash
set -euo pipefail

# EXP-06: control the overlapping MSI branch design with four non-overlapping
# streams: RGB | NIR | SAR | HSI. The model, data split, loss, budget, and
# seed follow the ordinary 4B BW experiment; only the optical branch split
# changes.
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

exp06_model="cxup_4b_BW_RGB_NIR"
exp06_seed=1919810

if $smoke_mode; then
  exp06_save_dir="smoke_test/exp06/output/${exp06_model}"
  exp06_log_dir="smoke_test/exp06/log"
  exp06_train_args=(--iters 100)
else
  exp06_save_dir="output/${exp06_model}_exp06"
  exp06_log_dir="log/exp06"
  exp06_train_args=()
fi

resume_args=()
if $resume_mode; then
  ckpt="$(latest_iter_ckpt "${exp06_save_dir}")"
  if [[ -n "${ckpt}" ]]; then
    echo "[EXP-06] resuming ${exp06_model} from ${ckpt}"
    resume_args=(--resume_model "${ckpt}")
  else
    echo "[EXP-06] no checkpoint to resume for ${exp06_model}; training from scratch"
  fi
fi

python PaddleCD/train.py \
  --config "PaddleCD/c2seg_config/${exp06_model}.yml" \
  --save_dir "${exp06_save_dir}" \
  --seed "${exp06_seed}" \
  "${exp06_train_args[@]}" \
  --log_dir "${exp06_log_dir}" \
  "${resume_args[@]}" \
  --do_eval
