#!/usr/bin/env bash
set -euo pipefail

# MDAS (Augsburg) supplementary experiments.
#
# Dataset: mdas_clip (64x64 patches, 277 train / 31 val, patch-level random
# 90/10 split).  Formal runs use the 100-epoch schedule from the configs:
#   floor(277 / 16) = 17 iters/epoch -> 1700 iters at batch size 16.
# The same configs are used for the smoke test, with --iters 100.
#
# Usage:
#   bash exp_mdas.sh --smoke                 # fast smoke test, all baselines
#   bash exp_mdas.sh                         # formal 100-epoch training
#   bash exp_mdas.sh --resume                # resume interrupted runs
#   bash exp_mdas.sh --models unet,cmx,mrsfn # run selected models only
#   bash exp_mdas.sh --gpu 1                 # set CUDA_VISIBLE_DEVICES
#   bash exp_mdas.sh --dry-run --smoke       # print commands, run nothing
#
# Output:
#   formal: output/mdas/<tag>/   log/mdas/<tag>/
#   smoke : smoke_test/mdas/output/<tag>/   smoke_test/mdas/log/<tag>/
#
# PaddleSeg loads MDAS_CLIP_ROOT from the repository .env; make sure it points
# at the mdas_clip directory before running.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
cd "${repo_root}"

source "${repo_root}/scripts/resume_helpers.sh"

smoke_mode=false
resume_mode=false
dry_run=false
gpu=""
models_filter=""

usage() {
  cat <<'EOF'
MDAS (Augsburg) supplementary experiments.

Usage:
  bash exp_mdas.sh --smoke
  bash exp_mdas.sh
  bash exp_mdas.sh --resume
  bash exp_mdas.sh --models unet,cmx,mrsfn
  bash exp_mdas.sh --gpu 1
  bash exp_mdas.sh --dry-run --smoke

Options:
  --smoke        100-iteration smoke test; outputs to smoke_test/mdas
  --resume       resume from the latest iter_* checkpoint
  --models CSV   comma-separated tags or config stems
  --gpu N        set CUDA_VISIBLE_DEVICES
  --dry-run      print commands without running
  --help         show this help

Formal runs use 1700 iters (100 epochs at batch 16, 277 train patches).
The same configs are used for smoke, with --iters 100.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke_mode=true; shift ;;
    --resume) resume_mode=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --gpu)
      gpu="${2:?--gpu requires a value}"
      shift 2
      ;;
    --models)
      models_filter="${2:?--models requires a comma-separated list}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${gpu}" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi

smoke_iters="${MDAS_SMOKE_ITERS:-100}"
formal_iters="${MDAS_TARGET_ITERS:-1700}"

if $smoke_mode; then
  output_root="smoke_test/mdas/output"
  log_root="smoke_test/mdas/log"
  train_args=(--iters "${smoke_iters}")
  target_iters="${smoke_iters}"
else
  output_root="output/mdas"
  log_root="log/mdas"
  train_args=(--iters "${formal_iters}")
  target_iters="${formal_iters}"
fi

# config stem | short tag
mdas_models=(
  "unet_MDAS|unet"
  "deeplabv3p_MDAS|deeplabv3p"
  "ocrnet_MDAS|ocrnet"
  "segformer_MDAS|segformer"
  "highdan_MDAS|highdan"
  "cmx_4b_MDAS|cmx"
  "upernet_MDAS|upernet"
  "MRSN_MDAS|mrsn"
  "MRSFN_MDAS|mrsfn"
)

model_selected() {
  local tag="$1"
  local stem="$2"
  [[ -z "${models_filter}" ]] && return 0
  local item
  local IFS=','
  for item in ${models_filter}; do
    item="${item// /}"
    if [[ "${item}" == "${tag}" || "${item}" == "${stem}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "[MDAS] mode=$($smoke_mode && echo smoke || echo formal) target_iters=${target_iters}"
echo "[MDAS] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unchanged>}"
if [[ -f "${repo_root}/.env" ]] && ! grep -q '^MDAS_CLIP_ROOT=' "${repo_root}/.env"; then
  echo "[MDAS] warning: MDAS_CLIP_ROOT is not defined in .env" >&2
fi

matched=0
for model_spec in "${mdas_models[@]}"; do
  IFS='|' read -r config_stem tag <<< "${model_spec}"
  if ! model_selected "${tag}" "${config_stem}"; then
    continue
  fi
  matched=$((matched + 1))

  config_path="PaddleCD/c2seg_config/${config_stem}.yml"
  if [[ ! -f "${config_path}" ]]; then
    echo "[MDAS] missing config: ${config_path}" >&2
    exit 1
  fi

  save_dir="${output_root}/${tag}"
  log_dir="${log_root}/${tag}"
  resume_args=()

  if $resume_mode; then
    ckpt="$(latest_iter_ckpt "${save_dir}")"
    if [[ -n "${ckpt}" ]]; then
      ckpt_iter="$(checkpoint_iter "${ckpt}")"
      if checkpoint_reaches_target "${ckpt}" "${target_iters}"; then
        echo "[MDAS] ${tag}: complete at iter_${ckpt_iter}; skip"
        continue
      fi
      echo "[MDAS] ${tag}: resume from iter_${ckpt_iter}"
      resume_args=(--resume_model "${ckpt}")
    else
      echo "[MDAS] ${tag}: no resumable checkpoint; start"
    fi
  else
    echo "[MDAS] ${tag}: start (target ${target_iters} iters)"
  fi

  cmd=(
    python PaddleCD/train.py
    --config "${config_path}"
    --save_dir "${save_dir}"
    --log_dir "${log_dir}"
    --seed 1919810
    "${train_args[@]}"
    "${resume_args[@]}"
    --do_eval
  )

  printf '[MDAS] %s\n' "${cmd[*]}"
  if $dry_run; then
    continue
  fi

  start_ts="$(date +%s)"
  "${cmd[@]}"
  end_ts="$(date +%s)"
  echo "[MDAS] ${tag}: finished in $((end_ts - start_ts))s"
done

if [[ "${matched}" -eq 0 ]]; then
  echo "[MDAS] no models matched --models '${models_filter}'" >&2
  exit 2
fi

echo "[MDAS] all requested runs completed"
