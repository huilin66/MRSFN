#!/usr/bin/env bash
set -euo pipefail

# MDAS (Augsburg) supplementary experiments.
#
# Dataset: mdas_clip (64x64 patches, 277 train / 31 val, patch-level random
# 90/10 split).  The data split is fixed; only the training seed varies.
#
# All baselines (including CMX) run with one seed (1919810) by default.
# Pass --seeds for extra seeds, e.g. --seeds 1919810,1919811,1919812.
# Formal runs use the 100-epoch schedule:
#   floor(277 / 16) = 17 iters/epoch -> 1700 iters at batch size 16.
# The same configs are used for the smoke test, with --iters 100.
#
# Usage:
#   bash exp_mdas.sh --smoke                      # smoke, 9 models
#   bash exp_mdas.sh                              # formal, 9 models
#   bash exp_mdas.sh --seeds 1919810,1919811,1919812  # optional repeatability
#   bash exp_mdas.sh --resume                     # resume interrupted runs
#   bash exp_mdas.sh --models unet,cmx,mrsfn      # run selected models only
#   bash exp_mdas.sh --gpu 1                      # set CUDA_VISIBLE_DEVICES
#   bash exp_mdas.sh --dry-run --seeds 1919810    # print commands, run nothing
#
# Output:
#   seed 1919810: output/mdas/<tag>/
#   other seeds : output/mdas/<tag>_seed<seed>/
#   smoke       : smoke_test/mdas/output/... (same naming)
#   logs        : log/mdas/... or smoke_test/mdas/log/...
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
seeds_filter=""

usage() {
  cat <<'EOF'
MDAS (Augsburg) supplementary experiments.

Usage:
  bash exp_mdas.sh --smoke
  bash exp_mdas.sh
  bash exp_mdas.sh --seeds 1919811,1919812
  bash exp_mdas.sh --resume
  bash exp_mdas.sh --models unet,cmx,mrsfn
  bash exp_mdas.sh --gpu 1
  bash exp_mdas.sh --dry-run --seeds 1919810

Options:
  --smoke        100-iteration smoke test; outputs to smoke_test/mdas
  --resume       resume from the latest iter_* checkpoint
  --seeds CSV    training seeds (default: 1919810)
  --models CSV   comma-separated tags or config stems
  --gpu N        set CUDA_VISIBLE_DEVICES
  --dry-run      print commands without running
  --help         show this help

Formal runs use 1700 iters (100 epochs at batch 16, 277 train patches).
The same configs are used for smoke, with --iters 100.
Seed 1919810 keeps unsuffixed output dirs; other seeds use _seed<seed>.
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
    --seeds)
      seeds_filter="${2:?--seeds requires a comma-separated list}"
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
base_seed="${MDAS_BASE_SEED:-1919810}"

if [[ -n "${seeds_filter}" ]]; then
  IFS=',' read -r -a mdas_seeds <<< "${seeds_filter}"
else
  mdas_seeds=(1919810)
fi
for seed in "${mdas_seeds[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[MDAS] invalid seed: '${seed}'" >&2
    exit 2
  fi
done

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
echo "[MDAS] seeds=${mdas_seeds[*]} (base_seed=${base_seed})"
echo "[MDAS] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unchanged>}"
if [[ -f "${repo_root}/.env" ]] && ! grep -q '^MDAS_CLIP_ROOT=' "${repo_root}/.env"; then
  echo "[MDAS] warning: MDAS_CLIP_ROOT is not defined in .env" >&2
fi

matched=0
for seed in "${mdas_seeds[@]}"; do
  if [[ "${seed}" == "${base_seed}" ]]; then
    seed_suffix=""
  else
    seed_suffix="_seed${seed}"
  fi

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

    save_dir="${output_root}/${tag}${seed_suffix}"
    log_dir="${log_root}/${tag}${seed_suffix}"
    resume_args=()

    if $resume_mode; then
      ckpt="$(latest_iter_ckpt "${save_dir}")"
      if [[ -n "${ckpt}" ]]; then
        ckpt_iter="$(checkpoint_iter "${ckpt}")"
        if checkpoint_reaches_target "${ckpt}" "${target_iters}"; then
          echo "[MDAS] seed=${seed} ${tag}: complete at iter_${ckpt_iter}; skip"
          continue
        fi
        echo "[MDAS] seed=${seed} ${tag}: resume from iter_${ckpt_iter}"
        resume_args=(--resume_model "${ckpt}")
      else
        echo "[MDAS] seed=${seed} ${tag}: no resumable checkpoint; start"
      fi
    else
      echo "[MDAS] seed=${seed} ${tag}: start (target ${target_iters} iters)"
    fi

    cmd=(
      python PaddleCD/train.py
      --config "${config_path}"
      --save_dir "${save_dir}"
      --log_dir "${log_dir}"
      --seed "${seed}"
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
    echo "[MDAS] seed=${seed} ${tag}: finished in $((end_ts - start_ts))s"
  done
done

if [[ "${matched}" -eq 0 ]]; then
  echo "[MDAS] no models matched --models '${models_filter}'" >&2
  exit 2
fi

echo "[MDAS] all requested runs completed (${matched} run(s))"
