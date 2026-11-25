#!/usr/bin/env bash
set -euo pipefail

# WHU-H2SR-MT supplementary experiments.
#
# Dataset: WHU_H2SR_MT_128 (128x128 patches, 2,281 train / 250 val,
# block-wise 90.1/9.9 split with block 8 held out for validation).
#
# HSI has 249 bands.  Sentinel-2 is reduced to a 10-channel temporal-median
# composite (B4,B3,B2,B8,B5,B6,B7,B8a,B11,B12) in msi/.  The first four
# channels are RGB+NIR; the remaining six occupy the "SAR" slots of the 4B and
# HighDAN models.  They are Sentinel-2 MSI bands, not real SAR.
#
# Formal runs use 100 epochs:
#   floor(2281 / 8) = 285 iters/epoch -> 28,500 iters at batch size 8.
# Smoke runs use the same configs with --iters 100.
#
# Usage:
#   bash exp_h2sr.sh --smoke
#   bash exp_h2sr.sh
#   bash exp_h2sr.sh --seeds 1919810,1919811,1919812
#   bash exp_h2sr.sh --resume
#   bash exp_h2sr.sh --models unet,cmx,mrsfn
#   bash exp_h2sr.sh --gpu 1
#   bash exp_h2sr.sh --dry-run
#
# Output:
#   output/h2sr/<tag>/        seed 1919810
#   output/h2sr/<tag>_seed... other seeds
#   log/h2sr/<tag>/
#   smoke_test/h2sr/...

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
WHU-H2SR-MT supplementary experiments.

Usage:
  bash exp_h2sr.sh --smoke
  bash exp_h2sr.sh
  bash exp_h2sr.sh --seeds 1919810,1919811,1919812
  bash exp_h2sr.sh --resume
  bash exp_h2sr.sh --models unet,cmx,mrsfn
  bash exp_h2sr.sh --gpu 1
  bash exp_h2sr.sh --dry-run

Options:
  --smoke        100-iteration smoke test; outputs to smoke_test/h2sr
  --resume       resume from the latest iter_* checkpoint
  --seeds CSV    training seeds (default: 1919810)
  --models CSV   comma-separated tags or config stems
  --gpu N        set CUDA_VISIBLE_DEVICES
  --dry-run      print commands without running
  --help         show this help

Formal runs use 28,500 iters (100 epochs at batch 8, 2,281 train patches).
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

smoke_iters="${H2SR_SMOKE_ITERS:-100}"
formal_iters="${H2SR_TARGET_ITERS:-28500}"
base_seed="${H2SR_BASE_SEED:-1919810}"

if [[ -n "${seeds_filter}" ]]; then
  IFS=',' read -r -a h2sr_seeds <<< "${seeds_filter}"
else
  h2sr_seeds=(1919810)
fi
for seed in "${h2sr_seeds[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "[H2SR] invalid seed: '${seed}'" >&2
    exit 2
  fi
done

if $smoke_mode; then
  output_root="smoke_test/h2sr/output"
  log_root="smoke_test/h2sr/log"
  train_args=(--iters "${smoke_iters}")
  target_iters="${smoke_iters}"
else
  output_root="output/h2sr"
  log_root="log/h2sr"
  train_args=(--iters "${formal_iters}")
  target_iters="${formal_iters}"
fi

# config stem | short tag
h2sr_models=(
  "unet_H2SR|unet"
  "deeplabv3p_H2SR|deeplabv3p"
  "ocrnet_H2SR|ocrnet"
  "segformer_H2SR|segformer"
  "highdan_H2SR|highdan"
  "cmx_4b_H2SR|cmx"
  "upernet_H2SR|upernet"
  "MRSN_H2SR|mrsn"
  "MRSFN_H2SR|mrsfn"
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

echo "[H2SR] mode=$($smoke_mode && echo smoke || echo formal) target_iters=${target_iters}"
echo "[H2SR] seeds=${h2sr_seeds[*]} (base_seed=${base_seed})"
echo "[H2SR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unchanged>}"
if [[ -f "${repo_root}/.env" ]] && ! grep -q '^H2SR_ROOT=' "${repo_root}/.env"; then
  echo "[H2SR] warning: H2SR_ROOT is not defined in .env" >&2
fi

matched=0
for seed in "${h2sr_seeds[@]}"; do
  if [[ "${seed}" == "${base_seed}" ]]; then
    seed_suffix=""
  else
    seed_suffix="_seed${seed}"
  fi

  for model_spec in "${h2sr_models[@]}"; do
    IFS='|' read -r config_stem tag <<< "${model_spec}"
    if ! model_selected "${tag}" "${config_stem}"; then
      continue
    fi
    matched=$((matched + 1))

    config_path="PaddleCD/c2seg_config/${config_stem}.yml"
    if [[ ! -f "${config_path}" ]]; then
      echo "[H2SR] missing config: ${config_path}" >&2
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
          echo "[H2SR] seed=${seed} ${tag}: complete at iter_${ckpt_iter}; skip"
          continue
        fi
        echo "[H2SR] seed=${seed} ${tag}: resume from iter_${ckpt_iter}"
        resume_args=(--resume_model "${ckpt}")
      else
        echo "[H2SR] seed=${seed} ${tag}: no resumable checkpoint; start"
      fi
    else
      echo "[H2SR] seed=${seed} ${tag}: start (target ${target_iters} iters)"
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

    printf '[H2SR] %s\n' "${cmd[*]}"
    if $dry_run; then
      continue
    fi

    start_ts="$(date +%s)"
    "${cmd[@]}"
    end_ts="$(date +%s)"
    echo "[H2SR] seed=${seed} ${tag}: finished in $((end_ts - start_ts))s"
  done
done

if [[ "${matched}" -eq 0 ]]; then
  echo "[H2SR] no models matched --models '${models_filter}'" >&2
  exit 2
fi

echo "[H2SR] all requested runs completed (${matched} run(s))"
