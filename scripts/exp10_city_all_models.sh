#!/usr/bin/env bash
set -euo pipefail

# EXP-10: train all models on the reversed city split (Wuhan-train / Beijing-
# val), then run tiled inference on the complete Beijing scene. Mirror of EXP-01
# (exp01_city_all_models.sh) using the *_city_WB configs and *_train_W_val_B
# output layout, so the two geographic directions stay directly comparable.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "$repo_root"

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
  exp10_output_root="smoke_test/exp10/output"
  exp10_model_suffix=""
  exp10_log_args=(--log_dir "smoke_test/exp10/log")
  exp10_scene_output="smoke_test/exp10/full_scene_train_W_val_B"
  exp10_train_args=(--iters 100)
else
  exp10_output_root="output"
  exp10_model_suffix="_train_W_val_B"
  exp10_log_args=()
  exp10_scene_output="ana/full_scene_city_train_W_val_B"
  exp10_train_args=()
fi

# The first model in EXP-10.
exp10_save_dir="${exp10_output_root}/unet_BW_city_WB${exp10_model_suffix}"
resume_args=()
if $resume_mode; then
  ckpt="$(latest_iter_ckpt "${exp10_save_dir}")"
  if [[ -n "${ckpt}" ]]; then
    echo "[EXP-10] resuming unet_BW_city_WB from ${ckpt}"
    resume_args=(--resume_model "${ckpt}")
  else
    echo "[EXP-10] no checkpoint to resume for unet_BW_city_WB; training from scratch"
  fi
fi
python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/unet_BW_city_WB.yml \
  --save_dir "${exp10_save_dir}" \
  "${exp10_train_args[@]}" \
  "${exp10_log_args[@]}" \
  "${resume_args[@]}" \
  --do_eval

# Remaining models/variants with a *_BW_city_WB.yml configuration. The base
# C2Seg_BW_city_WB.yml is inherited by these files and is not itself a model.
city_wb_models=(
  deeplabv3p_BW_city_WB
  ocrnet_BW_city_WB
  ocrnetW48_BW_city_WB
  segformer_BW_city_WB
  highdan_BW_city_WB
  MRSN_BW_city_WB
  cxup_1b_BW_city_WB
  cxup_2b_BW_city_WB
  cxup_3b_BW_city_WB
  cxup_4b_BW_city_WB
  cxup_4b_BW_loss_city_WB
  cxup_4b_BW_PMRG_city_WB
  cxup_4b_BW_PMRG_ML_city_WB
  cxup_4b_BW_PMRG_v2_lossV2_city_WB
)

for model in "${city_wb_models[@]}"; do
  exp10_save_dir="${exp10_output_root}/${model}${exp10_model_suffix}"
  resume_args=()
  if $resume_mode; then
    ckpt="$(latest_iter_ckpt "${exp10_save_dir}")"
    if [[ -n "${ckpt}" ]]; then
      echo "[EXP-10] resuming ${model} from ${ckpt}"
      resume_args=(--resume_model "${ckpt}")
    else
      echo "[EXP-10] no checkpoint to resume for ${model}; training from scratch"
    fi
  fi
  python PaddleCD/train.py \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --save_dir "${exp10_save_dir}" \
    "${exp10_train_args[@]}" \
    "${exp10_log_args[@]}" \
    "${resume_args[@]}" \
    --do_eval
done

# EXP-10 continuation: tiled inference on the complete Beijing scene for every
# city-disjoint checkpoint produced above.
all_city_wb_models=(
  unet_BW_city_WB
  "${city_wb_models[@]}"
)

for model in "${all_city_wb_models[@]}"; do
  python tools/infer_full_scene.py \
    --dataset BW \
    --scene beijing \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --model_path "${exp10_output_root}/${model}${exp10_model_suffix}/best_model/model.pdparams" \
    --output_dir "${exp10_scene_output}" \
    --crop_size 256 256 \
    --stride 256 256 \
    --batch_size 4 \
    --device gpu
done
