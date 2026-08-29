#!/usr/bin/env bash

# EXP-01: train all models on the Beijing-train / Wuhan-validation split.
# The first executable command starts the first model; the remaining models
# are trained below with the same city-disjoint data and evaluation protocol.
# Run this file from the repository root in the configured Paddle GPU environment.
python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/unet_BW_city.yml \
  --save_dir output/unet_BW_city_train_B_val_W \
  --do_eval
train_status=$?
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

# Remaining models/variants with a *_BW_city.yml configuration. The base
# C2Seg_BW_city.yml is inherited by these files and is not itself a model.
city_models=(
  deeplabv3p_BW_city
  ocrnet_BW_city
  ocrnetW48_BW_city
  segformer_BW_city
  highdan_BW_city
  MRSN_BW_city
  cxup_1b_BW_city
  cxup_2b_BW_city
  cxup_3b_BW_city
  cxup_4b_BW_city
  cxup_4b_BW_loss_city
  cxup_4b_BW_PMRG_city
  cxup_4b_BW_PMRG_ML_city
  cxup_4b_BW_PMRG_v2_lossV2_city
)

run_train() {
  local model="$1"
  python PaddleCD/train.py \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --save_dir "output/${model}_train_B_val_W" \
    --do_eval
  local status=$?
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
}

for model in "${city_models[@]}"; do
  run_train "$model"
done

# EXP-01 continuation: tiled inference on the complete Wuhan scene for every
# city-disjoint checkpoint produced above.
all_city_models=(
  unet_BW_city
  "${city_models[@]}"
)

run_full_scene_inference() {
  local model="$1"
  python tools/infer_full_scene.py \
    --dataset BW \
    --scene wuhan \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --model_path "output/${model}_train_B_val_W/best_model/model.pdparams" \
    --output_dir ana/full_scene_city_train_B_val_W \
    --crop_size 256 256 \
    --stride 256 256 \
    --batch_size 4 \
    --device gpu
  local status=$?
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
}

for model in "${all_city_models[@]}"; do
  run_full_scene_inference "$model"
done
