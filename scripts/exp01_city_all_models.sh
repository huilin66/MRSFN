#!/usr/bin/env bash
set -euo pipefail

# EXP-01: train all models on the Beijing-train / Wuhan-validation split, then
# run tiled inference on the complete Wuhan scene for every city-disjoint
# checkpoint.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

# The first model in EXP-01.
python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/unet_BW_city.yml \
  --save_dir output/unet_BW_city_train_B_val_W \
  --do_eval

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

for model in "${city_models[@]}"; do
  python PaddleCD/train.py \
    --config "PaddleCD/c2seg_config/${model}.yml" \
    --save_dir "output/${model}_train_B_val_W" \
    --do_eval
done

# EXP-01 continuation: tiled inference on the complete Wuhan scene for every
# city-disjoint checkpoint produced above.
all_city_models=(
  unet_BW_city
  "${city_models[@]}"
)

for model in "${all_city_models[@]}"; do
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
done
