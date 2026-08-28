# MRSN / MBFM

This repository provides the PaddlePaddle/PaddleSeg implementation for two multimodal remote sensing semantic segmentation works:

- **Multimodal Remote Sensing Network (MRSN)**, WHISPERS 2023.
- **Multi-Branch Fusion Model (MBFM)**, an extended multi-branch fusion framework with Pixel-wise Modality Reliability Gate (PMRG) and mixed loss.

The code supports training, validation, prediction, and experimental analysis on the C2Seg-BW multimodal remote sensing segmentation setting.

## Repository Layout

```text
PaddleCD/                 PaddleSeg-based training, validation, prediction code
PaddleCD/c2seg_config/    Experiment configs for MRSN/MBFM variants and baselines
tools/                    Analysis, visualization, split, and post-processing scripts
pic/                      Repository-level architecture image
docs/                     Reproducibility notes
upload/                   Release-ready model weights and train/validation logs
```

## Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/huilin66/MRSN.git
cd MRSN
pip install -r PaddleCD/requirements.txt
```

Install a PaddlePaddle GPU build that matches your CUDA environment. The experiments used an RTX 6000 GPU with CUDA 12.6.

For quick checks, verify that PaddlePaddle can be imported:

```bash
python -c "import paddle; print(paddle.__version__); print(paddle.device.get_device())"
```

## Data Preparation

The experiments use C2Seg-BW from the 2023 IEEE WHISPERS Cross-City Semantic Segmentation Challenge. The dataset is not redistributed in this repository.

Create a local `.env` file in the repository root and define the dataset root:

```bash
C2SEG_BW_ROOT=/path/to/C2Seg_BW/train
```

`PaddleCD/c2seg_config/C2Seg_BW.yml` reads this variable:

```yaml
train_dataset:
  dataset_root: ${C2SEG_BW_ROOT}
  train_path: ${C2SEG_BW_ROOT}\train.txt
val_dataset:
  dataset_root: ${C2SEG_BW_ROOT}
  val_path: ${C2SEG_BW_ROOT}\val.txt
```

The `.env` file is ignored by git so private dataset paths are not published.

## Geographic-Leakage Experiment (City-Level Split)

A reviewer concern was that the default random patch split can leak geography
between train/val (patches from the same full scene land in both sets). To
evaluate this, a **geographically disjoint** split can be built with

```bash
python tools/build_city_split_dataset.py --split train_B_val_W
```

The scene root is taken automatically from `C2SEG_BW_ROOT` in `.env` (the
full-scene TIFFs live in `.../C2Seg/src/tif_BW`), so no path needs to be
passed. `train_B_val_W` crops training patches from the **Beijing** full scene
and validation patches from the **Wuhan** full scene; swap the direction with
`--split train_W_val_B`. Output defaults to `data/C2Seg_BW_city_<split>` and
uses the same `msisar/hsi/lbl` layout + `train.txt` / `val.txt` expected by
`RS_MD3B`, rescaling full-scene HSI (0-1 reflectance) by 10000 to the patch DN
range.

Then point a config at it:

```bash
C2SEG_BW_CITY_ROOT=<abs/path>       # add to .env
python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/cxup_4b_BW_PMRG_v2_lossV2_city.yml \
  --save_dir output/city/... \
  --do_eval
```

## Training

Run training from the repository root:

```bash
python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml \
  --save_dir output/cxup_4b_BW_PMRG_v2_lossV2 \
  --do_eval
```

You can also run the notebook workflow in `main.ipynb`; `aug.ipynb` provides additional examples for model/config path setup.

## Validation

Run validation with a trained checkpoint:

```bash
python PaddleCD/val.py \
  --config PaddleCD/c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml \
  --model_path output/cxup_4b_BW_PMRG_v2_lossV2/best_model/model.pdparams \
  --batch_size 1
```

For notebook-style validation, refer to the command patterns in `aug.ipynb` and use the same config and checkpoint arguments shown above.

Additional analysis scripts for CAM, t-SNE, per-image mIoU, and summary tables are in `tools/`. See `docs/REPRODUCIBILITY.md` for more details.

## Released Models and Logs

The `upload/` directory is organized by model. Each model folder contains:

```text
model.pdparams    Best validation checkpoint
train.log         Training log
val.log           Validation log
```

Download links:

- Baidu Netdisk: https://pan.baidu.com/s/17bEaPhn7ICAIUG4jpm155A?pwd=a5g6, extraction code: `a5g6`
- Google Drive: https://drive.google.com/drive/folders/1PZLc_urcnLuYBfxbmJhZkUeadLpmFYAc?usp=sharing

When distributing through Google Drive or Baidu Netdisk, keep this per-model folder structure so weights and logs stay paired.

## Citation

If you use the original **Multimodal Remote Sensing Network (MRSN)** model or its released results, please cite:

```bibtex
@inproceedings{zhao2023multimodal,
  title={Multimodal remote sensing network},
  author={Zhao, Huilin and Chen, Chuan and Xia, Cong},
  booktitle={2023 13th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement

Thanks to PaddleSeg for providing the segmentation framework used in this codebase.

## License

Apache License 2.0.
