# Experimental Results

This document records the single-seed BW and AB evaluation results currently
available for the revised manuscript. Multi-seed repeatability results are
reserved for a separate section and are not mixed with these single-seed
tables.

## 1. Reporting conventions

- Seed: `1919810`.
- BW source: `output_bw/` and the matched validation logs in `log/val_bw/`.
- AB source: `output_ab/` and the matched validation logs in `log/val_ab1/`.
- `val_ab2` is retained as a separate AB evaluation pass and is not mixed into
  the main AB tables.
- Checkpoint: `best_model/model.pdparams`.
- Metrics: mIoU, F1, accuracy, and Kappa.
- FPS is reported as logged and is dependent on hardware and runtime state.

## 2. Baseline comparison

### 2.1 BW results

| Model | Technical configuration | mIoU | F1 | Accuracy | Kappa | Params | FLOPs | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | `unet_BW` | 0.7086 | 0.8233 | 0.8937 | 0.8563 | 13.48M | 36.02G | 161.61 |
| DeepLab | `deeplabv3p_BW` | 0.7825 | 0.8752 | 0.9269 | 0.9013 | 26.83M | 29.11G | 120.18 |
| OCRNet | `ocrnet_BW` | 0.7283 | 0.8379 | 0.9002 | 0.8651 | 12.19M | 14.41G | 47.83 |
| SegFormer | `segformer_BW` | 0.7464 | 0.8502 | 0.9051 | 0.8720 | 27.73M | 15.73G | 102.48 |
| HighDAN | `highdan_BW` | 0.7269 | 0.8371 | 0.9098 | 0.8768 | 16.65M | 162.18G | 54.89 |
| UpderNet | `cxup_1b_BW` | 0.8025 | 0.8878 | 0.9328 | 0.9094 | 30.01M | 31.70G | 143.93 |
| MRSN | `cxup_4b2h_BW` | 0.8659 | 0.9269 | 0.9595 | 0.9455 | 116.82M | 102.67G | 49.46 |
| MRSFN | `cxup_4b_BW_PMRG_v2_lossV2` | 0.8694 | 0.9287 | 0.9658 | 0.9539 | 116.51M | 94.18G | 42.86 |

### 2.2 AB results

| Model | Technical configuration | mIoU | F1 | Accuracy | Kappa | Params | FLOPs | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | `unet_AB` | 0.7009 | 0.8173 | 0.8422 | 0.8099 | 13.55M | 10.19G | 57.99 |
| DeepLab | `deeplabv3p_AB` | 0.5589 | 0.7008 | 0.7501 | 0.6967 | 26.86M | 7.43G | 34.07 |
| OCRNet | `ocrnet_AB` | 0.5809 | 0.7176 | 0.7778 | 0.7306 | 12.26M | 3.90G | 36.25 |
| SegFormer | `segformer_AB` | 0.6140 | 0.7472 | 0.7891 | 0.7455 | 28.13M | 4.34G | 43.62 |
| HighDAN | `highdan_AB` | 0.6423 | 0.7714 | 0.8115 | 0.7725 | 16.73M | 40.84G | 33.26 |
| UpderNet | `cxup_1b_AB` | 0.6769 | 0.7988 | 0.8288 | 0.7939 | 30.21M | 8.12G | 50.45 |
| MRSN | `cxup_4b2h_AB` | 0.7657 | 0.8625 | 0.8760 | 0.8510 | 117.02M | 25.87G | 43.24 |
| MRSFN | `cxup_4b_AB_PMRG_v2_lossV2` | 0.7571 | 0.8576 | 0.8905 | 0.8685 | 116.70M | 23.75G | 29.69 |

### 2.3 Baseline interpretation

- In BW, MRSFN exceeds the legacy MRSN by `+0.0035` mIoU.
- In AB, MRSN exceeds MRSFN by `+0.0086` mIoU.
- The BW and AB rankings are therefore split-specific and should not be
  summarized as a universal ranking.
- MRSN is the legacy structure; MRSFN is the final proposed model.

## 3. Ablation study

The ablation study separates branch capacity, PMRG, and ML. The `1B` row is
also the model used as the UpderNet baseline, while the final row is MRSFN.

### 3.1 BW results

| Configuration | Technical configuration | mIoU | F1 | Accuracy | Kappa | Params | FLOPs | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B / UpderNet | `cxup_1b_BW` | 0.8025 | 0.8878 | 0.9328 | 0.9094 | 30.01M | 31.70G | 143.93 |
| 2B | `cxup_2b_BW` | 0.8342 | 0.9079 | 0.9470 | 0.9286 | 58.50M | 51.71G | 104.47 |
| 3B | `cxup_3b_BW` | 0.8496 | 0.9173 | 0.9537 | 0.9377 | 87.00M | 71.73G | 85.27 |
| 4B | `cxup_4b_BW` | 0.8659 | 0.9269 | 0.9598 | 0.9459 | 115.49M | 91.76G | 49.96 |
| 4B + PMRG | `cxup_4b_BW_PMRG_v2` | 0.8671 | 0.9277 | 0.9602 | 0.9465 | 116.51M | 94.18G | 41.42 |
| 4B + ML | `cxup_4b_BW_lossV2` | 0.8684 | 0.9281 | 0.9656 | 0.9537 | 115.49M | 91.76G | 51.61 |
| 4B + PMRG + ML / MRSFN | `cxup_4b_BW_PMRG_v2_lossV2` | 0.8694 | 0.9287 | 0.9658 | 0.9539 | 116.51M | 94.18G | 42.86 |

### 3.2 AB results

| Configuration | Technical configuration | mIoU | F1 | Accuracy | Kappa | Params | FLOPs | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B / UpderNet | `cxup_1b_AB` | 0.6769 | 0.7988 | 0.8288 | 0.7939 | 30.21M | 8.12G | 50.45 |
| 2B | `cxup_2b_AB` | 0.7164 | 0.8281 | 0.8513 | 0.8212 | 58.70M | 13.13G | 60.53 |
| 3B | `cxup_3b_AB` | 0.7310 | 0.8388 | 0.8580 | 0.8293 | 87.19M | 18.14G | 31.56 |
| 4B | `cxup_4b_AB` | 0.7634 | 0.8610 | 0.8758 | 0.8508 | 115.68M | 23.14G | 54.07 |
| 4B + PMRG | `cxup_4b_AB_PMRG_v2` | 0.7644 | 0.8620 | 0.8752 | 0.8503 | 116.70M | 23.75G | 42.92 |
| 4B + ML | `cxup_4b_AB_lossV2` | 0.7553 | 0.8562 | 0.8897 | 0.8675 | 115.68M | 23.14G | 32.00 |
| 4B + PMRG + ML / MRSFN | `cxup_4b_AB_PMRG_v2_lossV2` | 0.7571 | 0.8576 | 0.8905 | 0.8685 | 116.70M | 23.75G | 29.69 |

### 3.3 Ablation interpretation

- BW shows a monotonic improvement from 1B through 4B.
- In BW, PMRG and ML both improve the 4B result, with the complete MRSFN
  configuration giving the highest mIoU in this table.
- AB also shows a branch-capacity increase through 4B, but the ML and complete
  MRSFN configurations are lower in mIoU than the plain 4B result.
- These single-seed BW and AB results should be reported descriptively; the
  EXP-03 multi-seed results are required for repeatability claims.

## 4. EXP-02 Backbone and capacity comparison

This supplementary BW experiment compares the ConvNeXt-Tiny branch-count
series with two larger-backbone variants. The 1B--4B rows vary the number of
branches while using ConvNeXt-Tiny; the two EXP-02 rows keep the 1B branch
layout and replace the backbone with ConvNeXt-Small or ConvNeXt-Base. This is
an empirical capacity/backbone check rather than a parameter-matched causal
comparison.

### 4.1 BW results

| Configuration | Backbone | Technical configuration | mIoU | F1 | Accuracy | Kappa | Params | FLOPs | FPS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B / UpderNet | ConvNeXt-Tiny | `cxup_1b_BW` | 0.8025 | 0.8878 | 0.9328 | 0.9094 | 30.01M | 31.70G | 143.93 |
| 2B | ConvNeXt-Tiny | `cxup_2b_BW` | 0.8342 | 0.9079 | 0.9470 | 0.9286 | 58.50M | 51.71G | 104.47 |
| 3B | ConvNeXt-Tiny | `cxup_3b_BW` | 0.8496 | 0.9173 | 0.9537 | 0.9377 | 87.00M | 71.73G | 85.27 |
| 4B | ConvNeXt-Tiny | `cxup_4b_BW` | 0.8659 | 0.9269 | 0.9598 | 0.9459 | 115.49M | 91.76G | 49.96 |
| 1B + larger backbone | ConvNeXt-Small | `cxup_1b_BW_small_exp02` | 0.8203 | 0.8993 | 0.9410 | 0.9204 | 51.65M | 53.80G | 64.78 |
| 1B + larger backbone | ConvNeXt-Base | `cxup_1b_BW_base_exp02` | 0.8301 | 0.9054 | 0.9454 | 0.9264 | 90.05M | 86.07G | 45.12 |

## 5. EXP-03 Repeatability

The following values are calculated over seeds `1919810`, `1919811`, and
`1919812`. Seed `1919810` comes from the original `output_bw/` checkpoints;
seeds `1919811` and `1919812` come from the `output/exp03_*` checkpoints. Each
seed contributes the metrics of its best-validation checkpoint. Values are
reported as mean ± sample standard deviation and are limited to the BW
repeatability experiment.

| Configuration | mIoU (mean ± std) | F1 (mean ± std) | Accuracy (mean ± std) | Kappa (mean ± std) |
|---|---:|---:|---:|---:|
| 1B / UpderNet | 0.8011 ± 0.0013 | 0.8870 ± 0.0008 | 0.9324 ± 0.0004 | 0.9089 ± 0.0005 |
| 2B | 0.8337 ± 0.0006 | 0.9075 ± 0.0004 | 0.9468 ± 0.0004 | 0.9283 ± 0.0006 |
| 3B | 0.8507 ± 0.0010 | 0.9180 ± 0.0006 | 0.9538 ± 0.0001 | 0.9378 ± 0.0002 |
| 4B | 0.8653 ± 0.0006 | 0.9265 ± 0.0004 | 0.9596 ± 0.0003 | 0.9456 ± 0.0004 |
| 4B + PMRG | 0.8666 ± 0.0007 | 0.9274 ± 0.0004 | 0.9601 ± 0.0003 | 0.9462 ± 0.0005 |
| 4B + ML | 0.8691 ± 0.0006 | 0.9284 ± 0.0003 | 0.9657 ± 0.0002 | 0.9539 ± 0.0003 |
| 4B + PMRG + ML / MRSFN | 0.8698 ± 0.0005 | 0.9289 ± 0.0003 | 0.9661 ± 0.0003 | 0.9543 ± 0.0005 |

The repeatability table is reported separately from the single-seed BW and AB
tables above and should be used for stability claims.

## 6. Model naming

- `MRSN`: earlier/legacy structure, represented by `cxup_4b2h_BW` and
  `cxup_4b2h_AB` in the current result archives.
- `UpderNet`: the 1B model, represented by `cxup_1b_BW` and `cxup_1b_AB`.
- `MRSFN`: final proposed model (`4B + PMRG + ML`), represented by
  `cxup_4b_BW_PMRG_v2_lossV2` and `cxup_4b_AB_PMRG_v2_lossV2`.

