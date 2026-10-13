# Efficiency benchmark

All rows measured in one environment within one run. Synthetic inputs of the exact validation shapes; models in eval mode; `no_grad`; CUDA-synchronized timing.

- Paddle: 3.3.1 (CUDA compiled: True)
- Device: gpu
- GPU: unknown
- Precision: FP32
- FLOPs backend: paddle.flops
- Batch size: 1
- Warmup: 20 iters, timed: 100 iters x 3 repeats
- Date: 2026-10-13 18:30

## main_bw

| Model | Config | Params (M) | FLOPs (G) | FPS |
|---|---|---:|---:|---:|
| UNet | `unet_BW` | 13.48 | 36.02 | 267.57 ± 1.57 |
| DeepLabV3+ | `deeplabv3p_BW` | 26.83 | 29.11 | 97.43 ± 3.68 |
| OCRNet | `ocrnet_BW` | 12.19 | 14.41 | 24.30 ± 0.50 |
| SegFormer | `segformer_BW` | 27.73 | 15.73 | 82.39 ± 1.59 |
| HighDAN | `highdan_BW` | 16.65 | 162.18 | 30.53 ± 0.17 |
| CMX (MiT-B4) | `cmx_4b_BW` | 141.81 | 35.59 | 15.10 ± 0.09 |
| UPerNet (1B) | `cxup_1b_BW` | 30.01 | 31.70 | 164.04 ± 0.08 |
| MRSN (4B2H) | `MRSN_BW` | 116.82 | 102.67 | 48.31 ± 0.17 |
| MRSFN (full) | `cxup_4b_BW_PMRG_v2_lossV2` | 116.51 | 94.18 | 47.06 ± 0.08 |

## main_ab

| Model | Config | Params (M) | FLOPs (G) | FPS |
|---|---|---:|---:|---:|
| UNet | `unet_AB` | 13.55 | 10.19 | 324.16 ± 7.88 |
| DeepLabV3+ | `deeplabv3p_AB` | 26.86 | 7.43 | 98.12 ± 2.37 |
| OCRNet | `ocrnet_AB` | 12.26 | 3.90 | 24.35 ± 0.19 |
| SegFormer | `segformer_AB` | 28.13 | 4.34 | 82.00 ± 0.96 |
| HighDAN | `highdan_AB` | 16.73 | 40.84 | 29.67 ± 1.24 |
| CMX (MiT-B4) | `cmx_4b_AB` | 142.20 | 9.31 | 15.13 ± 0.14 |
| UPerNet (1B) | `cxup_1b_AB` | 30.21 | 8.12 | 179.40 ± 0.91 |
| MRSN (4B2H) | `MRSN_AB` | 117.02 | 25.87 | 54.24 ± 0.41 |
| MRSFN (full) | `cxup_4b_AB_PMRG_v2_lossV2` | 116.70 | 23.75 | 46.34 ± 0.52 |

## ablation_bw

| Model | Config | Params (M) | FLOPs (G) | FPS |
|---|---|---:|---:|---:|
| 1-branch (UPerNet ref.) | `cxup_1b_BW` | 30.01 | 31.70 | 164.04 ± 0.08 |
| 2-branch | `cxup_2b_BW` | 58.50 | 51.71 | 96.60 ± 0.15 |
| 3-branch | `cxup_3b_BW` | 87.00 | 71.73 | 66.89 ± 0.10 |
| 4-branchv0 | `cxup_4b_BW_RGB_NIR` | 115.49 | 91.74 | 51.55 ± 0.06 |
| 4-branch | `cxup_4b_BW` | 115.49 | 91.76 | 51.39 ± 0.13 |
| 4-branch + PMRG | `cxup_4b_BW_PMRG_city` | 116.51 | 94.18 | 46.92 ± 0.42 |
| 4-branch + ML | `cxup_4b_BW` | 115.49 | 91.76 | 51.39 ± 0.13 |
| MRSFN (4-branch + PMRG + ML) | `cxup_4b_BW_PMRG_v2_lossV2` | 116.51 | 94.18 | 47.06 ± 0.08 |

## backbone_bw

| Model | Config | Params (M) | FLOPs (G) | FPS |
|---|---|---:|---:|---:|
| UPerNet (1 branch) | `cxup_1b_BW` | 30.01 | 31.70 | 164.04 ± 0.08 |
| 1 branch | `cxup_1b_BW_small` | 51.65 | 53.80 | 102.28 ± 0.54 |
| 2 branches | `cxup_2b_BW` | 58.50 | 51.71 | 96.60 ± 0.15 |
| 3 branches | `cxup_3b_BW` | 87.00 | 71.73 | 66.89 ± 0.10 |
| 1 branch | `cxup_1b_BW_base` | 90.05 | 86.07 | 71.22 ± 0.16 |
| 2 branches | `cxup_2b_BW_small` | 101.77 | 95.91 | 56.51 ± 0.10 |
| 4-branchv0 | `cxup_4b_BW_RGB_NIR` | 115.49 | 91.74 | 51.55 ± 0.06 |
| 4-branch | `cxup_4b_BW` | 115.49 | 91.76 | 51.39 ± 0.13 |
| 1 branch | `cxup_1b_BW_large` | 199.28 | 177.61 | 39.76 ± 0.02 |

## city_bw

| Model | Config | Params (M) | FLOPs (G) | FPS |
|---|---|---:|---:|---:|
| UNet | `unet_BW_city` | 13.48 | 36.02 | 262.98 ± 0.20 |
| DeepLabV3+ | `deeplabv3p_BW_city` | 26.83 | 29.11 | 100.05 ± 1.21 |
| OCRNet | `ocrnet_BW_city` | 12.19 | 14.41 | 24.42 ± 0.36 |
| SegFormer | `segformer_BW_city` | 27.73 | 15.73 | 82.09 ± 1.20 |
| HighDAN | `highdan_BW_city` | 16.65 | 162.18 | 31.10 ± 0.08 |
| CMX (MiT-B4) | `cmx_4b_BW_city` | 141.81 | 35.59 | 15.52 ± 0.03 |
| UPerNet (1B) | `cxup_1b_BW_city` | 30.01 | 31.70 | 162.75 ± 0.67 |
| MRSN | `MRSN_BW_city` | 116.82 | 102.67 | 48.45 ± 0.02 |
| MRSFN | `cxup_4b_BW_PMRG_v2_lossV2_city` | 116.51 | 94.18 | 47.47 ± 0.12 |
