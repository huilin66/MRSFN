# MRSN / MRSFN 项目交接文档

> 文档版本：v2.0
> 更新时间：2026-08-29  
> 项目目录：`E:\repository\MRSN`  
> 目的：帮助后续工作快速理解项目、复现实验并继续处理论文实验。

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `handoff-update`
- Origin Date: `2026-08-29`
- Verification Status: `UNVERIFIED`（本文档记录实验计划与当前仓库状态；下方实验尚未全部执行）
- Version Label: `handoff_v2.0`
- Upstream Dependencies: `E:\repository\academic research\mrsn\ieee_version\experiment_handoff.md`、当前仓库代码与 `.env`

## 1. 项目背景

本项目是基于 PaddlePaddle/PaddleSeg 的多模态遥感语义分割实现，面向 C2Seg 跨城市语义分割数据。项目同时保留历史 MRSN 基线，并继续发展当前提交版本 MRSFN：

- **MRSN（Multimodal Remote Sensing Network）**：发表于 WHISPERS 2023 的原始多模态遥感分割模型。
- **MRSFN**：当前提交模型/扩展工作的名称，在多分支融合基础上加入 Pixel-wise Modality Reliability Gate（PMRG）和混合损失。
- **MBFM**：历史交接材料中使用的旧名称。后续公共说明中将当前模型写作 MRSFN，但不能把所有 `MRSN` 技术标识机械替换为 `MRSFN`；历史 MRSN 模型、论文和基线仍应保持可追溯。

输入数据包含：

- MSI：4 个通道；
- SAR：2 个通道；
- HSI：116 个通道；
- 标签：14 个类别，类别编号为 0–13，Background 是真实类别而非 ignore 类别。

默认实验以 256×256 patch 为基本样本。C2Seg-BW 默认随机 patch 划分在同一完整场景的 patch 之间可能产生地理泄漏，因此当前项目新增了北京/武汉城市级互斥划分实验，用于验证模型在地理独立场景上的表现。

官方测试标签不在仓库中，当前报告的指标主要是内部验证集指标。

## 2. 仓库结构

```text
README.md                         项目说明、安装、训练和验证命令
docs/REPRODUCIBILITY.md           可复现实验说明和已报告指标
PaddleCD/                         PaddleSeg 定制训练框架
PaddleCD/train.py                训练入口
PaddleCD/val.py                  验证入口
PaddleCD/predict.py              patch/目录预测入口
PaddleCD/export.py               模型导出入口
PaddleCD/c2seg_config/           C2Seg-AB/BW、模型和城市划分配置
PaddleCD/paddleseg/datasets/     数据集实现，包括 RSCD、RS_MD2B、RS_MD3B
PaddleCD/paddleseg/models/       模型实现，包括 cx_uper.py 和大量 PaddleSeg 模型
PaddleCD/paddleseg/transforms/   Compose、Normalize2、旋转/翻转等变换
tools/                            数据转换、城市切分、全场景推理和分析脚本
data/                             本地数据和城市级 patch（被 gitignore 忽略）
output/                           本地训练 checkpoint（被 gitignore 忽略）
log/                              训练/验证日志（被 gitignore 忽略）
ana/、vis/                        分析结果和可视化（被 gitignore 忽略）
upload/                           发布模型及 train.log/val.log（被 gitignore 忽略）
pic/                              网络结构图
main.ipynb、aug.ipynb             Notebook 实验流程
```

仓库内没有发现自动化测试目录或 pytest 配置。

## 3. 核心数据与模型链路

### 3.1 配置加载

配置入口是 `PaddleCD/paddleseg/cvlibs/config.py`。它会：

1. 从当前目录及父目录加载 `.env`；
2. 展开 YAML 中的 `${VARIABLE}` 环境变量；
3. 处理 `_base_` 配置继承；
4. 通过 PaddleSeg 的 manager 注册表实例化 dataset、model、transform、loss 和 optimizer。

基础数据配置位于 `PaddleCD/c2seg_config/C2Seg_BW.yml` 和 `C2Seg_AB.yml`。模型配置通常只覆盖 `model` 和 `loss`，例如 `cxup_4b_BW_PMRG_v2_lossV2.yml`。

### 3.2 数据集

`RS_MD3B` 定义在 `PaddleCD/paddleseg/datasets/rscd.py` 中，并由 `datasets/__init__.py` 导入后注册。文件列表通常形如：

```text
msi/sample.tiff sar/sample.tiff lbl/sample.tiff
```

读取时会将：

- `msi/...` 映射到 `msisar/...`，得到 4 MSI + 2 SAR 共 6 通道；
- `sar/...` 映射到 `hsi/...`，得到 116 通道 HSI；
- `lbl/...` 读取为分割标签。

`Normalize2` 对两路输入分别使用 `mean1/std1` 和 `mean2/std2` 做归一化。

### 3.3 模型

核心实现位于 `PaddleCD/paddleseg/models/cx_uper.py`：

- `CX_Uper`：单 backbone 堆叠输入基线；
- `CX_Uper_2B`、`CX_Uper_3B`、`CX_Uper_4B`：逐步增加模态/分支数量；
- `CX_Uper_4B2H`：MRSN 配置使用的四分支/双 head 变体；
- `CX_Uper_4B_PMRG`：PMRG 初版；
- `CX_Uper_4B_PMRG_V2`：当前 MRSFN/PMRG 主实验模型；这是代码技术标识，不因公共模型命名调整而重命名。

4B 模型使用四个 ConvNeXt backbone，输入通道分别为 3、3、2、116；低层特征融合后进入 UPerHead。PMRG V2 在多个 backbone stage 上预测每个模态的像素级可靠性门控，并通过 baseline-preserving 的方式调节模态特征。

### 3.4 损失与训练

常用训练配置：

- batch size：16；
- iterations：40,000；
- optimizer：AdamW；
- 初始学习率：0.0002；
- StepDecay：每 5,000 iterations 衰减为 0.5；
- seed：1,919,810；
- 常用损失：`Poly1Loss_Smooth + 0.5 * DiceLoss`。

## 4. 常用运行方式

从项目根目录运行：

```bash
python PaddleCD/train.py \
  --config PaddleCD/c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml \
  --save_dir output/cxup_4b_BW_PMRG_v2_lossV2 \
  --do_eval
```

```bash
python PaddleCD/val.py \
  --config PaddleCD/c2seg_config/cxup_4b_BW_PMRG_v2_lossV2.yml \
  --model_path output/cxup_4b_BW_PMRG_v2_lossV2/best_model/model.pdparams \
  --batch_size 1
```

城市级数据生成入口：

```bash
python tools/build_city_split_dataset.py --split train_B_val_W
```

完整场景推理入口是 `tools/infer_full_scene.py`；该脚本已经包含 HSI 与 MSI/SAR 不同空间分辨率时的坐标映射和 resize 逻辑。

## 5. 当前仓库状态

截至 2026-08-29：

- 当前分支为 `re`，HEAD 为 `ff014fc`，工作区存在未提交修改：`.gitignore`、`PaddleCD/c2seg_config/C2Seg_BW_city.yml`、`README.md` 和 `handoff.md`；这些修改继续保留，后续操作前需先核对归属。
- `output/` 中已有多种 baseline、MRSN、4B、PMRG 和损失变体的 checkpoint。
- 当前 MRSFN/PMRG V2 验证日志记录的代表性结果为：mIoU `0.8694`、F1 `0.9287`、ACC `0.9658`、Kappa `0.9539`、参数量约 `116.51M`、FLOPs 约 `94.18G`。原始配置、日志和 checkpoint 中的技术名称仍可能使用 `MBFM` 或 `cxup_*`，应按上下文解释。
- 本地城市级数据目录 `data/C2Seg_BW_city_train_B_val_W` 包含北京训练 1,855 个 patch、武汉验证 850 个 patch。
- 远程城市级数据集 `C2SEG_TRAIN_B_VAL_W` 已重新生成并校验，当前 `.env` 的 `C2SEG_BW_CITY_ROOT` 已指向该目录。
- 数据、输出、日志和分析目录均被 `.gitignore` 忽略，不能假设其他环境会自动拥有这些本地文件。

## 6. 已知问题与风险

### 6.1 城市级数据中的 HSI patch 与生成脚本

此前检查本地城市数据发现，2,705 个样本中有 2,381 个 HSI TIFF 至少有一个零空间维度，例如 `(0, 256, 116)`。这些旧文件虽然出现在 `train.txt`/`val.txt` 中，但无法与 256×256 的 MSI/SAR patch 对齐，不应直接用于训练。

生成脚本现已改为：若 `.env` 设置了 `C2SEG_CITY_ROOT`，则根据 `--split` 自动输出到 `C2SEG_CITY_ROOT/C2SEG_<SPLIT>`；同时按 MSI/SAR 场景坐标比例读取 HSI，并 resize 到目标 patch 尺寸。远程 `C2SEG_TRAIN_B_VAL_W` 数据已按此逻辑重新生成，并逐项验证：

```text
msisar: (256, 256, 6)
hsi:    (256, 256, 116)
label:  (256, 256)
```

### 6.2 环境变量与文档不完全一致

城市生成脚本使用 `C2SEG_CITY_ROOT` 作为输出父目录，并自动创建 `C2SEG_TRAIN_B_VAL_W` 或 `C2SEG_TRAIN_W_VAL_B` 子目录。当前训练配置 `C2Seg_BW_city.yml` 使用的 `C2SEG_BW_CITY_ROOT` 已指向 `C2SEG_TRAIN_B_VAL_W`；切换划分方向时需要同步更新该变量。

另外，README/可复现文档中对 `C2SEG_BW_ROOT` 是否包含 `train` 子目录的示例，与当前 YAML 中自动追加 `/train` 的写法不完全一致。应以当前 YAML 和实际数据目录结构为准。

### 6.3 当前执行环境不能直接训练

当前 shell 使用 Python 3.14，且未安装 PaddlePaddle；配置中的默认数据路径还是远程 UNC 路径，当前环境也无法访问该远程路径。因此复现实验前需要准备兼容的 Python/Paddle GPU 环境和可访问的数据根目录。

### 6.4 其他代码级注意事项

- `train.py` 的默认 config 是旧的 Linux 绝对路径，运行时应始终显式传入 `--config`。
- `export.py` 使用单输入 3 通道 `InputSpec`，而多模态模型实际接收 6 通道和 116 通道两路输入；导出功能需要针对自定义模型重新验证。
- 当前代码中不支持的 backbone 名称会回退到 ConvNeXt-Base；执行 EXP-02 前应先显式增加 `convnext_large` 路由。仅做 4B-Large 对比只能说明规模敏感性，不能作为容量控制实验。
- PMRG 扰动实验应在 stream splitting 之后做 branch-level masking；直接遮蔽原始 MSI 应标记为 sensor-level MSI missing，因为 RGB 与 NIRGB 存在通道重叠。
- 当前没有自动化测试，修改数据读取、模型输入或配置注册逻辑后，应至少执行 dataset sample shape 检查、model dry-run 和一次小规模验证。

## 7. 后续工作内容

优先顺序建议：

1. 准备 Paddle GPU 运行环境，使用 `exp_add.sh` 按同一城市划分训练全部 15 个 BW 模型/变体；
2. 使用全部城市划分 checkpoint 完成下方 EXP-01 的完整场景/地理独立评估；
3. 完成下方 EXP-02 的容量控制分支对比；
4. 完成下方 EXP-03 的重复运行与不确定性统计；
5. 完成下方 EXP-04 的 PMRG 门控可视化和模态扰动实验；
6. 完成下方 EXP-05 的 CMX-adapted 两流基线；
7. 汇总 checkpoint、指标、推理参数、数据来源和实验 provenance，形成论文可用表格；
8. 最后再执行 REPO-01 的公共文档命名对齐；该项当前仅作为交接计划，不在本次更新中改动公共代码/文档。

### 源实验交接内容（原文更新版）

以下内容来自用户指定的实验交接文档。指定路径 `E:\repository\academic research\mrsn\ieee\_version\experiment\_handoff.md` 不存在；实际找到的对应文件为 `E:\repository\academic research\mrsn\ieee_version\experiment_handoff.md`。以下按该实际文件的最新内容保留，并已纳入当前项目状态。

**Status:** Planned  
**Purpose:** Address reviewer concerns R2-1 (geographic independence), R2-2 (branch/capacity confounding), repeatability of component claims, and modern multimodal baseline coverage (R2-6).

#### EXP-01 — Full-scene / geographic-independent evaluation

- **Protocol:** Run tiled inference on the complete original scene, align the prediction with the scene-level GT, and compute global and per-scene mIoU, macro-F1, OA/ACC, and Kappa from the confusion matrix.
- **Required record:** Scene name, GT/prediction paths, tile size and stride, and whether any pixels from this scene were used in training, validation, or model selection.
- **Caution:** Call it *geographically independent* only when the held-out scene provenance is verifiable; otherwise report it as a full-scene evaluation and state the limitation.
- **Deliverables:** Prediction/GT files, metric table, and a short provenance note.

#### EXP-02 — Capacity-controlled branch comparison

- **Goal:** Separate the effect of modality-specific branches from model capacity.
- **Protocol:** Under the same BW split, loss, crop, training budget, and seeds, compare the current dual-view MSI Tiny model (without PMRG/ML) with a single-branch stacked-input model using ConvNeXt-Base or width-adjusted capacity matched as closely as possible to the four-stream model.
- **Record:** Actual trainable parameters, FLOPs, inference speed, mIoU, macro-F1, OA/ACC, and Kappa; use three seeds if feasible.
- **Implementation note:** In the current code, unsupported backbone names fall through to ConvNeXt-Base. Add an explicit `convnext_large` route before using Large; a 4B-Large comparison alone is a scale-sensitivity test, not a capacity control.
- **Deliverables:** Configs/checkpoints, capacity and accuracy table, and seed summary.

#### EXP-03 — Repeated runs and uncertainty

- **Goal:** Quantify run-to-run variation and support the PMRG/ML component claims.
- **Protocol:** Repeat the dual-view MSI reference, dual-view MSI+PMRG, dual-view MSI+ML, and full MRSFN on the same BW split and training protocol with at least three fixed seeds (preferably five).
- **Record:** Per-run mIoU, macro-F1, OA/ACC, and Kappa; report mean $\pm$ standard deviation and list all seeds. Params/FLOPs need only be reported once per configuration.
- **Deliverables:** Per-run results, mean $\pm$ std table, seed list, and checkpoint/config paths.

#### EXP-04 — PMRG evidence under modality perturbation

- **Goal:** Verify that PMRG produces interpretable stream-dependent modulation and assess its behavior when an input stream is degraded.
- **Gate visualization:** In evaluation mode, save the three PMRG gate maps at $1/4$, $1/8$, and $1/16$ resolution, upsample them for visualization, and record mean gate weights. The code order is `NIRGB | RGB | SAR | HSI`; gate weights are learned modulation weights, not calibrated reliability probabilities.
- **Perturbation protocol:** Reuse EXP-03 checkpoints on the same BW validation set. Compare the dual-view MSI reference and dual-view MSI+PMRG under clean input, mean/zero-masked RGB, NIRGB, SAR, and HSI streams, plus one fixed-level Gaussian-noise condition.
- **Record:** mIoU, macro-F1, OA/ACC, Kappa, performance drop from clean input, and mean gate changes for the perturbed and unperturbed streams. Apply branch-level masking after stream splitting; masking raw MSI should be labeled sensor-level MSI missing because RGB and NIRGB overlap.
- **Deliverables:** Gate-map figures, perturbation metric table, clean-to-corrupted performance drops, and checkpoint/config paths.

#### EXP-05 — CMX-adapted two-stream baseline

- **Goal:** Add a modern cross-modal fusion baseline and compare it fairly with the existing 2B model.
- **Input and protocol:** Use exactly the same two input groups as 2B: `MSI+SAR | HSI`. Keep the split, preprocessing, crop size, optimizer, training budget, and seeds identical. Evaluate on BW first.
- **Reporting:** Put the CMX result in the final main results summary table, together with the existing 2B and complete-model results. Report mIoU, macro-F1, OA/ACC, Kappa, Params, FLOPs, and FPS.
- **Method note:** The result is a CMX-adapted C2Seg two-stream baseline, not an exact reproduction of the official RGB-X setting. Record the changed input stems, preprocessing, initialization, and any implementation differences.
- **Deliverables:** Config/code, checkpoint, final-summary-table row, and adaptation note.

#### REPO-01 — Repository documentation and naming alignment

- **Scope:** This is a documentation handoff only. Do not modify the code repository in the current step. When executed later, update public documentation only; do not rename source files or technical configuration identifiers solely for branding.
- **Current name:** Use `MRSFN` for the current submitted model and extension work (the multi-branch, PMRG, and mixed-loss configuration). Treat `MBFM` as a legacy name that should not appear as the current model name in public documentation.
- **Prior work:** Keep `MRSN` for the authors' earlier WHISPERS 2023 model, its baseline results, historical experiments, and citation. Do not mechanically replace every occurrence of `MRSN` with `MRSFN`.
- **Repository relationship:** Explain that MRSFN continues the authors' earlier MRSN work. Link the original MRSN repository as `https://github.com/huilin16/MRSN` and the current repository as `https://github.com/huilin66/MRSFN`.
- **README.md:** Update the title, current-model description, clone URL, and directory command to MRSFN; retain the original MRSN citation and clearly distinguish the prior model from the current extension.
- **docs/REPRODUCIBILITY.md:** Change the current-project scope to `MRSN/MRSFN` and rename only the final current-model row from `MBFM (4-branch + PMRG + ML)` to `MRSFN (4-branch + PMRG + ML)`. Keep MRSN baseline rows and historical descriptions unchanged.
- **Technical and historical names to preserve:** Keep `MRSN_BW.yml`, `MRSN_AB.yml`, `MRSN_BW_city.yml`, `CX_Uper*`, `cxup_*`, historical challenge/paper materials, and `mrsn` identifiers when they refer to the prior MRSN baseline. Change a `mrsn` label only when it actually denotes the current final model.
- **Verification after execution:** Search public repository documentation for residual current-model uses of `MBFM`, verify both repository links and clone commands, and confirm that the original MRSN model remains separately traceable. Do not push changes as part of this task.

## 8. 交接约定

- 不要把内部验证集指标表述为官方测试集结果。
- 所有实验记录应同时保存 config、checkpoint、数据划分、tile/crop/stride、seed 和运行环境信息。
- 涉及城市级实验时，明确记录训练城市、验证城市，以及场景是否参与过模型选择。
- 每个实验都要明确区分 `planned`、`executed`、`verified`；没有真实运行记录时不能填写结果或声称已完成。
- EXP-03 及后续组件实验应保留固定 seed、逐次结果和汇总统计；均值 ± 标准差不能替代逐次结果。
- 当前模型公共名称使用 MRSFN，历史模型使用 MRSN，MBFM 仅作为旧名称解释；保留技术配置名和历史标识以保证复现。
- REPO-01 执行前不得批量重命名 `MRSN`、`CX_Uper*`、`cxup_*` 或配置文件；不在本次 handoff 更新中推送提交。
- 修改前先检查 Git 工作区和本地未跟踪产物，避免覆盖已有实验结果。
