# MRSN / MBFM 项目交接文档

> 更新时间：2026-08-29  
> 项目目录：`E:\repository\MRSN`  
> 目的：帮助后续工作快速理解项目、复现实验并继续处理论文实验。

## 1. 项目背景

本项目是基于 PaddlePaddle/PaddleSeg 的多模态遥感语义分割实现，面向 C2Seg 跨城市语义分割数据。项目包含两条主要研究线：

- **MRSN（Multimodal Remote Sensing Network）**：发表于 WHISPERS 2023 的原始多模态遥感分割模型。
- **MBFM（Multi-Branch Fusion Model）**：在多分支融合基础上加入 Pixel-wise Modality Reliability Gate（PMRG）和混合损失的扩展模型。

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
- `CX_Uper_4B_PMRG_V2`：当前 MBFM/PMRG 主实验模型。

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

- 当前分支为 `re`，HEAD 为 `ff014fc`，工作区干净；该提交主要新增城市级地理互斥切分配置和生成脚本。
- `output/` 中已有多种 baseline、MRSN、4B、PMRG 和损失变体的 checkpoint。
- 当前 MBFM/PMRG V2 验证日志记录的代表性结果为：mIoU `0.8694`、F1 `0.9287`、ACC `0.9658`、Kappa `0.9539`、参数量约 `116.51M`、FLOPs 约 `94.18G`。
- 本地城市级数据目录 `data/C2Seg_BW_city_train_B_val_W` 包含北京训练 1,855 个 patch、武汉验证 850 个 patch。
- 数据、输出、日志和分析目录均被 `.gitignore` 忽略，不能假设其他环境会自动拥有这些本地文件。

## 6. 已知问题与风险

### 6.1 城市级数据中的 HSI patch 与生成脚本

此前检查本地城市数据发现，2,705 个样本中有 2,381 个 HSI TIFF 至少有一个零空间维度，例如 `(0, 256, 116)`。这些文件虽然出现在 `train.txt`/`val.txt` 中，但无法与 256×256 的 MSI/SAR patch 对齐，当前不应直接用于训练。

生成脚本现已改为：若 `.env` 设置了 `C2SEG_CITY_ROOT`，则根据 `--split` 自动输出到 `C2SEG_CITY_ROOT/C2SEG_<SPLIT>`；同时按 MSI/SAR 场景坐标比例读取 HSI，并 resize 到目标 patch 尺寸。需要重新生成城市数据，并逐项验证：

```text
msisar: (256, 256, 6)
hsi:    (256, 256, 116)
label:  (256, 256)
```

### 6.2 环境变量与文档不完全一致

城市生成脚本使用 `C2SEG_CITY_ROOT` 作为输出父目录，并自动创建 `C2SEG_TRAIN_B_VAL_W` 或 `C2SEG_TRAIN_W_VAL_B` 子目录。训练配置 `C2Seg_BW_city.yml` 仍要求 `C2SEG_BW_CITY_ROOT` 指向最终生成的子目录，因此运行城市训练前需要显式设置该变量。

另外，README/可复现文档中对 `C2SEG_BW_ROOT` 是否包含 `train` 子目录的示例，与当前 YAML 中自动追加 `/train` 的写法不完全一致。应以当前 YAML 和实际数据目录结构为准。

### 6.3 当前执行环境不能直接训练

当前 shell 使用 Python 3.14，且未安装 PaddlePaddle；配置中的默认数据路径还是远程 UNC 路径，当前环境也无法访问该远程路径。因此复现实验前需要准备兼容的 Python/Paddle GPU 环境和可访问的数据根目录。

### 6.4 其他代码级注意事项

- `train.py` 的默认 config 是旧的 Linux 绝对路径，运行时应始终显式传入 `--config`。
- `export.py` 使用单输入 3 通道 `InputSpec`，而多模态模型实际接收 6 通道和 116 通道两路输入；导出功能需要针对自定义模型重新验证。
- 当前没有自动化测试，修改数据读取、模型输入或配置注册逻辑后，应至少执行 dataset sample shape 检查、model dry-run 和一次小规模验证。

## 7. 后续工作内容

优先顺序建议：

1. 修复并重新生成城市级数据，确认所有 HSI patch 空间尺寸有效；
2. 准备 Paddle GPU 运行环境，验证基础配置和现有 checkpoint；
3. 完成下方 EXP-01 的完整场景/地理独立评估；
4. 完成下方 EXP-02 的容量控制分支对比；
5. 汇总 checkpoint、指标、推理参数、数据来源和实验 provenance，形成论文可用表格。

### 原始实验交接内容

以下内容来自用户指定的实验交接文档。指定路径 `E:\repository\academic research\mrsn\ieee\_version\experiment\_handoff.md` 不存在；实际找到的对应文件为 `E:\repository\academic research\mrsn\ieee_version\experiment_handoff.md`。原文保留如下：

# Experiment Handoff

**Status:** Planned  
**Purpose:** Address reviewer concerns R2-1 (geographic independence) and R2-2 (branch/capacity confounding).

## EXP-01 — Full-scene / geographic-independent evaluation

- **Protocol:** Run tiled inference on the complete original scene, align the prediction with the scene-level GT, and compute global and per-scene mIoU, macro-F1, OA/ACC, and Kappa from the confusion matrix.
- **Required record:** Scene name, GT/prediction paths, tile size and stride, and whether any pixels from this scene were used in training, validation, or model selection.
- **Caution:** Call it *geographically independent* only when the held-out scene provenance is verifiable; otherwise report it as a full-scene evaluation and state the limitation.
- **Deliverables:** Prediction/GT files, metric table, and a short provenance note.

## EXP-02 — Capacity-controlled branch comparison

- **Goal:** Separate the effect of modality-specific branches from model capacity.
- **Protocol:** Under the same BW split, loss, crop, training budget, and seeds, compare the current 4B-Tiny model (without PMRG/ML) with a single-branch stacked-input model using ConvNeXt-Base or width-adjusted capacity matched as closely as possible to 4B-Tiny.
- **Record:** Actual trainable parameters, FLOPs, inference speed, mIoU, macro-F1, OA/ACC, and Kappa; use three seeds if feasible.
- **Implementation note:** In the current code, unsupported backbone names fall through to ConvNeXt-Base. Add an explicit `convnext_large` route before using Large; a 4B-Large comparison alone is a scale-sensitivity test, not a capacity control.
- **Deliverables:** Configs/checkpoints, capacity and accuracy table, and seed summary.

## 8. 交接约定

- 不要把内部验证集指标表述为官方测试集结果。
- 所有实验记录应同时保存 config、checkpoint、数据划分、tile/crop/stride、seed 和运行环境信息。
- 涉及城市级实验时，明确记录训练城市、验证城市，以及场景是否参与过模型选择。
- 修改前先检查 Git 工作区和本地未跟踪产物，避免覆盖已有实验结果。
