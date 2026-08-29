# MRSN / MRSFN 项目交接文档

> 文档版本：v2.8
> 更新时间：2026-08-29  
> 项目目录：`E:\repository\MRSN`  
> 目的：帮助后续工作快速理解项目、复现实验并继续处理论文实验。

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Origin Mode: `handoff-update`
- Origin Date: `2026-08-29`
- Verification Status: `UNVERIFIED`（本文档记录实验计划与当前仓库状态；下方实验尚未全部执行）
- Version Label: `handoff_v2.8`
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

EXP-02 基于现有普通 BW 实验补充 1B 模型的 backbone size：用
ConvNeXt-Small 和 ConvNeXt-Base 替换现有 1B 的 ConvNeXt-Tiny，使其参数量
分别接近现有 2B-Tiny 和 3B-Tiny，并为 4B-Tiny 提供更接近的标准 backbone
参照。该实验不使用 EXP-01 的城市划分。

### 3.4 损失与训练

常用训练配置：

- batch size：16；
- iterations：40,000；
- optimizer：AdamW；
- 初始学习率：0.0002；
- StepDecay：每 5,000 iterations 衰减为 0.5；
- seed：1,919,810；
- 普通 BW 基线损失：`CrossEntropyLoss_Smooth + 0.5 * DiceLoss`；EXP-03 的 loss 变体使用
  `Poly1Loss_Smooth + 0.5 * DiceLoss`。

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

实验脚本入口：

```bash
bash exp_add.sh
bash scripts/exp01_city_all_models.sh
bash scripts/exp02_capacity_control.sh
bash scripts/exp03_repeatability.sh
bash scripts/exp04_pmrg_evidence.sh
```

`exp_add.sh` 只负责逐行调度已经准备好的实验；每个实验的训练、推理和参数
集中在 `scripts/` 下对应的独立脚本中。实验脚本之间不互相调用，也不以
`exp_add.sh` 的执行顺序作为前置条件；需要复用时只直接读取仓库中已经存在的
配置、模型定义或 checkpoint。

## 5. 当前仓库状态

截至 2026-08-29：

- 当前分支为 `re`，HEAD 为 `ff014fc`，工作区存在未提交修改和新增实验文件，包括 `.gitignore`、`PaddleCD/c2seg_config/C2Seg_BW_city.yml`、`PaddleCD/paddleseg/models/cx_uper.py`、`README.md`、`handoff.md`、`exp_add.sh`、`scripts/`、`tools/summarize_exp03.py` 以及 EXP-02/EXP-03 配置；这些修改继续保留，后续操作前需先核对归属。
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
- `CX_Uper` 现在已显式支持 `convnext_base` 和 `convnext_large`，同时保留未知名称回退到 ConvNeXt-Base 的兼容行为。仅做 4B-Large 对比只能说明规模敏感性，不能作为容量控制实验。
- PMRG 扰动实验应在 stream splitting 之后做 branch-level masking；直接遮蔽原始 MSI 应标记为 sensor-level MSI missing，因为 RGB 与 NIRGB 存在通道重叠。
- 当前没有自动化测试，修改数据读取、模型输入或配置注册逻辑后，应至少执行 dataset sample shape 检查、model dry-run 和一次小规模验证。

## 7. 后续工作内容

优先顺序建议：

1. 准备 Paddle GPU 运行环境，使用 `exp_add.sh` 按同一城市划分训练全部 15 个 BW 模型/变体；
2. 使用全部城市划分 checkpoint 完成下方 EXP-01 的完整场景/地理独立评估；
3. 按现有普通 BW 实验协议完成下方 EXP-02 的 1B-Small 和 1B-Base 补充实验；
4. 完成下方 EXP-03 的 1B--4B 分支数量趋势及 4B 组件增益的重复运行与不确定性统计；
5. 实现并完成下方 EXP-04 的 PMRG 门控可视化和模态缺失/噪声证据实验；
6. 完成下方 EXP-05 的 CMX-adapted 两流基线；
7. 汇总 checkpoint、指标、推理参数、数据来源和实验 provenance，形成论文可用表格；
8. 最后再执行 REPO-01 的公共文档命名对齐；该项当前仅作为交接计划，不在本次更新中改动公共代码/文档。

### EXP-02 当前准备状态

- `cxup_1b_BW_small.yml`：保留 1B 单 backbone 结构，将 backbone 替换为 ConvNeXt-Small；
- `cxup_1b_BW_base.yml`：保留 1B 单 backbone 结构，将 backbone 替换为 ConvNeXt-Base；
- 两组配置继承普通 `C2Seg_BW.yml`，不使用 EXP-01 的 `C2Seg_BW_city.yml` 或城市划分数据；
- 现有普通 BW 日志中的参考参数量为：1B-Tiny `30.01M`、2B-Tiny `58.50M`、3B-Tiny `87.00M`、4B-Tiny `115.49M`；在 `mrsn` 环境中实测新增配置为：1B-Small `51.65M`（可训练 `51.64M`），1B-Base `90.05M`（可训练 `90.04M`）；新配置的 FLOPs/FPS 仍以运行日志为准；
- 上述参数统计使用 Paddle 3.3.1 在 CPU 上构建模型，并临时关闭预训练权重加载；该操作不改变参数数量，正式训练仍按配置的默认初始化策略执行；
- `scripts/exp02_capacity_control.sh` 使用现有实验的 seed `1919810`，依次运行两个补充配置；
- `exp_add.sh` 仅作为总调度器，每个实验占一行命令；EXP-01 至 EXP-04 均可从 `scripts/` 独立执行；
- 该实验目前是 `prepared`，尚未在兼容的 Paddle GPU 环境中执行。

### EXP-03 当前准备状态

- EXP-03 使用普通 BW 协议，不使用城市级划分；固定比较以下 7 个条件：
  `cxup_1b_BW.yml`、`cxup_2b_BW.yml`、`cxup_3b_BW.yml`、`cxup_4b_BW.yml`、
  `cxup_4b_BW_PMRG.yml`、`cxup_4b_BW_loss.yml`、
  `cxup_4b_BW_PMRG_v2_lossV2.yml`。
- 分支稳定性链为 `1B -> 2B -> 3B -> 4B`，用于检验增加分支后性能提升是否稳定；
  组件稳定性链以 `4B` 为共同 baseline，分别检验 `PMRG`、`Loss` 以及
  `PMRG + Loss` 的增益是否稳定。`cxup_4b_BW_PMRG_ML.yml` 与完整配置内容相同，
  不重复作为独立条件运行。
- `scripts/exp03_repeatability.sh` 为上述 7 个条件分别使用 seed
  `1919810`、`1919811`、`1919812`，共 21 次运行；每次输出到
  `output/exp03_<config>_seed<seed>/`，日志单独保存到
  `log/exp03/<config>_seed<seed>/`，并开启 `--do_eval`。
- 全部训练结束后运行 `python tools/summarize_exp03.py`；工具从每个日志中选取最高验证
  mIoU 的评估点，生成 `ana/exp03/per_run.csv`、`condition_summary.csv`、
  `matched_deltas.csv` 和 `stability_summary.csv`。
- 结果记录须包含每次运行的 mIoU、macro-F1/F1、OA/ACC、Kappa，以及 Params、FLOPs、FPS；
  汇总报告均值 ± 标准差和逐 seed 结果。稳定性判断还需记录同 seed 的
  `2B-1B`、`3B-2B`、`4B-3B`、`4B-1B`、`4B+PMRG-4B`、
  `4B+Loss-4B`、`4B+PMRG+Loss-4B`，以及在已改变另一组件时的
  `Full-PMRG`、`Full-Loss` 差值，并检查增益方向是否在各 seed 上保持一致。
- 当前状态为 `prepared`，脚本和配置已就绪，但 21 次训练尚未执行；在没有逐次运行记录前，
  不能声称分支、PMRG 或 loss 的提升已经得到稳定性证明。

### EXP-04 当前设计状态

- EXP-04 收敛为独立的 PMRG 证据实验，分为两部分：`Gate weight` 可视化，以及
  缺失/噪声 stream 下的性能与 gate 响应测试；不重新训练模型，不调用 EXP-03，
  也不依赖 `exp_add.sh` 的执行顺序。
- 当前直接使用仓库中已经存在的同 seed checkpoint：
  `cxup_4b_BW.yml`（`CX_Uper_4B`）和 `cxup_4b_BW_PMRG.yml`
  （`CX_Uper_4B_PMRG_V2`），seed metadata 为现有实验的 `1919810`。路径固定为
  `output/cxup_4b_BW/best_model/model.pdparams` 和
  `output/cxup_4b_BW_PMRG/best_model/model.pdparams`；不能自动改用
  `output/exp03_*` 或其他简称/完整 loss checkpoint。
- 当前已有 checkpoint 对应一个 seed，因此本次独立评估为
  `2 个模型 × 1 个 seed × 6 个条件 = 12 次验证`。若以后有额外的已有 checkpoint
  对，需要通过命令行显式传入对应路径逐次运行；EXP-04 不自动发现或生成 EXP-03
  checkpoint，也不把其他实验的运行结果当作本实验的重复样本。
- Gate 仅在 PMRG 模型推理时缓存/保存，不改变训练过程和默认输出接口。保存三个尺度
  `1/4`、`1/8`、`1/16` 的 gate，顺序固定为 `NIRGB | RGB | SAR | HSI`，并上采样到
  输入图像大小。可视化样本需固定记录原始 RGB、GT、clean prediction、四个 gate map
  和误差区域叠加图；原始 RGB 应从归一化前数据读取，或使用配置的均值/标准差反归一化。
- Gate 统计至少包括验证集平均值、标准差、相对于均匀值 `0.25` 的偏移，并尽可能按
  类别或可验证的场景分组。Gate weight 表示特征调制行为，不得称为校准后的可靠性概率。
- 最小扰动版本包含 6 个条件：`Clean`、`Missing-RGB`、`Missing-NIRGB`、
  `Missing-SAR`、`Missing-HSI` 和一个固定噪声条件。四个 missing 条件都在模型分支
  拆分之后执行；在 `Normalize2` 之后置零表示用该分支各通道的归一化均值替换，若在
  归一化之前实现则必须使用配置中的 `mean1/mean2`，不能使用原始值 `0`。
- `Missing-RGB` 与 `Missing-NIRGB` 必须标记为 `branch-level view missing`，不是物理
  传感器缺失；`Missing-SAR` 与 `Missing-HSI` 才近似独立模态缺失。当前最小版本将
  `Noisy` 预注册为 HSI stream 的归一化空间高斯噪声，`sigma=1.0`、噪声 seed 为
  `20260829`，并对两个模型和任意显式传入的已有 checkpoint 对使用同一份按样本固定的
  噪声；如更换 stream、强度或 seed，必须同步修改实验记录。
- 当前最小计算量为 `2 个模型 × 1 个已有 seed × 6 个条件 = 12 次验证`。记录 mIoU、
  F1、ACC、Kappa、相对 Clean 的性能下降，以及受损 stream 的 gate 下降和其他 stream
  的补偿上升；只有在显式提供多个已有 checkpoint 对时，才做跨 seed 配对汇总。
- `PaddleCD/paddleseg/models/cx_uper.py` 已加入推理期 gate capture 和分支级扰动接口，
  默认两参数 forward 行为保持不变；`tools/eval_exp04_pmrg_evidence.py` 已实现
  指标、gate 统计、固定噪声、PNG/NPZ 可视化和 manifest 输出；
  `scripts/exp04_pmrg_evidence.sh` 直接调用该 evaluator。
- 合成输入的 6 条件 model dry-run 已通过；真实验证仍需在可访问配置数据根目录的
  `mrsn` 环境中运行。真实数据运行前应确认两个固定 checkpoint 存在，并检查生成的
  `ana/exp04/metrics.csv`、`gate_stats.csv`、`gate_deltas_from_clean.csv` 和
  `manifest.json`。

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
- **Protocol:** Relative to the existing ordinary BW experiments (`cxup_1b_BW.yml`, `cxup_2b_BW.yml`, `cxup_3b_BW.yml`, and `cxup_4b_BW.yml`), add 1B-Small and 1B-Base runs by replacing the 1B ConvNeXt-Tiny backbone with a larger size. Keep the existing BW split, loss, crop, training budget, and seed, then compare the resulting capacities and accuracies with 2B-Tiny, 3B-Tiny, and 4B-Tiny.
- **Record:** Actual trainable parameters, FLOPs, inference speed, mIoU, macro-F1, OA/ACC, and Kappa. The direct supplement uses the existing seed `1919810`; repeat with additional seeds only when extending the uncertainty analysis.
- **Implementation note:** `CX_Uper` has explicit `convnext_small`, `convnext_base`, and `convnext_large` routes, while unknown names retain the ConvNeXt-Base fallback for compatibility. Standard ConvNeXt-Base is a closer 1B reference for 3B/4B than ConvNeXt-Tiny; exact 4B matching would require a separate width-adjusted backbone.
- **Deliverables:** Configs/checkpoints, capacity and accuracy table, and seed summary.

#### EXP-03 — Repeated runs and uncertainty

- **Goal:** Test whether the branch-count improvement and the 4B PMRG/loss improvements are
  repeatable rather than artifacts of one random seed.
- **Conditions:** Run `cxup_1b_BW.yml`, `cxup_2b_BW.yml`, `cxup_3b_BW.yml`, and
  `cxup_4b_BW.yml` for the branch-count chain; then run `cxup_4b_BW_PMRG.yml`,
  `cxup_4b_BW_loss.yml`, and `cxup_4b_BW_PMRG_v2_lossV2.yml` for the 4B component
  ablations. The 4B baseline is shared by both comparisons. Do not count the byte-identical
  `cxup_4b_BW_PMRG_ML.yml` as another condition.
- **Protocol:** Use the same ordinary BW split, preprocessing, crop, optimizer, training budget,
  initialization policy, and data root for all conditions. Use the three fixed seeds
  `1919810`, `1919811`, and `1919812` initially (five seeds are preferable if compute permits).
- **Record:** For every run, record mIoU, macro-F1/F1, OA/ACC, Kappa, checkpoint/config path,
  seed, Params, FLOPs, and FPS. Report per-condition mean $\pm$ standard deviation and retain
  all per-seed values.
- **Stability analysis:** For matched seeds, calculate `2B-1B`, `3B-2B`, and `4B-3B` to assess
  the branch trend, plus the total `4B-1B` change. Calculate `4B+PMRG-4B`,
  `4B+Loss-4B`, and `4B+PMRG+Loss-4B` to assess the component gains; also report
  `4B+PMRG+Loss-4B+PMRG` and `4B+PMRG+Loss-4B+Loss` as matched-factor cross-checks.
  Treat a gain as repeatable only when its direction is consistent across seeds and the
  summary uncertainty is reported; this is evidence of stability, not a causal proof by itself.
- **Deliverables:** Per-run result table, mean $\pm$ std summary, matched-seed delta table,
  seed list, and checkpoint/config paths.

#### EXP-04 — PMRG evidence experiment

- **Goal:** Provide direct evidence for PMRG through gate visualization and matched
  missing/noisy-stream evaluation, without retraining or depending on another EXP.
- **Part A — Gate visualization:** Use the existing checkpoints for
  `cxup_4b_BW_PMRG.yml` and save the three gate outputs generated by
  `CX_Uper_4B_PMRG_V2` at $1/4$, $1/8$, and $1/16$. The gate order is
  `NIRGB | RGB | SAR | HSI`. Upsample the maps to the input size and show the original
  RGB image, GT, clean prediction, four stream maps, and an error-region overlay. Report
  validation-set mean/std gate weights and, where metadata permits, class- or scene-level
  summaries. Gate weights describe feature modulation and are not calibrated reliability
  probabilities.
- **Part B — Missing/noisy streams:** On the same BW validation set, compare the matched
  `cxup_4b_BW.yml` (`CX_Uper_4B`) and `cxup_4b_BW_PMRG.yml`
  (`CX_Uper_4B_PMRG_V2`) existing checkpoints. Use six conditions:
  `Clean`, `Missing-RGB`, `Missing-NIRGB`, `Missing-SAR`, `Missing-HSI`, and one fixed
  Gaussian-noise condition. The minimum preregistered noise condition is HSI noise in
  normalized space with sigma `1.0`, noise seed `20260829`, and the same per-sample noise
  realization for both models and any explicitly supplied checkpoint pair.
- **Perturbation semantics:** Apply missing-stream replacement after `Normalize2` and after
  the model's branch split; zero then means replacement by the branch's per-channel training
  mean. If perturbation is implemented before normalization, use the configured `mean1/mean2`
  values rather than raw zero. Label RGB and NIRGB conditions as `branch-level view missing`
  because those views share MSI bands; label raw-MSI masking as sensor-level missing instead.
  SAR and HSI masking is closer to independent modality missingness.
- **Record:** For every model, checkpoint pair, and condition, record mIoU, macro-F1/F1, OA/ACC, Kappa,
  per-class metrics where practical, and
  $\Delta\mathrm{mIoU}=\mathrm{mIoU}_{clean}-\mathrm{mIoU}_{perturbed}$. Also record mean
  gate changes for the damaged and compensating streams. Report all per-run values; when
  multiple explicitly supplied checkpoint pairs exist, add matched-pair mean $\pm$ standard
  deviation. Do not treat pixels as independent training replicates.
- **Compute:** The current independent design is `2 models x 1 existing seed x 6 conditions = 12`
  validation passes. Checkpoint paths must be verified to match the two named configurations;
  no EXP-03 output directory, `dual-view` shorthand, or full-loss checkpoint is an implicit
  substitute.
- **Deliverables:** Gate-map figures, gate statistics, perturbation metric table, clean-to-
  corrupted drops, matched-seed PMRG comparison when multiple explicit pairs exist,
  seed/checkpoint/config manifest, and the evaluator configuration. The evaluator is an
  independent EXP-04 implementation.

#### EXP-05 — CMX-adapted two-stream baseline

- **Goal:** Add a modern cross-modal fusion baseline and compare it fairly with the existing 2B model.
- **Input and protocol:** Use exactly the same two input groups as 2B: `MSI+SAR | HSI`. Keep the split, preprocessing, crop size, optimizer, training budget, and seeds identical. Evaluate on BW first.
- **Reporting:** Put the CMX result in the final main results summary table, together with the existing 2B and complete-model results. Report mIoU, macro-F1, OA/ACC, Kappa, Params, FLOPs, and FPS.
- **Method note:** The result is a CMX-adapted C2Seg two-stream baseline, not an exact reproduction of the official RGB-X setting. Record the changed input stems, preprocessing, initialization, and any implementation differences.
- **Deliverables:** Config/code, checkpoint, final-summary-table row, and adaptation note.

#### REPO-01 — Repository documentation and naming alignment

- **Scope:** This is a documentation-only task. The public-documentation portion has now been applied to `README.md` and `docs/REPRODUCIBILITY.md`; no source code, configuration, technical identifier, or checkpoint was renamed.
- **Current name:** Use `MRSFN` for the current submitted model and extension work (the multi-branch, PMRG, and mixed-loss configuration). Treat `MBFM` as a legacy name that should not appear as the current model name in public documentation.
- **Prior work:** Keep `MRSN` for the authors' earlier WHISPERS 2023 model, its baseline results, historical experiments, and citation. Do not mechanically replace every occurrence of `MRSN` with `MRSFN`.
- **Repository relationship:** Explain that MRSFN continues the authors' earlier MRSN work. Link the original MRSN repository as `https://github.com/huilin16/MRSN` and the current repository as `https://github.com/huilin66/MRSFN`.
- **README.md:** Update the title, current-model description, clone URL, and directory command to MRSFN; retain the original MRSN citation and clearly distinguish the prior model from the current extension.
- **docs/REPRODUCIBILITY.md:** Change the current-project scope to `MRSN/MRSFN` and rename only the final current-model row from `MBFM (4-branch + PMRG + ML)` to `MRSFN (4-branch + PMRG + ML)`. Keep MRSN baseline rows and historical descriptions unchanged.
- **Technical and historical names to preserve:** Keep `MRSN_BW.yml`, `MRSN_AB.yml`, `MRSN_BW_city.yml`, `CX_Uper*`, `cxup_*`, historical challenge/paper materials, and `mrsn` identifiers when they refer to the prior MRSN baseline. Change a `mrsn` label only when it actually denotes the current final model.
- **Verification after execution:** Search public repository documentation for residual current-model uses of `MBFM`, verify both repository links and clone commands, and confirm that the original MRSN model remains separately traceable. Do not push changes as part of this task.
- **Current status:** The public-documentation search is clean: `README.md` and `docs/REPRODUCIBILITY.md` use `MRSFN` for the current model, retain MRSN for the earlier model, contain the corrected repository relationship/clone commands, and no longer use `MBFM` as a public model label. No push was performed.

## 8. 交接约定

- 不要把内部验证集指标表述为官方测试集结果。
- 所有实验记录应同时保存 config、checkpoint、数据划分、tile/crop/stride、seed 和运行环境信息。
- 涉及城市级实验时，明确记录训练城市、验证城市，以及场景是否参与过模型选择。
- 每个实验都要明确区分 `planned`、`executed`、`verified`；没有真实运行记录时不能填写结果或声称已完成。
- EXP-03 及后续组件实验应保留固定 seed、逐次结果和汇总统计；均值 ± 标准差不能替代逐次结果。
- 当前模型公共名称使用 MRSFN，历史模型使用 MRSN，MBFM 仅作为旧名称解释；保留技术配置名和历史标识以保证复现。
- REPO-01 执行前不得批量重命名 `MRSN`、`CX_Uper*`、`cxup_*` 或配置文件；不在本次 handoff 更新中推送提交。
- 修改前先检查 Git 工作区和本地未跟踪产物，避免覆盖已有实验结果。
