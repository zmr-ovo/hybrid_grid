# Hybrid Grid 后续研究与实现计划

## 1. 目标与当前基础

后续工作围绕两条主线展开：

1. 在固定总参数预算下，利用 GOP 间的公共信息和 GOP 内的独立信息，提高长视频表示效率。
2. 改进 Grid 熵模型，在尽量保持重建质量和计算效率的条件下降低整体码率，并完成真实码流验证。

当前已经完成：

- 未压缩论文模型的训练、恢复、评测和 FPS 流程整理。
- Grid 可微量化、基础熵模型、端到端率失真训练和整体估计 BPP。
- 非 Grid 参数的差异化 QAT：二维 Linear 权重使用按输出通道 8-bit QAT，一维参数与熵模型参数保持 FP32。
- 固定 GOP 数据接口、GOP 局部时间坐标、网络 LoRA、矩阵 Grid LoRA、结构化 Grid LoRA和单 GOP 全参数微调对照。
- 分层全 GOP 模型组装：一份 GOP0 共享模型、一份供所有后续 GOP 使用的公共 LoRA，以及每个后续 GOP 的独立 LoRA。
- 分层模型的公共/独立参数统计与路由单元测试。
- 公共训练阶段和独立训练阶段的参数冻结接口，以及 optimizer step更新隔离测试。
- `train_gop_hierarchical.py` 分层训练入口：公共 LoRA训练、逐 GOP独立 LoRA训练、完整验证、checkpoint、断点恢复和 eval-only。

已有实验表明：只依靠由 GOP0 训练的共享模型和独立 LoRA，后续 GOP 需要独自承担过多内容变化；Grid 是影响后续 GOP 适配能力的重要部分。因此，下一步不再更新共享模型，而是在共享模型与独立 LoRA之间增加一份后续 GOP公共 LoRA。

## 2. 总体原则

### 2.1 保持已有入口独立

- `train.py`：未压缩全视频基线。
- `train_compression.py`：端到端压缩训练。
- `train_gop_upper_bound.py`：单 GOP 全参数微调上限。
- `train_gop_grid_residual.py`：已有单 GOP Grid/LoRA实验。
- 新的分层 GOP训练使用独立入口，不修改以上脚本的默认行为。

### 2.2 每次只解决一个问题

- 先完成参数冻结与梯度隔离，再实现训练循环。
- 先验证公共 LoRA是否改善后续 GOP，再研究 rank分配。
- GOP-LoRA主线稳定后，再继续熵模型改进。
- 熵模型的概率估计有效后，再实现算术编码和真实 BPP。

### 2.3 公平比较

- 使用相同数据、分辨率、随机种子和评测代码。
- GOP方案统计共享模型、公共 LoRA和全部独立 LoRA的总参数量。
- 主要比较使用与全视频基线相近的总参数预算。
- 训练集最佳 PSNR只作为训练状态，不作为最终重建结果。
- 未完成算术编码前，码率统一标注为估计 BPP。

## 3. 阶段一 分层 GOP-LoRA

### 3.1 模型结构

模型只保留以下一种全 GOP组装方式：

```text
GOP0 = 共享模型

GOPk = 共享模型
     + 后续 GOP公共网络 LoRA
     + 后续 GOP公共结构化 Grid LoRA
     + GOPk独立网络 LoRA
     + GOPk独立结构化 Grid LoRA
```

其中 `k >= 1`。GOP0不使用公共或独立 LoRA，保证已经训练好的 GOP0共享模型不被改变。

等价地：

```text
theta_0 = theta_shared
theta_k = theta_shared + delta_common + delta_k
```

- `theta_shared`：GOP0训练得到并在后续阶段冻结的共享模型。
- `delta_common`：所有后续 GOP共同使用的网络与 Grid LoRA。
- `delta_k`：当前 GOP独有的网络与 Grid LoRA。

### 3.2 文件结构

```text
hybrid_grid/
├── train_gop_hierarchical.py       # 后续新增的分层训练入口
├── gop_lora/
│   ├── joint.py                    # 模型组装、参数分组和统计
│   ├── model.py                    # 分层全 GOP模型
│   ├── grid_residual.py            # 公共与独立结构化 Grid LoRA
│   ├── injection.py                # 网络 LoRA注入
│   ├── linear.py                   # 公共与独立线性 LoRA
│   ├── partition.py
│   └── video_dataset.py
└── tests/
    ├── test_gop_joint_model.py
    └── test_train_gop_hierarchical.py
```

不再新增 `train_gop_joint.py`，因为共享模型不会参与全 GOP联合更新。

### 3.3 已完成内容

- `assemble_all_gop_model()` 直接组装分层模型，不保留旧的“只有独立 LoRA”全 GOP模式。
- 网络 LoRA覆盖 `all_linear`，输出层自动限制有效 rank。
- Grid LoRA采用保持空间、时间和通道结构的分解方式。
- 公共网络 LoRA与公共 Grid LoRA只对 GOP1至最后一个 GOP生效。
- 每个后续 GOP拥有独立的网络 LoRA和 Grid LoRA。
- 参数分组可以分别统计共享模型、公共网络、公共 Grid、独立网络和独立 Grid参数。
- 已增加零初始化、GOP路由、公共/独立隔离以及参数无重复无遗漏测试。

### 3.4 三阶段训练流程

#### 阶段 A GOP0共享模型

- 使用 GOP0训练较小的共享模型。
- Bunny当前实验使用300轮，已有 checkpoint可以直接复用。
- 加载时检查模型结构、Grid层数、特征维度、GOP长度和时间尺度。

#### 阶段 B 后续 GOP公共 LoRA

- 冻结整个 GOP0共享模型。
- 冻结所有 GOP独立 LoRA。
- 只训练公共网络 LoRA和公共 Grid LoRA。
- 训练数据为 GOP1至最后一个 GOP的全部帧，不包含 GOP0。
- 第一轮正式实验最多150轮，warmup比例先使用 `0.05`。
- 每10轮评测一次；连续3次评测提升不足 `0.02 dB` 时允许提前停止。

#### 阶段 C 每 GOP独立 LoRA

- 冻结共享模型和公共 LoRA，但公共 LoRA继续参与前向。
- 每次只训练当前 GOP的独立网络 LoRA和独立 Grid LoRA。
- 其他 GOP独立 LoRA全部冻结。
- 30帧 GOP第一轮最多训练80轮。
- Bunny最后一个12帧 GOP可最多训练150轮，使实际更新次数与完整 GOP接近。
- 每10轮评测一次，并采用与阶段 B一致的早停原则。

### 3.5 已完成的参数控制

在 `gop_lora/joint.py` 中增加清晰的训练阶段接口：

```text
configure_common_training(model)
configure_local_training(model, gop_index)
```

公共训练阶段：

- `shared.requires_grad = False`
- `common_network.requires_grad = True`
- `common_grid.requires_grad = True`
- `local.requires_grad = False`

独立训练阶段：

- `shared.requires_grad = False`
- `common.requires_grad = False`
- 当前 GOP独立 LoRA为 `True`
- 其他 GOP独立 LoRA为 `False`

对应测试执行真实的 forward、backward和 optimizer step，并比较更新前后的参数值，而不只检查 `requires_grad`。

### 3.6 数据遍历与指标

- 阶段 B按 GOP顺序遍历 GOP1至最后一个 GOP，每个 GOP内部保持帧级 shuffle。
- 一个 batch必须只包含一个 GOP，避免混用独立 LoRA。
- 全视频指标按照实际帧数加权，最后一个较短 GOP不能与完整 GOP拥有相同权重。
- 分别报告 GOP0、每个后续 GOP和全视频的 PSNR、MS-SSIM。
- 阶段 C训练一个 GOP时，仍需评测完整视频，检查是否错误影响其他 GOP。
- 已实现完整视频一次前向评测，并从同一结果中计算阶段选择指标：公共阶段以所有后续 GOP的帧数加权 PSNR选择最佳模型，独立阶段以当前 GOP的 PSNR选择最佳模型。

### 3.7 参数量与预算

Bunny第一轮实验以全视频基线约2.78M总参数为目标：

- GOP0共享模型约1.76M。
- 剩余约1.02M分配给公共 LoRA和全部独立 LoRA。
- 公共网络/Grid rank与独立网络/Grid rank必须分别配置。
- 正式训练前输出实际参数量，不依赖人工估算。
- 如果超过目标预算，应在训练开始前明确警告；自动 rank搜索放到阶段二。

### 3.8 Checkpoint

训练入口完成后，checkpoint至少保存：

- checkpoint类型与版本。
- GOP0共享模型参数。
- 公共网络 LoRA和公共 Grid LoRA。
- 全部 GOP独立网络 LoRA和独立 Grid LoRA。
- 当前训练阶段、epoch与当前 GOP。
- 优化器、学习率调度器和随机状态。
- 模型、GOP和 LoRA配置。
- 公共阶段最佳结果、各 GOP独立阶段最佳结果和完整视频评测结果。

需要支持：

- 从公共 LoRA阶段恢复。
- 从任意 GOP独立训练阶段恢复。
- `eval-only`评测中间或最终 checkpoint。

以上内容已实现。入口支持 `--stage all/common/local/eval`、`--target_gops`、`--hierarchical_checkpoint` 和 `--resume`；保存 `common_best/latest.pth`、`gop_k_best/latest.pth` 与 `final.pth`。

### 3.9 日志

启动时输出：

- 帧数、GOP长度、GOP数量和各 GOP帧数。
- 共享模型参数量。
- 公共网络 LoRA和公共 Grid LoRA参数量。
- 每个 GOP独立网络 LoRA和 Grid LoRA参数量。
- LoRA总参数量、模型总参数量及与目标预算的差值。
- 当前阶段所有参数组的学习率。

训练和验证时输出：

- 当前阶段、epoch和当前 GOP。
- loss、PSNR、MS-SSIM和学习率。
- 每个 GOP及完整视频指标。
- 最佳结果、对应阶段和 epoch。

### 3.10 阶段一验收条件

- GOP0只使用共享模型。
- 每个后续 GOP同时使用同一份公共 LoRA和自己的独立 LoRA。
- 公共阶段只更新公共 LoRA。
- 独立阶段只更新当前 GOP独立 LoRA。
- 参数统计无遗漏、无重复。
- 一个命令可以完成公共 LoRA训练和后续独立 LoRA训练。
- 支持断点恢复和 `eval-only`。
- 固定随机种子后重复评测结果一致。
- 不影响 baseline、compression及已有单 GOP入口。

## 4. 阶段二 固定参数预算下的 rank分配

阶段一确认分层方案有效后，再优化公共与独立 LoRA的预算比例：

- 分别配置公共 network rank、公共 Grid rank、独立 network rank和独立 Grid rank。
- 支持为不同 GOP设置不同独立 Grid rank。
- 根据实际 Grid尺寸计算每层和每个 GOP参数量。
- 增加 `--report_params_only`，只组装模型并输出参数量。
- 增加总参数预算检查，超过预算时在训练前报错。
- 先使用固定 rank，再考虑按帧数、初始重建误差或内容变化分配 rank。

主要比较：

- 只有独立 LoRA的已有单 GOP结果。
- 公共 LoRA，不加独立 LoRA。
- 公共 LoRA加独立 LoRA。
- 不同公共/独立预算比例。
- 相同约2.78M总参数量下与全视频基线比较。

## 5. 阶段三 GOP方案实验与决策

必做实验：

- 全视频未压缩基线。
- GOP0共享模型。
- 单 GOP全参数微调上限。
- 已有网络 LoRA、完整 Grid增量、矩阵 Grid LoRA和结构化 Grid LoRA。
- 后续 GOP公共 LoRA。
- 公共 LoRA加每 GOP独立 LoRA。
- 固定总参数预算下的 rank分配。

将 GOP-LoRA作为论文主要提升方向需要满足：

- 相近总参数量下完整视频质量稳定超过或接近全视频基线。
- 公共 LoRA相对只有独立 LoRA能够提高质量或减少独立参数。
- 多个序列上趋势一致。
- GOP间没有严重质量失衡。
- 额外推理和 GOP切换开销可接受。

如果最终未超过基线，则保留为消融与失败分析，并将主要精力转向熵模型改进。

## 6. 阶段四 轻量跨层条件熵模型

### 6.1 分布诊断

先新增离线分析工具，输出：

- 每层 Grid符号直方图、经验熵和模型交叉熵。
- 交叉熵与经验熵的差值。
- 每层对总 Grid bit的贡献比例。
- 不同量化步长下的符号范围与零值比例。

如果经验熵高，优先改进量化；如果经验熵较低而交叉熵高，优先改进熵模型。

### 6.2 轻量条件模型

- 最粗层继续使用独立熵模型。
- 已解码粗层通过三线性插值对齐到当前层。
- 轻量通道映射预测当前层符号的均值和尺度。
- 各位置并行计算，不采用逐点自回归或 Transformer。
- 保持现有固定 Grid量化步长与非 Grid混合 QAT策略。

验收时比较相同量化符号下的估计 bit，并将新增熵模型参数计入整体 BPP。

## 7. 阶段五 真实算术编码与真实 BPP

仅在条件熵模型有效后实现：

- 将量化符号和预测分布转换为离散 CDF。
- 实现每层 Grid的 encode/decode。
- 保存形状、量化步长、熵模型配置和必要元数据。
- 验证解码符号逐元素一致。
- 分别统计 Grid码流、非 Grid参数、熵模型和元数据。
- 输出整体真实 BPP，并与估计 BPP比较。
- 推理 FPS与真实编解码时间分开报告。

## 8. 阶段六 完整实验与论文输出

- Bunny用于快速验证和主要消融。
- UVG用于压缩率失真对比。
- DAVIS用于复杂运动、重建质量和速度验证。
- 报告 PSNR、MS-SSIM、参数量、估计/真实 BPP、FPS、编码时间、解码时间和显存。
- 所有结论注明比较对象、参数预算和训练条件。
- GOP-LoRA与熵模型分别消融，核心结果完成后再考虑下游应用。

## 9. 推荐实现顺序

严格按照以下顺序推进：

1. 完成分层全 GOP模型组装与路由测试。已完成，等待服务器 PyTorch环境验证。
2. 实现公共阶段和独立阶段的参数冻结接口，并增加真实 optimizer step隔离测试。已完成，等待服务器 PyTorch环境验证。
3. 新增 `train_gop_hierarchical.py` 最小训练入口，只实现公共阶段与独立阶段的训练遍历。已完成，等待服务器 PyTorch环境验证。
4. 增加分 GOP及完整视频指标汇总。
5. 增加 checkpoint、断点恢复、`eval-only`和完整参数统计。
6. 在 Bunny上运行10轮公共 LoRA加5轮独立 LoRA的 smoke test。
7. 完成公共 LoRA最多150轮、独立 LoRA最多80/150轮的正式实验。
8. 根据实际参数统计实现约2.78M预算下的 rank配置。
9. 与全视频基线、单 GOP上限和已有独立 LoRA结果比较。
10. 确认 GOP方向是否扩展到 UVG和 DAVIS。
11. 新增 Grid分布诊断工具。
12. 实现并验证轻量跨层条件熵模型。
13. 接入算术编码并统计真实 BPP。
14. 完成完整消融、速度评测和论文图表。

## 10. 当前立即执行的任务

下一步只实现：

> 为分层训练入口增加分 GOP训练指标与完整视频验证指标汇总，明确区分公共 LoRA阶段和各独立 LoRA阶段的结果。

本步不实现：

- checkpoint、断点恢复或 `eval-only`。
- 图片导出。
- 自动 rank分配或预算搜索。
- 熵模型、量化和算术编码。
- 多 GPU并行。

恢复服务器连接后，应先运行模型组装、参数隔离和最小训练入口测试；通过后再继续 smoke test。
