# 历史实验与论文实现映射

> 状态：在重构代码前固化历史依据。  
> 范围：ICASSP 小论文、旧 `videotest` 仓库，以及当前 `hybrid_grid` 仓库。  
> 本文只记录能够由论文、源码和历史命令文件支持的结论；没有证据的内容不会写成既定事实。

## 1. 核心结论

旧仓库中，最接近论文“利用熵模型进行端到端压缩”实验的代码链是：

```text
videotest/ZZZ/train_entropy_qat.py
    -> videotest/ZZZ/model_entropy.py
        -> videotest/ZZZ/entropy.py
        -> videotest/ZZZ/encoding.py
```

与之对应、可信度最高的历史运行命令位于：

```text
videotest/ZZZ/new.txt
videotest/ZZZ/qat.txt
videotest/ZZZ/1.txt
```

作出这一判断的直接依据有四点：

1. `model_entropy.py` 为每个网格层分别建立了一个 `EntropyBottleneck`。
2. 训练时对网格参数使用加性噪声或离散符号量化，并用概率似然构造可微码率项。
3. 量化后的网格确实通过 `hash_encoder(..., grids=quantized_grids)` 参与重建，因此失真是由量化表示产生的。
4. `train_entropy_qat.py` 优化率失真目标，并在评估阶段对非网格参数做 8 bit 量化和 Huffman 长度估计。

需要特别说明：虽然文件名包含 `qat`，旧代码只对网格做了训练期可微量化；非网格网络权重是在评估阶段才量化，实质上是 PTQ，不是真正的 QAT。

`Z_New` 下的代码是一次生成真实码流的后续尝试，但存在会破坏正确性的错误，不能直接迁移。`hashnerv_design/train_compress.py` 则是更早期的原型，架构和训练调度与论文最终描述的距离更远。

## 2. 证据等级与代码版本

本文使用以下证据等级：

- **已确认**：可由源码、论文或历史命令直接验证。
- **强推断**：多个独立证据相互吻合，但没有 checkpoint 或完整日志证明它就是论文最终运行。
- **待确认**：现有仓库历史不足以建立精确对应关系。

本次核对的材料如下：

| 材料 | 版本或位置 | 用途 |
|---|---|---|
| `hybrid_grid` | 本地 `main`，commit `545e004` | 后续整理和实现的目标仓库 |
| `videotest` | GitHub 仓库中的 `ZZZ`、`Z_New`、`hashnerv_design` | 历史代码来源 |
| ICASSP 小论文 | *A Hybrid Grid-Based Method for Video Representation*，DOI `10.1109/ICASSP55912.2026.11463942` | 方法和实验目标 |

目前没有发现可以把“论文曲线中的每一个点”与“某条命令和某个 checkpoint”一一对应的结果清单，也没有发现完整可解码码流。因此，本文不会把某条历史命令直接认定为论文最终配置。

## 3. 论文目标实现

论文的主体表示可以概括为：

```text
帧坐标 (x, y, t)
  -> 多分辨率显式三维网格
  -> 自适应位置编码与门控
  -> 时间感知通道调制
  -> 轻量解码器
  -> RGB 帧
```

压缩部分进一步加入：

- 每个网格层独立的熵模型；
- 网格参数的可微量化；
- 其他网络参数的低比特量化；
- 率失真联合优化：`L = distortion + lambda * rate`。

与复现直接相关的论文设置：

| 项目 | 论文设置 |
|---|---|
| 优化器 | AdamW |
| 初始学习率 | `5e-3` |
| 学习率调度 | cosine decay |
| 重建实验训练轮数 | 1200 epochs |
| 压缩实验训练轮数 | 300 epochs |
| UVG 数据 | 7 个序列，每个取连续 120 帧，1080p |
| lambda 调度 | 前期为 0，之后线性增长 |
| 论文描述的最大 lambda | `5e-4` |
| 非网格参数 | 压缩实验中采用 8 bit 量化 |

后续可作为回归参照、但暂时不能作为自动验收阈值的论文结果：

| 实验 | 论文结果 |
|---|---|
| Bunny 0.35M/0.75M/1.5M/3M 平均重建质量 | 34.79 dB |
| Bunny 3M 重建质量 | 37.88 dB |
| Bunny 3M 完整模型消融 | 37.88 dB |
| UVG 压缩比较 | 相同 BPP 下最高 PSNR 提升 7.25% |

之所以暂时不能把这些数值作为程序测试阈值，是因为历史代码中缺少最终 checkpoint、随机种子、精确预处理清单以及论文点位与命令的映射。

## 4. 旧仓库代码谱系

| 历史代码 | 可微熵损失 | 量化网格参与重建 | 非网格量化 | 可靠真实码流 | 结论 |
|---|---:|---:|---:|---:|---|
| `ZZZ/train_entropy_qat.py` + `ZZZ/model_entropy.py` | 是 | 是 | 评估期 8 bit PTQ | 否 | **主要迁移来源** |
| `ZZZ/train_entropy.py` + `ZZZ/model_entropy.py` | 是 | 是 | 按 32 bit 统计 | 否 | 网格熵模型消融参考 |
| `hashnerv_design/train_compress.py` | 是 | 取决于早期实现 | 按 32 bit 统计 | 否 | 早期原型，仅供参考 |
| `Z_New/train_entropy.py` + `Z_New/model_entropy.py` | 尝试实现 | **编码分支中没有** | 不完整 | 尝试实现 | 有关键错误，不直接迁移 |
| `Z_New/train_entropy_com.py` | 尝试实现 | 接口不一致 | 不完整 | 尝试实现 | 只保留设计意图 |

### 4.1 `ZZZ` 主链的实际训练过程

```text
原始网格参数
  -> 各层独立 EntropyBottleneck
     -> 训练前期：加性噪声近似量化
     -> 训练后期：round 得到离散符号
  -> 量化网格
  -> 网格插值与解码器
  -> 重建帧

符号似然
  -> -log2(p)
  -> 平均每个网格值的估计 bit 数
  -> distortion + lambda(epoch) * rate_per_grid_value
```

在 `ZZZ/model_entropy.py` 中，各层估计 bit 数先乘以该层网格参数量，再对所有层求和，最后除以网格参数总数。因此模型返回的 `bits` 是“平均每个网格值的 bit 数”，不是视频 BPP。

在 `ZZZ/train_entropy_qat.py` 中：

- `epoch < epochs * mode_switch_ratio` 时使用 `noise`，之后使用 `symbols`；
- 调度后的 lambda 与平均网格 bit 数相乘；
- 评估时对非网格张量做 `quant_bit` 位均匀量化；
- 使用离散整数的频数估计 Huffman 编码长度；
- 最后将网格估计 bit 与非网格估计 bit 合并为 BPP。

### 4.2 `train_entropy.py` 为什么不是完整压缩路径

`ZZZ/train_entropy.py` 包含可微网格熵约束，但评估时把所有非网格参数按 32 bit 统计。它适合验证“只压缩网格”的消融，不对应论文中“网格熵编码 + 其余参数 8 bit 量化”的完整方案。

### 4.3 `Z_New` 为什么不能作为迁移基线

`Z_New` 尝试在评估中执行 `compress -> decompress`，但存在以下阻断性问题：

1. 已构造 `quantized_grids`，随后却调用 `self.hash_encoder(normalized_coords)`，没有把量化网格传入重建。
2. 离散符号转换直接把 `(value - min)` 转成整数，没有先除以量化步长，小浮点值会大量坍缩到同一个符号。
3. PMF/CDF 的构造维度更接近张量位置，而不是定义明确的符号字母表。
4. 不同通道的符号支持长度可能不同，却被直接 `stack`。
5. `ZZZ` 与 `Z_New` 中的 `compress`、`decompress` 和 likelihood 调用参数不一致。
6. shape、dtype、scale、CDF 等解码元数据没有形成明确的流格式，无法在独立进程中仅凭码流解码。

可以保留的只有接口目标：新实现应当提供清晰的 `compress()` 和 `decompress()`，并统计真实文件长度；现有实现本身应重写。

## 5. 历史命令与参数映射

### 5.1 最接近论文描述的公共配置

`ZZZ/new.txt` 中带 `_new` 的记录与论文压缩协议吻合度最高：

| 参数 | 值 |
|---|---:|
| 入口脚本 | `train_entropy_qat.py` |
| batch size | 1 |
| epochs | 300 |
| learning rate | `5e-3` |
| distortion | L2 |
| 分辨率 | 1080 x 1920 |
| lambda base | `1e-7` |
| lambda max | `5e-4` |
| 量化模式切换比例 | `0.9` |
| 非网格参数量化 | 8 bit |

其中 `lambda_max=5e-4`、300 epochs 和 8 bit 与论文描述直接吻合。因此将其判断为最接近最终实验的命令记录，证据等级为**强推断**。

### 5.2 `ZZZ/new.txt` 中保留的精确配置

| 历史标签 | 序列 | 网格层数 | 每层特征维度 | PE 频率数 | 隐层 | 时间缩放 | 基础/最高分辨率 | warm-up | 网格量化步长 |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0.75M | go | 7 | 6 | 10 | 256 | 0.5 | 10 / 38 | 30 | `1e-3` |
| 0.75M | yacht | 7 | 6 | 10 | 256 | 0.5 | 10 / 38 | 30 | `1e-3` |
| 0.75M | shake | 7 | 6 | 10 | 256 | 0.3 | 14 / 42 | 30 | `1e-4` |
| 2.3M | go | 10 | 6 | 14 | 256 | 0.5 | 12 / 55 | 30 | `1e-3` |
| 2.3M | yacht | 10 | 6 | 14 | 256 | 0.4 | 14 / 58 | 50 | `5e-4` |
| 2.3M | jockey | 10 | 6 | 14 | 256 | 0.5 | 12 / 55 | 30 | `1e-3` |

`ZZZ/qat.txt` 还记录了覆盖全部七个 UVG 序列的更大范围搜索，常见 `lambda_max=1e-4`，并包含 `5e-5`、`2e-4`、`5e-4` 等变体。`ZZZ/1.txt` 主要是 `train_entropy.py` 的早期试验。这些文件证明进行了多轮超参数探索，但不能合并成一个虚构的“唯一最终配置”。

### 5.3 非压缩/架构实验命令

`Z_New/order_new.txt` 记录了使用 `train_1.py` 的 UVG 3M 标签架构试验。公共参数通常为：网格层数 10、每层特征维度 6、PE 频率数 14、隐层 256、300 epochs、学习率 `5e-3`。

代表性序列参数如下：

| 序列 | 时间缩放 | 基础分辨率 | 最高分辨率 | 文件中同时存在的备选项 |
|---|---:|---:|---:|---|
| beauty | 0.1 | 20 | 105 | 0.5 / 14 / 60 |
| bosphorus | 0.3 | 17 | 70 | 0.1 / 20 / 105；0.5 / 14 / 60 |
| go | 0.5 | 14 | 60 | 0.6 / 12 / 57 |
| honeybee | 0.1 | 20 | 105 | 无 |
| jockey | 0.3 | 17 | 70 | 无 |
| shake | 0.3 | 17 或 18 | 70 | 两条相邻试验 |
| yacht | 0.5 | 14 或 15 | 60 | 两条相邻试验 |

压缩命令使用 `2_3M` 标签，而论文展示 3M 模型点，二者关系目前**待确认**。可能原因包括参数统计口径、熵模型开销或后续调参，但在找到 checkpoint 和日志前不能选定其中任何一种解释。

## 6. 码率口径

重构后必须把以下三个量使用不同变量名和日志标签，禁止继续统称为 `bits` 或 `bpp`。

### 6.1 历史网格平均码率

旧模型训练时返回：

```text
legacy_rate_per_value
    = 所有网格层估计 bit 数之和 / 所有网格参数数量
```

它的单位是 `bit / grid value`，用于可微率失真损失。

### 6.2 估计视频 BPP

对于 `T * H * W` 个视频像素，旧评估大致采用：

```text
estimated_grid_bits = grid_parameter_count * legacy_rate_per_value
estimated_video_bpp
    = (estimated_grid_bits + estimated_non_grid_bits) / (T * H * W)
```

这是估计值。旧实现没有完整且一致地计入 shape、scale、CDF/概率表、熵模型参数、padding/alignment 和容器头等信息。

### 6.3 真实视频 BPP

目标实现必须采用：

```text
actual_video_bpp
    = 完整可解码文件的字节数 * 8 / (T * H * W)
```

“完整可解码文件”必须包含：网格码流、其余量化权重、熵模型状态或概率表、张量 shape/dtype/scale、架构配置以及必要的流头。实验中需要并列报告 `estimated_video_bpp` 和 `actual_video_bpp`。

## 7. 旧代码中必须修复、不能照搬的问题

### 7.1 训练与评估驱动

- `eval_only` 可能在加载指定 checkpoint 前执行，导致评估随机模型。
- 一处 `eval_only` 调用传参与 `evaluate` 签名不一致。
- `evaluate` 依赖全局 `args`，接口不明确，也不利于单元测试。
- 计时代码无条件调用 CUDA 同步，CPU 环境会报错。
- GPU 推理计时没有形成完整的前后同步窗口，FPS 可能被高估。
- 部分平均值除以 dataset 长度，部分除以 loader 长度；旧实验因 batch size=1 才没有暴露问题。
- `best_loss` 没有被一致更新。
- checkpoint 中 epoch 的语义不统一，恢复训练可能重复或跳过一轮。
- `quant_bit == -1` 时，评估返回值中可能出现未赋值变量。

### 7.2 量化与码率统计

- 一些均匀量化使用 `2**bits` 作分母；若端点都可表示，应使用与 `2**bits - 1` 一致的约定。
- 常数张量或全零张量求 scale 时缺少稳健处理。
- 评估量化会跳过 `hash_encoder` 和位置编码张量，但 BPP 的参数归类未始终采用相同规则。
- Huffman 表开销采用经验常数估计，而不是实际序列化格式。
- 熵模型参数和必要解码元数据没有完整计入 BPP。
- 旧 `compress_decompress_levels()` 存在量化步长参数缺失/不一致和字符串字节数统计错误。

### 7.3 当前 `hybrid_grid` 基线问题

这些问题要在加入熵模型前解决：

- `fixed_res` 只改变坐标尺寸，没有同步 resize 图像，会导致坐标与 RGB 目标尺寸不一致。
- 时间坐标使用 `frame_idx / total_frames`；需要明确保留旧行为，还是使用 `max(total_frames - 1, 1)` 令末帧映射到 1。
- `eval_only` 位于 resume/checkpoint 加载之前。
- `eval_only` 的调用参数与 `evaluate` 定义不一致。
- FPS 统计无条件进行 CUDA 同步，且计时窗口不完整。
- 统计代码混用 dataset 与 dataloader 的长度。
- 当前均匀量化同样存在分母约定和常数张量问题。
- 当前默认 `Decoder` 增加了卷积细化层，而 `DecoderOri` 更接近论文的轻量 MLP 基线。
- 当前默认 `SpatioTemporalModulation` 使用逐像素 `(x,y,t)`，而 `TemporalModulation` 更接近论文的仅时间通道调制。
- 当前位置编码默认频率底数与论文时期的指数底数行为不同。

上述架构变化可以作为毕业论文的新尝试，但必须通过显式配置控制，不能在复现实验中静默替换论文基线。

## 8. 下一阶段的迁移边界

应从 `ZZZ` 迁移的行为：

- 每个网格层独立建模；
- 基于 likelihood 的可微码率估计；
- 前期噪声量化、后期离散符号量化；
- 使用量化网格进行重建；
- lambda 率失真调度；
- 网格参数和其他网络参数分别统计码率。

必须重写而不是复制的部分：

- `Z_New` 的算术编码尝试；
- 依赖全局参数的评估函数；
- 混在评估流程中的 PTQ/Huffman 代码；
- 非正式的字节数与元数据估算；
- shape、dtype、scale 和 codec 状态均不明确的隐式接口。

目标接口应表达以下边界：

```text
model.forward(coords, quantization_mode)
    -> reconstruction, rate_breakdown

codec.compress(model, metadata)
    -> serialized_representation, exact_bit_count

codec.decompress(serialized_representation)
    -> 可在独立进程解码的模型/表示

evaluator(...)
    -> distortion_metrics, estimated_bpp, actual_bpp, timing_metadata
```

## 9. 新实验必须保存的复现清单

每次运行至少保存：

- 代码 commit 和工作区是否有未提交修改；
- 完整展开后的配置；
- 随机种子和确定性设置；
- 数据集、精确帧列表、帧数、空间分辨率和 resize 规则；
- 模型 preset，以及各组件精确参数量；
- 优化器、学习率调度、epoch 和 batch size；
- distortion 定义及 reduction 方式；
- lambda 和量化模式调度；
- 各网格层量化步长；
- 网格估计 bit、非网格 bit、side information bit 和总 BPP；
- codec 开启时的实际序列化字节数；
- 逐帧及汇总 PSNR/MS-SSIM，并记录汇总方式；
- device、dtype、计时预热、同步策略和 FPS；
- checkpoint 路径及 best/final checkpoint 选择规则。

## 10. 现有材料无法回答的问题

1. UVG RD 曲线的每个论文点分别来自哪个 checkpoint？
2. 论文中的 3M 是否包含熵模型参数，和命令中的 `2_3M` 使用了何种不同口径？
3. 同一序列存在多条配置时，最终采用的是哪一条？
4. 各论文实验的随机种子和精确帧区间是什么？
5. 论文 BPP 是否完整包含熵模型、元数据和编码表开销，还是沿用了旧代码估计值？
6. 上传仓库之外是否曾存在修正后的真实 codec？

如果找到旧 output 目录、`.pth` checkpoint、训练日志、shell history 或导出压缩文件，应先建立清单，再更新本文中的证据等级。

## 11. 审核门槛

本文只完成“历史实验固化”，尚未修改任何模型或训练逻辑。进入代码实现前，需要确认：

- 以 `ZZZ` 主链作为熵模型迁移来源；
- 接受“可微估计码率”和“真实文件 BPP”是两个不同指标；
- 论文兼容 preset 默认使用原始解码器与仅时间调制；
- `2.3M` 等历史标签不被无证据地改写为论文 3M 配置；
- 无法证明的历史信息继续显式标记为待确认。

审核通过后，按 `plan.md` 的顺序继续：先修复并稳定当前基线和评估契约，再加入可微网格熵模型，最后实现真实 codec 与真实文件 BPP。
