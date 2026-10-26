# 原Prefix 811：无PE，保留Base/Mid校准

独立目录，不修改原Prefix 811、Unified系列或已有实验。
这是原Prefix 811的去PE设计，不是Unified h300的版本。

## 结构

```text
坐标 -> Grid插值（60维） -> 单层Grid Gate -> 全宽时间调制
     -> Base/Mid校准，Full直通 -> 原Additive-Skip通道前缀Decoder -> RGB
```

- 10层Grid、6维特征，811分组，前缀为8/9/10层；缺失层补零。
- 真正删除PE编码器和PE Gate，不使用84维占位或无用参数。
- Grid Gate为`Linear(60,60)+Sigmoid`，三档都完整使用，不做Unified隐藏层分档。
- 原TemporalModulation为`1 -> 64 -> 120`，产生60维gamma与beta，
  三档都使用完整64维；GroupNorm和`norm(x)*gamma+beta`公式不变。
- 校准为Base/Mid各60维scale/bias，共240参数，初始化1和0；Full绕过。
- Decoder为`60 -> w -> w -> w -> 128 -> RGB`，w=128/192/256。
  仅第三隐藏层有输入Additive Skip，没有三个残差MLP模块。
- 原始x/y/t仍用于Grid插值，t仍用于时间调制；无PE不等于不用坐标。

## 参数预算

| 档位 | 实际前向有效参数 | 包含低档功能的累计参数 |
|---|---:|---:|
| Base | 919,507 | 919,507 |
| Mid | 1,577,299 | 1,577,419 |
| Full | 2,712,587 | 2,712,827 |

原有PE Prefix 811总参数2,783,749，本版总参数2,712,827，少70,922，约2.55%。
本次不扩大Decoder或Grid补足参数，不是等参数实验。
同时去PE并增加校准，与原811比较不能把所有变化单独归因于PE。
较干净的PE消融还需“原Prefix 811+相同校准”的有PE对照；本目录未提前实现。
Mid和Full累计模型需保存低档校准参数，但其前向不会使用其他档的校准。

## 训练与输出

保持300轮、lr=0.005、warmup=0.2、weight_decay=0、seed=42，
EMA目标损失占比20/30/50、decay=0.99，每batch三档顺序反向传播、一次更新。
从零训练，无teacher、预训练、量化或熵模型。

```bash
python -m unittest discover -s scalable_prefix_811_no_pe_calibrated/tests -v
bash scalable_prefix_811_no_pe_calibrated/run.sh 0 /absolute/path/to/bunny 3
bash scalable_prefix_811_no_pe_calibrated/run.sh 0 /absolute/path/to/bunny 300
```

3轮和300轮均独立从零开始。详细日志路径会由脚本打印。
终端日志：`logs/scalable_prefix_811_no_pe_calibrated/`。
模型、内部日志和TensorBoard：
`output/scalable_prefix_811_no_pe_calibrated/<timestamp>/<exp_name>/`。
保存 `prefix_no_pe_calibrated_latest.pth` 与 `prefix_no_pe_calibrated_best_full.pth`。
含模型/校准权重、EMA、优化器、轮数、各档最佳PSNR与配置。
支持独立CLI的 `--resume PATH`、`--eval_only --resume PATH`，仅接受本实验同配置
断点；拒绝原有PE、Unified和其他模型。optimizer step保持CPU以兼容旧PyTorch。
续训不保证随机数据顺序与未中断时逐步一致。

继承的 `--pe_freq` 保留默认0，模型不会创建PE；不能传14。
旧Unified的 `--gate_hidden_dim`、`--feature_dim`、`--num_blocks`不再适用。
本版无PE、Gate/时间调制定义均由本目录明确实现，校准算子复用已有
`scalable_unified_811_calibrated.model.PrefixFeatureCalibration`，并不实例化Unified网络。
