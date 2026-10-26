# Unified 811 + 原Additive-Skip通道前缀（无校准）

这是新目录。旧 `scalable_unified_811/` 的三残差MLP实验不变，也不覆盖旧结果。

## 模型

沿用Unified的两层Gate与时间调制内部通道前缀，但Decoder直接复用
`scalable_prefix_resnet.model.PrefixAdditiveSkipDecoder`：

```text
固定输入144 -> w -> w -> w -> 固定128 -> RGB
      |                 ^
      +-- Linear(144,w)-+  只在第三隐藏层相加，再GELU
```

- Grid前缀：811，即8/9/10层；PE仍为84维。
- Decoder w：128/192/256，所有档位共享一套权重，通过行列切片激活。
- 只有一个输入Additive Skip；不注册InternalPrefixMLP，不新增三处残差。
- Gate内部宽度24/36/48；固定输入144、输出60/84。
- 时间MLP内部宽度32/48/64，固定输出288（144维gamma和beta）。
- 未启用校准，额外参数0。

默认总参数2,783,845；累计/活动参数为949,445 / 1,627,837 / 2,783,845。
旧三MLP Unified总参数2,784,309，新版少464；低档预算也不同。
这是Decoder结构对照，不是所有档位完全等参数对照。

同seed下Decoder初始化与原Additive前缀实现一致。由于Unified Gate与普通
baseline不同，不宣称整个Full路径与37.03 checkpoint等价。
与旧三MLP版本使用相同Gate/时间调制定义，但构造顺序改变，
不保证跨这两种不同架构的全部随机初始化张量相同。

## 训练与运行

从零300轮、lr=0.005、warmup=0.2、weight_decay=0、seed=42；
EMA目标损失占比20/30/50，decay=0.99。每batch三档顺序反向传播、一次更新。
没有压缩、预训练、teacher或其他新训练策略。

```bash
python -m unittest discover -s scalable_unified_additive_811/tests -v
bash scalable_unified_additive_811/run.sh 0 /absolute/path/to/bunny 3
bash scalable_unified_additive_811/run.sh 0 /absolute/path/to/bunny 300
```

smoke和完整实验分别从零开始。脚本打印实际详细日志位置。
输出：`logs/scalable_unified_additive_811/`、
`output/scalable_unified_additive_811/<timestamp>/<exp_name>/`。
checkpoint：`additive_latest.pth`、`additive_best_full.pth`。

支持独立入口 `python -m scalable_unified_additive_811.train`，以及同配置
`--resume PATH`、`--eval_only --resume PATH`。拒绝旧三MLP、校准版等其他断点。
恢复模型、优化器、EMA、轮数与三档best；optimizer step保持CPU以兼容旧PyTorch。
恢复后随机数据顺序不保证与完全不中断时逐步一致。

## 对照关系

1. 对比旧 `scalable_unified_811`：主要观察替换Decoder的效果。
2. 对比新 `scalable_unified_additive_811_calibrated`：仅观察576个校准参数的效果。
3. 对比原普通811（31.49/33.95/36.09）：同时存在Gate/时间调制分档差异，
   不能误认为只改变校准。

日志分别记录三档distortion、PSNR、MS-SSIM、best、有效权重、loss贡献和LR。
默认hidden_dim=256时输出投影固定128；若自行更改hidden_dim，原类的输出
投影为hidden_dim//2，不属于本次默认对照配置。
