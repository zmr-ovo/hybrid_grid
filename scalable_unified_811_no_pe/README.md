# Unified 811，仅Grid特征，内部最大宽度300

独立实验目录，不修改其他模型、训练入口或checkpoint。
依赖项目原有模块，包括校准实验的 `PrefixFeatureCalibration`。

## 结构

`坐标 -> Grid插值 -> Grid Gate -> 时间调制 -> Base/Mid校准 -> Decoder -> RGB`。

- 无PE编码器、无PE Gate，不保留假的84维占位或无用参数。
- 坐标仍用于Grid插值，原始时间t仍输入时间MLP；不是完全移除坐标。
- 10层Grid、每层6维，按811使用前8/9/10层，缺失层特征补零，接口60维。
- Grid Gate：`60 -> 24/36/48 -> 60`。
- 时间MLP：`1 -> 32/48/64 -> 120`，产生60维gamma与60维beta，原公式不变。
- Base/Mid校准各60维scale/bias，共240参数，恒等初始化；Full绕过。
- Decoder输入和输入skip投影为 `60 -> 128`；3个模块外部维度128，
  内部最大宽度300，三档为150/225/300。保留内部残差与Additive Skip。

## 参数预算

| 档位 | 实际前向有效参数 | 保留低档功能的累计参数 |
|---|---:|---:|
| Base | 997,917 | 997,917 |
| Mid | 1,659,914 | 1,660,034 |
| Full | 2,783,023 | 2,783,263 |

对照有PE校准版本为2,784,885个总参数；本版本少1,622个，约0.058%。
这是近似参数匹配，不是严格相等。没有用额外无用参数凑数。
总参数接近不意味着三档参数量、计算量和延时都相同。
同时删除PE并扩大隐藏层，因此本实验检验的是相近总预算下的设计取舍，
不是“固定全部网络维度、只删除PE信息”的单因素消融。

同seed下Grid初始化应与有PE版本相同；不同尺寸的Gate/时间模块/Decoder
无法保持逐张量初始化或初始输出一致，不把它们作为兼容断点。

## 训练与运行

保持300轮、lr=0.005、warmup=0.2、weight_decay=0、seed=42，
EMA目标占比20/30/50、decay=0.99，每batch三档顺序反向传播，一次更新。
无预训练、蒸馏、量化、熵模型或额外损失。
独立训练文件沿用有PE校准版循环，只改模型构造、实验标签、默认宽度及路径。

```bash
python -m unittest discover -s scalable_unified_811_no_pe/tests -v
bash scalable_unified_811_no_pe/run.sh 0 /absolute/path/to/bunny 3
bash scalable_unified_811_no_pe/run.sh 0 /absolute/path/to/bunny 300
```

300轮重新从零开始，不继承smoke。脚本打印详细日志路径。
终端日志：`logs/scalable_unified_811_no_pe/`。
模型、内部日志、TensorBoard：`output/scalable_unified_811_no_pe/<timestamp>/<exp_name>/`。
保存 `no_pe_latest.pth` 与 `no_pe_best_full.pth`，同时记录三档最佳PSNR，
但best_full文件不等于每个档位各自最优的模型权重。

入口：`python -m scalable_unified_811_no_pe.train`，默认hidden_dim=300。
继承的 `--pe_freq` 参数仅为配置兼容，固定默认0且不会创建PE；不要传14。
支持本实验同配置的 `--resume PATH` 和 `--eval_only --resume PATH`。
拒绝有PE/其他实验checkpoint；恢复优化器时step保持CPU，兼容旧PyTorch。
数据随机顺序不保证与不中断运行逐步一致。

主要比较相同预算的有PE校准版；三档PSNR、MS-SSIM与训练耗时都要记录。
此处仅记录实现和配置，没有预先填写任何质量结果。
