# Unified 811 + Base/Mid特征校准

独立目录，不修改 `scalable_unified_811`、原Unified或其他实验。
依赖项目已有模块，单独同步此目录并不使其成为自包含项目。

## 唯一新增的模型功能

`Gate -> 时间调制 -> 档位特征校准 -> Decoder`。

Base/Mid分别使用144维scale和bias；Full直接返回原特征。
scale初始化全1、bias全0，不使用随机初始化，新增576个可学习标量，
不增加Linear或MLP。参数在同一档的帧和像素间共享。
同seed下原有张量初始化不变，初始化三档输出应完全一致；
Full永远不直接使用校准参数，但共享参数训练后仍可能改变Full质量。

Grid仍为811；Decoder内部宽度120/180/240，Gate为24/36/48，
时间MLP为32/48/64。仍从零300轮、lr=0.005、warmup=0.2、seed=42，
EMA目标损失占比20/30/50，每batch三档顺序反向传播、一次更新。
不增加蒸馏、压缩、权重调度或额外损失。

## 参数预算口径

| 档位 | 该档前向实际使用参数 | 保留所有低档功能的累计参数 |
|---|---:|---:|
| Base | 1,009,613 | 1,009,613 |
| Mid | 1,666,489 | 1,666,777 |
| Full | 2,784,309 | 2,784,885 |

Mid前向不使用Base校准，Full前向不使用任何校准，但嵌套模型仍需保存低档
校准参数。因此累计预算与活跃前向预算不同；它们均不是压缩码率。

## 运行和输出

```bash
python -m unittest discover -s scalable_unified_811_calibrated/tests -v
bash scalable_unified_811_calibrated/run.sh 0 /absolute/path/to/bunny 3
bash scalable_unified_811_calibrated/run.sh 0 /absolute/path/to/bunny 300
```

3轮为独立smoke；300轮重新从零开始。
详细终端日志：`logs/scalable_unified_811_calibrated/<name>_<run_id>.log`。
模型、TensorBoard和内部日志：
`output/scalable_unified_811_calibrated/<timestamp>/<exp_name>/`。

- `calibrated_latest.pth`：周期保存或最后一轮。
- `calibrated_best_full.pth`：Full Val PSNR改善时保存；三档best分数分别记录，
  不表示这个文件同时包含三档各自最好的权重。
- checkpoint有独立架构和策略标签，含校准参数、EMA、优化器、配置和分组。
- 支持 `python -m scalable_unified_811_calibrated.train --resume PATH ...`，
  只接受本实验相同配置的断点；`--eval_only --resume PATH`仅评测。
  续训须提供原模型配置，epochs是目标总轮数；恢复后数据随机顺序不保证逐步一致。

日志输出三档distortion、实际权重、loss贡献、PSNR、MS-SSIM、最佳PSNR及LR。
主要对照应是相同配置的未校准Unified 811（其最终质量尚未记录），
其次参考原双前缀811的31.49/33.95/36.09；不要将不同架构误当成仅校准的消融。

测试覆盖：恒等初始化、与未校准模型同seed输出一致、Full校准旁路、
各档校准/活动Grid的梯度路由、总参数与累计预算、一次联合更新、断点恢复及隔离。
校准为待验证方案，不承诺提高质量。
