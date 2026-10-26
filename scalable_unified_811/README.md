# Unified 811 独立实验

目的：测试原 Unified 721 的低档改善能否与 811 分组合并。
这是待验证方案，不假设两项收益可以相加。

本目录只有新入口、启动脚本和测试；不修改旧模型或训练入口。
复用 `scalable_ablation.unified_internal` 的模型及 EMA 循环，避免复制训练逻辑。
独立目录不代表完全自包含，运行时需保留项目原有依赖模块。

| 档位 | Grid前缀 | Decoder内部宽度 | Gate内部宽度 | 时间调制内部宽度 | 有效参数 |
|---|---|---:|---:|---:|---:|
| Base | L0–L7 | 120 | 24 | 32 | 1,009,325 |
| Mid | L0–L8 | 180 | 36 | 48 | 1,666,201 |
| Full | L0–L9 | 240 | 48 | 64 | 2,784,309 |

Decoder模块外部维度128、3个内部MLP模块；Gate固定输出60/84维，
时间调制gamma/beta合计288维。只有Base增加L7，Mid/Full结构不变。
与Unified 721在相同seed下初始化张量一致，Full路径应完全相同。
这不保证联合训练后的Full质量相同。

默认300轮、Adam lr=0.005、warmup=0.2、weight_decay=0、seed=42。
每batch三档顺序反向传播，一次更新；EMA目标损失占比20/30/50，decay=0.99。
从零训练，不使用预训练、teacher、额外校准、量化或熵模型。

## 运行

在项目根目录、激活环境后：

```bash
python -m unittest discover -s scalable_unified_811/tests -v
bash scalable_unified_811/run.sh 0 /absolute/path/to/bunny 3
bash scalable_unified_811/run.sh 0 /absolute/path/to/bunny 300
```

3轮为smoke，不是300轮预训练；完整实验另行从零开始。
脚本打印详细日志路径：`logs/scalable_unified_811/<name>_<run_id>.log`。
模型、TensorBoard、训练日志位于 `output/scalable_unified_811/<timestamp>/<exp_name>/`。
checkpoint沿用 `unified_internal_latest.pth` 与 `unified_internal_best_full.pth` 名称，
但存放在独立目录；含配置、分组、各模块宽度、EMA、优化器及三档最佳PSNR。
checkpoint架构标签沿用Unified架构，具体分组以metadata为准。

入口支持 `--resume PATH` 和 `--eval_only --resume PATH`，只接受匹配的
Unified 811 checkpoint，拒绝Unified 721和原双前缀811断点。续训仍使用总epochs。
恢复不保证与未中断训练逐步保持相同随机数据顺序。

## 对照

- 原Unified 721，300轮：30.39 / 34.06 / 36.05 dB。
- 原双前缀811，300轮：31.49 / 33.95 / 36.09 dB。

保持seed、分辨率、训练预算和EMA一致，比较同轮Val Base/Mid/Full。
参数量只是有效标量数，不代表压缩BPP或实际计算量。
