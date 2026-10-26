# Unified 811 + 原Additive-Skip通道前缀 + Base/Mid校准

独立目录，不覆盖旧三残差MLP校准版 `scalable_unified_811_calibrated/`。
模型与训练循环复用本次新建的 `scalable_unified_additive_811`，以保证
两个新版本只相差校准；不修改其他入口。

## 结构

`Grid+PE -> Unified Gate -> Unified时间调制 -> 档位校准 -> Additive Decoder`。

Decoder为固定144输入、128/192/256通道前缀、固定128投影、RGB输出。
只在第三隐藏层保留输入Additive Skip，没有三个残差MLP。
Base/Mid各有144维scale和bias，初始化1和0；Full直接绕过。
校准位于时间调制之后，不修改时间MLP生成gamma/beta的公式。
同seed时与新无校准版本的原有参数和初始化输出一致；训练后Full仍可能因
共享参数更新而变化，不承诺完全不掉点。

## 参数预算

| 档位 | 当前前向活动参数 | 包含所有低档功能的累计存储参数 |
|---|---:|---:|
| Base | 949,733 | 949,733 |
| Mid | 1,628,125 | 1,628,413 |
| Full | 2,783,845 | 2,784,421 |

总计新增576个参数。Full不直接使用校准，但累计模型仍存储两组低档校准参数。
上述预算不是压缩BPP。

## 运行

```bash
python -m unittest discover -s scalable_unified_additive_811_calibrated/tests -v
bash scalable_unified_additive_811_calibrated/run.sh 0 /absolute/path/to/bunny 3
bash scalable_unified_additive_811_calibrated/run.sh 0 /absolute/path/to/bunny 300
```

与新无校准版本保持相同300轮、lr=0.005、warmup=0.2、seed=42、
EMA 20/30/50及其余配置；均从零开始，无压缩或teacher。
依赖新无校准目录及项目已有的PrefixFeatureCalibration等组件。

详细日志：`logs/scalable_unified_additive_811_calibrated/`。
模型/TensorBoard：`output/scalable_unified_additive_811_calibrated/<timestamp>/<exp_name>/`。
checkpoint为 `additive_calibrated_latest.pth` 和 `additive_calibrated_best_full.pth`。
保存配置、三档best、校准权重、EMA及优化器。只允许本实验同配置断点续训，
不能直接加载旧三MLP或新无校准版作为resume。
best_full只表示Full最优，不意味着该文件同时具有三档各自最优权重。
数据随机顺序在续训后不保证逐步一致。

测试：恒等初始化、Full旁路、校准参数梯度路由、一次联合更新、参数预算、
断点往返及模型间隔离。两组的实际质量和显存需运行后评估。
