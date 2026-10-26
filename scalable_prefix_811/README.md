# 8/1/1 双前缀对照实验

入口：`python -m scalable_prefix_811.train`，从项目根目录运行。

只把原 `scalable_prefix_resnet.train_ema` 的 Grid 分组从 7/2/1 改为
8/1/1。复用相同模型与训练循环，不修改已有模块。Full 的结构、参数量、
随机初始化及计算路径不因分组改变；Base 新增 L7，Mid 和 Full 的活动层不变。

| 档位 | Grid | Decoder 宽度 | 默认配置累计有效参数 |
|---|---|---:|---:|
| Base | L0–L7 | 128 | 969,045 |
| Mid | L0–L8 | 192 | 1,637,589 |
| Full | L0–L9 | 256 | 2,783,749 |

默认 EMA 目标损失占比 20/30/50，decay=0.99；每 batch 三档顺序反向传播，
一次参数更新。纯失真、随机初始化，无量化、熵模型、蒸馏或预训练。
目标损失占比不等于固定系数，也不等于梯度占比。

输出默认保存在 `output/scalable_prefix_811/<timestamp>/<exp_name>/`。
日志及 TensorBoard、`ema_latest.pth`、`ema_best_full.pth` 沿用参考训练器格式。
checkpoint 保存实际 hierarchy、hidden_widths、EMA 和各档 best PSNR；
本入口拒绝用 7/2/1 checkpoint 续训。正常对照实验不要传 `--resume`。

与 7/2/1 对比时保持 300 epochs、lr=0.005、warmup=0.2、seed=42、
分辨率、Grid 配置以及 EMA 配置一致，比较同一轮的 Val Base/Mid/Full。
参数量是有效标量数量，不是压缩 bits。暂不保证 Full 不掉点。

测试：`python -m unittest discover -s scalable_prefix_811/tests -v`。

## 单档独立训练对照

- `python -m scalable_prefix_811.train_base_only`：只训练 Base，8 层 Grid、
  128 宽 Decoder，有效参数 969,045。
- `python -m scalable_prefix_811.train_mid_only`：只训练 Mid，9 层 Grid、
  192 宽 Decoder，有效参数 1,637,589。

两者使用相同随机初始化流程和默认 300 轮、lr=0.005、warmup=0.2、seed=42。
每一帧都用于选定档位；不使用 EMA、其他档损失或联合训练 checkpoint。
这些是独立能力参考，不是最终方案的预训练阶段，也不是理论上限保证。
继承的 target_shares/ema_decay 参数在单档中无作用。

默认输出分别位于 `output/scalable_prefix_811_single/base_only/` 与
`output/scalable_prefix_811_single/mid_only/`，含日志、TensorBoard、
`single_best.pth`（按该档 Val PSNR）、`single_latest.pth`。
`--resume PATH` 恢复相同档位的断点，`--eval_only --resume PATH` 仅评测该档。
续训会恢复模型、优化器和 epoch，但不保证随机数据顺序与未中断运行逐步一致。
为兼容保留完整模型张量，checkpoint 大小不代表档位有效参数，更不是实际码率。

顺序运行（先在服务器激活环境）：

```bash
bash scalable_prefix_811/run_single_queue.sh 0 /absolute/path/to/bunny 300
```

用 3 替换 300 可做 smoke；脚本仅依次运行这两个有限实验，Base 失败不会启动
Mid。它不等待或修改当前 811 任务，必须选用可用且有权限使用的 GPU。
两阶段之间不共享权重；不预约 GPU，不保证进程切换期间仍有空闲显存。
终端日志存入 `logs/prefix_811_single/<run_id>/`，脚本会打印实际位置。
可通过 `PYTHON_BIN` 和 `WORKERS` 环境变量指定解释器及 worker 数。
