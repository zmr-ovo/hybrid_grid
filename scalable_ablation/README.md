# Scalable representation controlled experiments

This directory tracks four experiments in priority order. Each starts from
random initialization and uses its own training entry and output directory.

| Priority | Experiment | Status |
| --- | --- | --- |
| 1 | Grid prefixes with a full-width Decoder | Implemented in `grid_only/` |
| 2 | Decoder width prefixes with all Grids present | Implemented in `decoder_only/` |
| 3 | Grid + width prefixes with online Full-to-prefix distillation | Planned |
| 4 | Fixed block feature dimensions with variable internal MLP widths | Implemented in `internal_mlp/` |

## Grid-only

Entry: `python -m scalable_ablation.grid_only.train`.

Compare against `scalable_prefix_resnet.train_ema`, using identical model
settings, EMA shares 20/30/50, seed, dataset, resolution, learning rate,
warmup and epoch count. Only Decoder width truncation is disabled: Base,
Mid and Full all execute its full width. Gate and temporal modulation remain
shared exactly as in the EMA reference. No compression modules are enabled.

With ten Grids, the default hierarchy is 7/2/1. The three active Grid counts
are 7/9/10. Network weights are all needed at every operating point; savings
come from the omitted Grids. Lower levels need not have substantially faster
inference. Peak memory may be higher than narrower prefixes, but the three
backward passes remain sequential.

The reference width-ratio metadata is retained internally to reproduce
initialization. Every actual Decoder call uses its Full level. Logs and
checkpoint `hidden_widths` report the actual full widths for all levels.

Checkpoints: `grid_only_best_full.pth`, `grid_only_latest.pth`. Resume and
eval-only accept only this experiment's checkpoints. Use the original EMA
target shares on resume. This entry inherits the reference EMA loader and
training code; it does not restore DataLoader RNG progress on resume.

Tests: `python -m unittest discover -s scalable_ablation/tests -p 'test_grid_only.py' -v`.

Compare validation Base/Mid/Full against the 300-epoch EMA reference
29.68/34.00/36.07 dB. An improvement identifies the additional cost of width
truncation; it does not prove a particular gradient mechanism. If Full stays
similar, investigate Grid-prefix adaptation and the shared Gate/modulation.

## Decoder-only

Entry: `python -m scalable_ablation.decoder_only.train`.

All operating points use all Grid layers, with shared Gate and temporal
modulation receiving identical features for a given frame. Only Decoder
hidden widths change (default 128/192/256). Loss, EMA, initialization, data
and optimizer follow the same reference as Grid-only. No compression is
enabled. This is primarily a compute-scalability diagnostic: all points
need the full Grid payload, so it does not provide Grid bitrate scalability.

The checkpoint records the hierarchy for compatibility, but it does not
control Grid masking in this experiment. `active_grid_levels` records all
levels; `hidden_widths` records the actual widths. Outputs are isolated in
`output/scalable_ablation/decoder_only`, with checkpoints named
`decoder_only_best_full.pth` and `decoder_only_latest.pth`.

Tests: `python -m unittest discover -s scalable_ablation/tests -p 'test_decoder_only.py' -v`.

Compare with Grid-only and the original joint Grid/width prefix experiment
to separate the effects of feature removal and narrower decoding. Do not
compare low-level PSNR as if these methods used equal bitrate: Decoder-only
retains all Grid parameters at every level.

## Fixed module dimensions, internal hidden-width prefixes

Entry: `python -m scalable_ablation.internal_mlp.train`.

This adopts S-NVRC Section 3.4's d -> h_k -> d principle for a pixel-wise MLP.
It does not reproduce the paper's spatial ConvNeXt decoder or training loss.
Grid masking, Gate and temporal modulation follow the original EMA model.

Default Decoder:

```text
fused features (144)
  -> stem + GELU (128)
  -> residual MLP 1: 128 -> h_k -> 128
  -> residual MLP 2: 128 -> h_k -> 128
  -> add input projection (144 -> 128), GELU
  -> residual MLP 3: 128 -> h_k -> 128
  -> shared output projection + GELU (128 -> 128)
  -> shared RGB head + Sigmoid
```

Each block computes x + Linear2(GELU(Linear1(x))). Its input/output dimension
is always 128; only Linear1 output rows, its bias, and Linear2 input columns
are sliced. Linear2 output bias is shared across all levels. Grid prefixes
contain 7/9/10 levels and internal widths are 120/180/240. The stem, input
skip and output head are shared at full size, with no per-level heads.

`--feature_dim 128 --hidden_dim 240 --num_blocks 3` gives 239,315 Decoder
parameters. For the existing Bunny 10-level Grid configuration, total
parameters are 2,784,213, versus the old 2,783,749 (+464, about 0.017%).
This changes network depth and computation, not just slicing. Full output
is not equivalent to the old Decoder and old checkpoints cannot be resumed.
Residuals use ordinary initialization, with no extra normalization or
learned gates inside the new Decoder.

Training uses the same sequential three-prefix EMA 20/30/50 loss as the
other controlled experiments. It starts from scratch with no compression
or distillation. Keep data, LR, seed and epochs unchanged for comparison.
Module dimensions, widths, hierarchy, EMA and optimizer are checkpointed.
Files are `internal_mlp_best_full.pth` and `internal_mlp_latest.pth` under
`output/scalable_ablation/internal_mlp`. This architecture needs its own
Full-only control if results require separating capacity from joint loss.

Tests: `python -m unittest discover -s scalable_ablation/tests -p 'test_internal_mlp.py' -v`.
Verify actual slicing against dense masked MLPs, fixed intermediate shapes,
inactive Grid/channel gradients, budgets and a sequential optimizer step.

## Gate, time and Decoder with internal-width prefixes

Entry: `python -m scalable_ablation.unified_internal.train`.

This additional comparison extends `internal_mlp` by making Gate and time
modulation internally slimmable too. All module input/output dimensions
remain fixed. Gate changes from one Linear to Linear -> GELU -> Linear ->
Sigmoid; the two Grid/PE gates each own one shared MLP, not per-level heads.
Time remains Linear -> ReLU -> Linear -> gamma/beta with the reference
`GroupNorm(x) * gamma + beta` formula (no new identity offset).

Defaults for Bunny:

| Module | Base | Mid | Full | Fixed interface |
| --- | --- | --- | --- | --- |
| Grid | 7 levels | 9 levels | 10 levels | zero-filled 60 features |
| Grid Gate hidden | 24 | 36 | 48 | 144 -> 60 |
| PE Gate hidden | 24 | 36 | 48 | 144 -> 84 |
| Time hidden | 32 | 48 | 64 | 1 -> 288 (gamma/beta) |
| Decoder block hidden | 120 | 180 | 240 | 128 -> 128 |

Default model parameter total: 2,784,309; this is +96 compared with the
Decoder-only-internal-width architecture (2,784,213), +560 versus the old
2,783,749 model. Gate total is 20,976, compared to 20,880 previously.
The low hidden channels remain shared: this does not eliminate all possible
optimization conflicts. Same-time gamma/beta may now differ across levels.

The new model preserves Grid/PE/Decoder initialization of `internal_mlp`
for the same seed; at default time hidden width 64 it also copies the
original Full time initialization. Gate necessarily changes its function.
This comparison changes both Gate parameterization and hidden-channel
activation. It cannot isolate Gate changes from time changes individually.

Training still uses sequential three-prefix EMA 20/30/50, without rate,
quantization or distillation. Checkpoints use a distinct architecture and
strategy, record all module widths and hierarchy, and restore Adam step
tensors on CPU. Resume only this experiment's checkpoints with the same
configuration. DataLoader RNG progress is not restored, as in the reference.

Outputs: `output/scalable_ablation/unified_internal`, with
`unified_internal_best_full.pth` and `unified_internal_latest.pth`.
Tests: `python -m unittest discover -s scalable_ablation/tests -p 'test_unified_internal.py' -v`.
