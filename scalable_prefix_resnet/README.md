# Strict Grid and Decoder Prefix Experiment

This directory contains an isolated S-NVRC-style capacity experiment.  It
does not modify or import the training behavior of `scalable_module_resnet`
or `scalable_width_resnet`.

## Design

- Grid prefix: Base uses L0-L6, Mid uses L0-L8, Full uses L0-L9.
- Decoder prefix: Base/Mid/Full use 50%/75%/100% of the same hidden channels.
- Shared components: PE, Gate, temporal modulation, the final projection and
  RGB head are identical at all operating points.
- Full compatibility: the Full forward path and parameter tensors match the
  Additive-Skip `model_1` baseline.
- Sampling: every mini-batch trains exactly one operating point in a balanced
  Base -> Mid -> Full round-robin sequence.

This first version intentionally excludes quantization, entropy rate and real
bitstreams.  Its purpose is to verify that multi-level optimization can retain
the Full reconstruction capability before compression is reintroduced.

## Full-checkpoint warm start

The direct-prefix evaluator loads an Additive-Skip checkpoint without
training and reports its Base, Mid and Full truncation quality:

```bash
python -m scalable_prefix_resnet.evaluate_checkpoint_prefixes \
  --checkpoint /path/to/train_latest.pth -d /path/to/frames
```

The warm-start entry initializes the same prefixes from that checkpoint and
performs pure-distortion fine-tuning.  Each batch backpropagates Base, Mid and
Full sequentially, followed by one optimizer update.  Full is anchored to a
frozen copy of the source model.  Rate, quantization and entropy coding remain
disabled:

```bash
python -m scalable_prefix_resnet.train_warmstart \
  --checkpoint /path/to/train_latest.pth -d /path/to/frames --epochs 100
```

## Entry

Run from the repository root:

```bash
python -m scalable_prefix_resnet.train -d ./data/bunny [options]
```

## EMA-normalized joint training

`train_ema` is an independent from-scratch capacity experiment.  Every
mini-batch recomputes Base, Mid and Full sequentially and performs one shared
optimizer update.  It tracks the moving-average distortion of each prefix and
automatically adjusts detached effective weights so their expected loss
contributions follow `20% / 30% / 50%`.  This prevents the larger Base and Mid
errors from numerically overwhelming Full.

The entry does not load a Full checkpoint, use a teacher, or enable rate and
quantization:

```bash
python -m scalable_prefix_resnet.train_ema \
  -d ./data/bunny \
  --target_shares 0.2 0.3 0.5 \
  --ema_decay 0.99
```

Checkpoints include the EMA state, so resumed training preserves the same
loss scale instead of reinitializing it from the first resumed batch.
