# Width-Progressive Additive-Skip Model

This directory implements Scheme B independently from the existing baseline,
compression and module-progressive experiments.

- Grid hierarchy: `7/2/1` by default.
- Feature groups: Base 126 channels, E1 12 channels and E2 6 channels.
- Gates: Base is invariant; E1/E2 gates causally inspect decoded lower groups.
- Time modulation: one shared trunk with stage-specific gamma/beta heads.
- Missing enhancement features: deterministic zeros after stage modulation.
- Decoder: the model_1 Additive-Skip MLP.
- Network hierarchy: Base, Mid and Full use strict channel prefixes with
  default width ratios `0.5/0.75/1.0`.
- Decoder input columns are also strict `126/138/144` prefixes.
- Full parameter count remains close to the model_1 single-rate budget.

Every batch uses sequential Base, Mid and Full forward/backward passes followed
by one optimizer update, so every prefix sees every frame without storing three
720p autograd graphs simultaneously.

Run from the repository root:

```bash
python -m scalable_width_resnet.train \
  -d /absolute/path/to/frames \
  --level_group_sizes 7 2 1 \
  --width_ratios 0.5 0.75 1.0 \
  --prefix_weights 1 1 1
```

