# ResNet Baseline

This directory contains the first controlled architecture experiment for the
scalable-video study. It keeps the uncompressed Hybrid Grid baseline unchanged
except for replacing the decoder's skip concatenation with additive residual
MLP blocks.

Run training from the repository root:

```bash
python -m resnet_baseline.train -d ./data/bunny [other baseline arguments]
```

Checkpoints use the architecture identifier
`hybrid_grid_resnet_baseline_v1`, so they cannot be loaded accidentally by the
original paper baseline.
