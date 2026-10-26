# Full-only Capacity Test

This isolated experiment trains only the Full prefix of the exact hierarchical
model in `scalable_module_resnet`. Base and Mid are not evaluated during the
training step and contribute no loss or gradient.

Interpretation after 300 epochs:

- Full PSNR near 36.8--37.0 dB: the architecture has enough capacity and the
  main joint-training problem is gradient conflict among the three prefixes.
- Full PSNR still near 35.5--36.0 dB: the late E1/E2 adapters are an
  architectural bottleneck and should be redesigned before tuning weights.

Run from the repository root with:

```bash
python -m test.train_full_only -d /absolute/path/to/frames
```

