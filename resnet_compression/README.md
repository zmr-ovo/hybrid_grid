# ResNet Compression

This directory applies the established Hybrid Grid compression pipeline to
``resnet_baseline/model_1.py``. The Decoder uses two independent projections
instead of a concatenated Linear, while remaining mathematically equivalent to
the original baseline. It changes no compression algorithm:

- differentiable per-level Grid quantization;
- factorized, cross-scale or recursive cross-scale entropy models;
- mixed non-Grid QAT;
- detailed Grid, network, entropy and metadata bit accounting;
- checkpoint-independent `.hgm` model bitstreams.

Run from the repository root:

```bash
python -m resnet_compression.train -d /absolute/path/to/frames [options]
```

The saved configuration contains `reconstruction_model=additive_skip`, allowing the
standard evaluation and codec tools to reconstruct the correct architecture.
