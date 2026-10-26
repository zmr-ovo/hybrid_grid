# Hierarchical Gated Additive-Skip Model

This is the revised Scheme A for the scalable representation study. It uses a
7/2/1 Grid hierarchy and explicitly assigns Grid, Gate and temporal-modulation
parameters to Base, E1 and E2:

- B: L0--L6 + shared PE, Base Grid/PE gates and a full Additive-Skip decoder;
- Mid: B hidden + gated L7--L8 feature through a lightweight E1 adapter;
- Full: Mid hidden + gated L9 feature through a lightweight E2 adapter.

The time branch uses one shared 1-to-64 trunk and three small gamma/beta heads.
The RGB head is shared by all prefixes. E1 and E2 therefore learn refinements
instead of rebuilding an RGB frame independently.

The first experiment is uncompressed on purpose. It tests whether progressive
feature refinement removes the Full-quality conflict observed in the old
zero-padded shared decoder. Quantization and entropy coding are not enabled.

Run from the repository root:

```bash
python -m scalable_module_resnet.train \
  -d /absolute/path/to/frames \
  --level_group_sizes 7 2 1 \
  --prefix_weights 1 1 1 \
  --branch_dim 16
```
