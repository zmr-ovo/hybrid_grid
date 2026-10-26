"""Keep all Grid features and vary only the Decoder hidden-channel prefix.

Gate and temporal modulation receive identical features at all three levels.
Initialization and the Full path match the original strict-prefix model.
"""

from scalable_prefix_resnet.model import PrefixScalableResNet


class DecoderOnlyScalableResNet(PrefixScalableResNet):
    architecture = "hybrid_grid_ablation_decoder_only_additive_skip_v1"

    def _grid_prefix(self, coords_hw, quality_level, grids):
        self.hierarchy.validate_quality_level(quality_level)
        return super()._grid_prefix(coords_hw, 2, grids)

    def parameter_budget(self):
        common = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if not name.startswith("decoder.")
        )
        return tuple(
            common + self.decoder.active_parameter_count(quality_level)
            for quality_level in range(3)
        )
