"""Utilities for staged EMA-balanced scalable-prefix training.

The schedule changes only the desired Base/Mid/Full loss contributions.
Distortion EMAs are preserved across stage boundaries, avoiding an abrupt
reset of the effective weights when the curriculum advances.
"""

from dataclasses import dataclass

from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer,
    validate_target_shares,
)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_epoch: int
    end_epoch: int
    target_shares: tuple

    def contains(self, epoch):
        return self.start_epoch <= epoch < self.end_epoch


class PrefixCurriculum:
    """Three-stage Base/Mid/Full contribution schedule.

    Epochs use zero-based, half-open intervals internally. For example,
    boundaries ``60, 150`` with 300 total epochs mean epochs 1-60, 61-150
    and 151-300 in user-facing logs.
    """

    def __init__(
        self,
        total_epochs,
        boundaries=(60, 150),
        stage_shares=((0.05, 0.15, 0.80),
                      (0.10, 0.20, 0.70),
                      (0.15, 0.25, 0.60)),
    ):
        total_epochs = int(total_epochs)
        boundaries = tuple(int(value) for value in boundaries)
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        if len(boundaries) != 2:
            raise ValueError("curriculum_boundaries must contain two epochs")
        if not 0 < boundaries[0] < boundaries[1] < total_epochs:
            raise ValueError(
                "curriculum boundaries must satisfy 0 < first < second < epochs"
            )
        if len(stage_shares) != 3:
            raise ValueError("curriculum requires three target-share groups")

        normalized = tuple(
            validate_target_shares(values) for values in stage_shares
        )
        self.total_epochs = total_epochs
        self.boundaries = boundaries
        self.stages = (
            CurriculumStage("full_priority", 0, boundaries[0], normalized[0]),
            CurriculumStage(
                "transition", boundaries[0], boundaries[1], normalized[1],
            ),
            CurriculumStage(
                "balanced", boundaries[1], total_epochs, normalized[2],
            ),
        )

    def stage(self, epoch):
        epoch = int(epoch)
        if not 0 <= epoch < self.total_epochs:
            raise ValueError("epoch is outside the curriculum")
        for stage in self.stages:
            if stage.contains(epoch):
                return stage
        raise RuntimeError("curriculum stage lookup failed")

    def state_dict(self):
        return {
            "total_epochs": self.total_epochs,
            "boundaries": self.boundaries,
            "stage_shares": tuple(
                stage.target_shares for stage in self.stages
            ),
        }


class CurriculumEMABalancer(EMADistortionBalancer):
    """EMA balancer whose target contribution shares may change by epoch."""

    def set_target_shares(self, shares):
        self.target_shares = validate_target_shares(shares)

    def load_ema_state(self, state):
        if abs(float(state["decay"]) - self.decay) > 1e-12:
            raise ValueError("checkpoint EMA decay does not match")
        ema = state.get("ema")
        self.ema = None if ema is None else tuple(float(value) for value in ema)

