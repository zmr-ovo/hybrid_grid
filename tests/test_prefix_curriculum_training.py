import unittest

from scalable_prefix_resnet.curriculum_training import (
    CurriculumEMABalancer,
    PrefixCurriculum,
)


class PrefixCurriculumTest(unittest.TestCase):
    def test_default_stage_boundaries(self):
        curriculum = PrefixCurriculum(300)

        self.assertEqual(curriculum.stage(0).name, "full_priority")
        self.assertEqual(curriculum.stage(59).name, "full_priority")
        self.assertEqual(curriculum.stage(60).name, "transition")
        self.assertEqual(curriculum.stage(149).name, "transition")
        self.assertEqual(curriculum.stage(150).name, "balanced")
        self.assertEqual(curriculum.stage(299).name, "balanced")

    def test_default_target_shares(self):
        curriculum = PrefixCurriculum(300)

        self.assertEqual(
            curriculum.stage(0).target_shares, (0.05, 0.15, 0.8),
        )
        self.assertEqual(
            curriculum.stage(100).target_shares, (0.1, 0.2, 0.7),
        )
        self.assertEqual(
            curriculum.stage(200).target_shares, (0.15, 0.25, 0.6),
        )

    def test_ema_is_preserved_when_target_changes(self):
        balancer = CurriculumEMABalancer((0.05, 0.15, 0.8))
        balancer.update((0.004, 0.002, 0.001))
        original_ema = balancer.ema

        balancer.set_target_shares((0.1, 0.2, 0.7))

        self.assertEqual(balancer.ema, original_ema)
        contributions = [
            weight * distortion
            for weight, distortion in zip(
                balancer.weights(), (0.004, 0.002, 0.001),
            )
        ]
        normalized = [value / sum(contributions) for value in contributions]
        for actual, expected in zip(normalized, (0.1, 0.2, 0.7)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_rejects_invalid_boundaries(self):
        with self.assertRaises(ValueError):
            PrefixCurriculum(300, boundaries=(0, 150))
        with self.assertRaises(ValueError):
            PrefixCurriculum(300, boundaries=(150, 60))
        with self.assertRaises(ValueError):
            PrefixCurriculum(300, boundaries=(60, 300))


if __name__ == "__main__":
    unittest.main()
