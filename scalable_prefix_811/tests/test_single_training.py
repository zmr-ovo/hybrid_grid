"""Single-prefix routing, validation and checkpoint safety checks (CPU)."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scalable_prefix_811.single_training import (
    build_parser, freeze_unused_grids, load_checkpoint, run_epoch, save_checkpoint,
)
from scalable_prefix_811.train import build_model


def make_args(quality):
    args = build_parser(quality).parse_args(["-d", "/unused"])
    args.base_resolution = 2
    args.finest_resolution = 5
    args.aspect_ratio = [1, 1]
    args.time_scale = 1.0
    args.grid_feat_dim = 2
    args.pe_freq = 3
    args.hidden_dim = 16
    return args


def make_optimizer(model):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)


class SingleTrainingTest(unittest.TestCase):
    def test_entry_defaults_are_independent_and_match_joint(self):
        base = build_parser("base").parse_args(["-d", "/unused"])
        mid = build_parser("mid").parse_args(["-d", "/unused"])
        self.assertNotEqual(base.out_dir, mid.out_dir)
        self.assertEqual((base.epochs, base.lr, base.warmup), (300, 0.005, 0.2))
        self.assertEqual(base.level_group_sizes, [8, 1, 1])
        self.assertEqual(base.width_ratios, [0.5, 0.75, 1.0])

    def test_each_batch_uses_only_selected_prefix_and_one_update(self):
        for quality, level, active in (("base", 0, 8), ("mid", 1, 9)):
            with self.subTest(quality=quality):
                args = make_args(quality)
                model = build_model(args)
                freeze_unused_grids(model, level)
                optimizer = make_optimizer(model)
                batch = {"coords": torch.rand(1, 3, 12, 12),
                         "pixels": torch.rand(1, 3, 12, 12)}
                before = model.grid_encoder.levels[-1].grid.detach().clone()
                with patch.object(model, "forward_prefix", wraps=model.forward_prefix) as forward:
                    values = run_epoch(model, [batch, batch], torch.device("cpu"), F.mse_loss,
                                       args, optimizer=optimizer, metric_fn=lambda *_: 1.0)
                self.assertEqual([call.args[1] for call in forward.call_args_list], [level, level])
                self.assertGreater(values["distortion"], 0)
                for index, grid in enumerate(model.grid_encoder.levels):
                    self.assertEqual(grid.grid.grad is not None, index < active)
                self.assertTrue(torch.equal(before, model.grid_encoder.levels[-1].grid))
                self.assertTrue(all(int(state["step"]) == 2 for state in optimizer.state.values()))
                before_eval = {key: value.clone() for key, value in model.state_dict().items()}
                with patch.object(model, "forward_prefix", wraps=model.forward_prefix) as forward:
                    run_epoch(model, [batch], torch.device("cpu"), F.mse_loss,
                              args, metric_fn=lambda *_: 1.0)
                self.assertEqual(forward.call_args.args[1], level)
                self.assertTrue(model.training)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before_eval[key]), key)

    def test_resume_round_trip_and_quality_rejection(self):
        args = make_args("base")
        model = build_model(args)
        freeze_unused_grids(model, 0)
        optimizer = make_optimizer(model)
        run_epoch(model, [{"coords": torch.rand(1, 3, 12, 12),
                           "pixels": torch.rand(1, 3, 12, 12)}],
                  torch.device("cpu"), F.mse_loss, args, optimizer=optimizer,
                  metric_fn=lambda *_: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "single.pth")
            save_checkpoint(path, model, optimizer, 3, 30.0, 31.0, args)
            restored = build_model(args)
            freeze_unused_grids(restored, 0)
            restored_optimizer = make_optimizer(restored)
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer, args,
                                             torch.device("cpu")), (4, 30.0, 31.0))
            self.assertTrue(all(state["step"].device.type == "cpu"
                                for state in restored_optimizer.state.values()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            args.quality = "mid"
            with self.assertRaisesRegex(ValueError, "断点不能混用"):
                load_checkpoint(path, restored, restored_optimizer, args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
