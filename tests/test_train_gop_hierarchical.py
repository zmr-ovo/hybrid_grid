import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gop_lora import TrainableLoRAParameters
from train_gop_hierarchical import (
    _later_entries,
    _local_epochs,
    _selected_evaluation,
    _target_entries,
    _train_stage,
    _validate_args,
    load_hierarchical_checkpoint,
    make_optimizer,
    make_parser,
    save_hierarchical_checkpoint,
)
from train_gop_lora import Evaluation


class HierarchicalTrainingEntryTest(unittest.TestCase):
    def test_parser_defaults_to_a_short_smoke_run(self):
        args = make_parser().parse_args([
            '-d', './data/bunny',
            '--anchor_checkpoint', './anchor_best.pth',
        ])

        self.assertEqual(args.common_epochs, 10)
        self.assertEqual(args.local_epochs, 5)
        self.assertEqual(args.short_gop_epochs, 5)
        self.assertEqual(args.stage, 'all')
        _validate_args(args)

    def test_optimizer_keeps_network_and_grid_learning_rates_separate(self):
        parameters = TrainableLoRAParameters(
            network=(torch.nn.Parameter(torch.zeros(2)),),
            grid=(torch.nn.Parameter(torch.zeros(3)),),
        )

        optimizer = make_optimizer(
            parameters,
            network_lr=1e-3,
            grid_lr=5e-3,
            weight_decay=0.0,
        )

        self.assertEqual(
            [group['name'] for group in optimizer.param_groups],
            ['network_lora', 'grid_lora'],
        )
        self.assertEqual(
            [group['base_lr'] for group in optimizer.param_groups],
            [1e-3, 5e-3],
        )

    def test_short_final_gop_uses_its_own_epoch_count(self):
        full = SimpleNamespace(
            gop_index=3,
            loader=SimpleNamespace(dataset=tuple(range(30))),
        )
        short = SimpleNamespace(
            gop_index=4,
            loader=SimpleNamespace(dataset=tuple(range(12))),
        )
        args = SimpleNamespace(
            gop_size=30,
            local_epochs=80,
            short_gop_epochs=150,
        )

        self.assertEqual(_local_epochs(full, args, total_frames=132), 80)
        self.assertEqual(_local_epochs(short, args, total_frames=132), 150)

    def test_common_stage_excludes_gop_zero(self):
        entries = tuple(
            SimpleNamespace(gop_index=index) for index in range(4)
        )

        self.assertEqual(
            tuple(entry.gop_index for entry in _later_entries(entries)),
            (1, 2, 3),
        )

    def test_training_stage_runs_exactly_the_requested_epochs(self):
        parameters = TrainableLoRAParameters(
            network=(torch.nn.Parameter(torch.zeros(2)),),
            grid=(torch.nn.Parameter(torch.zeros(3)),),
        )
        metrics = SimpleNamespace(loss=0.1, psnr=20.0, msssim=0.8)

        with patch(
            'train_gop_hierarchical.train_epoch',
            return_value=metrics,
        ) as mocked_train_epoch:
            result = _train_stage(
                model=object(),
                entries=(object(),),
                parameters=parameters,
                loss_fn=object(),
                device=torch.device('cpu'),
                epochs=3,
                network_lr=1e-3,
                grid_lr=5e-3,
                warmup=0.05,
                log_interval=10,
                phase='TEST',
                seed=42,
                weight_decay=0.0,
            )

        self.assertEqual(mocked_train_epoch.call_count, 3)
        self.assertEqual(result.best_train_psnr, 20.0)
        self.assertEqual(result.best_val_psnr, 0.0)
        for epoch, call in enumerate(mocked_train_epoch.call_args_list):
            self.assertEqual(call.kwargs['epoch'], epoch)
            self.assertEqual(call.kwargs['epochs'], 3)

    def test_selected_evaluation_uses_frame_weighted_average(self):
        result = Evaluation(
            psnr=20.0,
            msssim=0.8,
            per_gop={
                0: (30.0, 0.9, 30),
                1: (20.0, 0.8, 30),
                2: (10.0, 0.6, 10),
            },
        )

        selected = _selected_evaluation(result, (1, 2))

        self.assertAlmostEqual(selected.psnr, 17.5)
        self.assertAlmostEqual(selected.msssim, 0.75)
        self.assertEqual(set(selected.per_gop), {1, 2})

    def test_target_entries_select_requested_later_gops(self):
        entries = tuple(
            SimpleNamespace(gop_index=index) for index in range(1, 5)
        )

        selected = _target_entries(entries, (1, 3))

        self.assertEqual(
            tuple(entry.gop_index for entry in selected), (1, 3),
        )

    def test_hierarchical_checkpoint_round_trip(self):
        class TinyModel(torch.nn.Module):
            architecture = 'hybrid_grid_hierarchical_gop_lora_v1'

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        args = make_parser().parse_args([
            '-d', './data/bunny',
            '--anchor_checkpoint', './anchor_best.pth',
        ])
        dataset = SimpleNamespace(gop_size=30, num_gops=5, total_frames=132)
        generator = torch.Generator().manual_seed(42)
        entries = (SimpleNamespace(gop_index=1, generator=generator),)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'common_latest.pth')
            save_hierarchical_checkpoint(
                model, optimizer, epoch=2, best_val_psnr=24.0,
                best_train_psnr=25.0, best_epoch=1, args=args,
                dataset=dataset, entries=entries, stage='common', path=path,
            )
            model.weight.data.zero_()
            result, start_epoch = load_hierarchical_checkpoint(
                path, model, torch.device('cpu'), args, dataset,
                optimizer=optimizer, entries=entries,
                expected_stage='common', restore_training=True,
            )

        self.assertEqual(start_epoch, 3)
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.best_val_psnr, 24.0)
        self.assertEqual(model.weight.item(), 1.0)

    def test_rejects_invalid_warmup(self):
        args = make_parser().parse_args([
            '-d', './data/bunny',
            '--anchor_checkpoint', './anchor_best.pth',
            '--common_warmup', '1.0',
        ])

        with self.assertRaisesRegex(ValueError, 'common_warmup'):
            _validate_args(args)


if __name__ == '__main__':
    unittest.main()
