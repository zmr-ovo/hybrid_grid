import unittest

from train_gop_grid_residual import _validate_args, make_parser


class TrainGOPGridResidualTest(unittest.TestCase):
    def test_parser_uses_gop2_and_rank8_defaults(self):
        args = make_parser().parse_args([
            '-d', 'frames', '--anchor_checkpoint', 'anchor.pth',
        ])

        self.assertEqual(args.target_gop, 2)
        self.assertEqual(args.rank, 8)
        self.assertEqual(args.alpha, 8.0)
        self.assertEqual(args.grid_rank, 0)
        self.assertEqual(args.grid_alpha, 1.0)
        self.assertEqual(args.grid_init, 'random')
        self.assertEqual(args.grid_adapter, 'auto')
        self.assertEqual(args.lora_target, 'decoder')
        self.assertIsNone(args.grid_lr)
        self.assertIsNone(args.network_lora_lr)
        self.assertEqual(args.out_dir, './output/gop_grid_residual')

    def test_accepts_rank_one_grid_lora(self):
        args = make_parser().parse_args([
            '-d', 'frames', '--anchor_checkpoint', 'anchor.pth',
            '--grid_rank', '1', '--grid_alpha', '1',
        ])

        _validate_args(args)
        self.assertEqual(args.grid_rank, 1)

    def test_accepts_optimized_lora_configuration(self):
        args = make_parser().parse_args([
            '-d', 'frames', '--anchor_checkpoint', 'anchor.pth',
            '--grid_rank', '2', '--grid_alpha', '2',
            '--grid_init', 'anchor_pca',
            '--lora_target', 'all_linear',
            '--grid_lr', '5e-3', '--network_lora_lr', '1e-3',
        ])

        _validate_args(args)
        self.assertEqual(args.grid_init, 'anchor_pca')
        self.assertEqual(args.lora_target, 'all_linear')

    def test_rejects_anchor_gop_as_target(self):
        args = make_parser().parse_args([
            '-d', 'frames', '--anchor_checkpoint', 'anchor.pth',
            '--target_gop', '0',
        ])

        with self.assertRaisesRegex(ValueError, 'target_gop'):
            _validate_args(args)

    def test_accepts_structured_grid_lora(self):
        args = make_parser().parse_args([
            '-d', 'frames', '--anchor_checkpoint', 'anchor.pth',
            '--grid_adapter', 'structured',
            '--grid_rank', '8', '--grid_alpha', '8',
        ])

        _validate_args(args)
        self.assertEqual(args.grid_adapter, 'structured')


if __name__ == '__main__':
    unittest.main()
