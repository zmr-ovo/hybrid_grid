import unittest

from gop_lora.partition import FixedGOPPartition


class FixedGOPPartitionTest(unittest.TestCase):
    def test_even_partition_has_expected_boundaries(self):
        partition = FixedGOPPartition(total_frames=120, gop_size=30)

        self.assertEqual(len(partition), 4)
        self.assertEqual(partition.bounds(0), (0, 30))
        self.assertEqual(partition.bounds(3), (90, 120))
        self.assertEqual(partition.locate(30).gop_index, 1)

    def test_last_gop_may_be_shorter(self):
        partition = FixedGOPPartition(total_frames=132, gop_size=30)

        self.assertEqual(len(partition), 5)
        self.assertEqual(partition.bounds(4), (120, 132))
        self.assertEqual(len(partition.frame_indices(4)), 12)
        self.assertEqual(partition.locate(120).local_time, 0.0)
        self.assertEqual(partition.locate(131).local_time, 1.0)

    def test_each_frame_maps_to_exactly_one_gop(self):
        partition = FixedGOPPartition(total_frames=17, gop_size=5)
        frames = [
            frame
            for gop_index in range(len(partition))
            for frame in partition.frame_indices(gop_index)
        ]

        self.assertEqual(frames, list(range(17)))
        for frame in frames:
            position = partition.locate(frame)
            self.assertIn(frame, partition.frame_indices(position.gop_index))

    def test_single_frame_gop_has_zero_local_time(self):
        partition = FixedGOPPartition(total_frames=5, gop_size=2)

        self.assertEqual(partition.bounds(2), (4, 5))
        self.assertEqual(partition.locate(4).local_time, 0.0)

    def test_rejects_invalid_inputs_and_indices(self):
        for total_frames, gop_size in ((0, 1), (1, 0), (True, 1)):
            with self.subTest(total_frames=total_frames, gop_size=gop_size):
                with self.assertRaises(ValueError):
                    FixedGOPPartition(total_frames, gop_size)

        partition = FixedGOPPartition(5, 2)
        for frame_index in (-1, 5):
            with self.assertRaises(IndexError):
                partition.locate(frame_index)
        for gop_index in (-1, 3):
            with self.assertRaises(IndexError):
                partition.bounds(gop_index)


if __name__ == '__main__':
    unittest.main()
