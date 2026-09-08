from collections.abc import Mapping

from torch.utils.data import Dataset

from .partition import FixedGOPPartition


class GOPVideoDataset(Dataset):
    """Add fixed-GOP metadata without changing the wrapped video dataset."""

    def __init__(self, dataset, gop_size):
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a torch Dataset")
        if len(dataset) < 1:
            raise ValueError("dataset must contain at least one frame")

        frame_indices = getattr(dataset, 'frame_indices', None)
        if frame_indices is None:
            frame_indices = tuple(range(len(dataset)))
        else:
            frame_indices = tuple(frame_indices)
        if len(frame_indices) != len(dataset):
            raise ValueError("dataset frame_indices must match its length")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in frame_indices
        ):
            raise ValueError("dataset frame_indices must be non-negative integers")

        total_frames = getattr(dataset, 'total_frames', None)
        if total_frames is None:
            total_frames = max(frame_indices) + 1
        self.dataset = dataset
        self.partition = FixedGOPPartition(total_frames, gop_size)
        if max(frame_indices) >= self.partition.total_frames:
            raise ValueError("frame_indices exceed dataset total_frames")
        self.frame_indices = frame_indices
        self._gop_sample_indices = tuple(
            tuple(
                sample_index
                for sample_index, frame_index in enumerate(frame_indices)
                if self.partition.locate(frame_index).gop_index == gop_index
            )
            for gop_index in range(self.partition.num_gops)
        )

    @property
    def gop_size(self):
        return self.partition.gop_size

    @property
    def num_gops(self):
        return self.partition.num_gops

    @property
    def total_frames(self):
        return self.partition.total_frames

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("wrapped dataset samples must be mappings")

        frame_index = int(sample.get('frame_idx', self.frame_indices[index]))
        if frame_index != self.frame_indices[index]:
            raise ValueError("sample frame_idx does not match dataset frame_indices")
        position = self.partition.locate(frame_index)

        result = dict(sample)
        result.update({
            'gop_idx': position.gop_index,
            'gop_id': position.gop_index,
            'gop_local_idx': position.local_index,
            'gop_local_time': position.local_time,
        })
        return result

    def gop_sample_indices(self, gop_index):
        self.partition.bounds(gop_index)
        return self._gop_sample_indices[gop_index]
