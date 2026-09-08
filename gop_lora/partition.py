from dataclasses import dataclass


def _positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


@dataclass(frozen=True)
class GOPPosition:
    gop_index: int
    local_index: int
    local_time: float


class FixedGOPPartition:
    """Map original video frame indices to fixed-size GOPs."""

    def __init__(self, total_frames, gop_size):
        self.total_frames = _positive_integer('total_frames', total_frames)
        self.gop_size = _positive_integer('gop_size', gop_size)
        self.num_gops = (
            self.total_frames + self.gop_size - 1
        ) // self.gop_size

    def __len__(self):
        return self.num_gops

    def bounds(self, gop_index):
        self._validate_gop_index(gop_index)
        start = gop_index * self.gop_size
        stop = min(start + self.gop_size, self.total_frames)
        return start, stop

    def frame_indices(self, gop_index):
        start, stop = self.bounds(gop_index)
        return tuple(range(start, stop))

    def locate(self, frame_index):
        if (not isinstance(frame_index, int) or isinstance(frame_index, bool) or
                not 0 <= frame_index < self.total_frames):
            raise IndexError("frame_index is outside the video")

        gop_index = frame_index // self.gop_size
        start, stop = self.bounds(gop_index)
        local_index = frame_index - start
        local_time = local_index / max(stop - start - 1, 1)
        return GOPPosition(gop_index, local_index, local_time)

    def _validate_gop_index(self, gop_index):
        if (not isinstance(gop_index, int) or isinstance(gop_index, bool) or
                not 0 <= gop_index < self.num_gops):
            raise IndexError("gop_index is outside the partition")
