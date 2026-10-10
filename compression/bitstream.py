"""Binary container for independently coded quantized Grid levels."""

from dataclasses import dataclass
import math
from pathlib import Path
import struct
import zlib

from .codec import EncodedEntropyGrids, EncodedGridLevel


_MAGIC = b'HGRIDBS\x00'
_VERSION = 1
_HEADER = struct.Struct('<8sHBBH')
_LEVEL_HEADER = struct.Struct('<H4IdqIQI')
_ENTROPY_MODEL_IDS = {'cross_scale_recursive': 1}
_ENTROPY_MODEL_NAMES = {
    identifier: name for name, identifier in _ENTROPY_MODEL_IDS.items()
}


@dataclass(frozen=True)
class GridBitstreamStats:
    header_bits: int
    level_metadata_bits: int
    payload_bits: int
    total_bits: int


@dataclass(frozen=True)
class GridBitstream:
    encoded: EncodedEntropyGrids
    quant_steps: tuple
    cdf_precision: int
    entropy_model: str
    stats: GridBitstreamStats


def _read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated Grid bitstream")
    return data


def _validate(encoded, quant_steps, cdf_precision, entropy_model):
    if not isinstance(encoded, EncodedEntropyGrids) or not encoded.levels:
        raise TypeError("encoded must be a non-empty EncodedEntropyGrids")
    if len(quant_steps) != len(encoded.levels):
        raise ValueError("quant_steps must match the encoded Grid levels")
    normalized_steps = tuple(float(step) for step in quant_steps)
    if any(not math.isfinite(step) or step <= 0 for step in normalized_steps):
        raise ValueError("quant_steps must be finite and positive")
    if (not isinstance(cdf_precision, int) or isinstance(cdf_precision, bool) or
            cdf_precision < 1 or cdf_precision > 24):
        raise ValueError("cdf_precision must be an integer between 1 and 24")
    if entropy_model not in _ENTROPY_MODEL_IDS:
        raise ValueError("unsupported entropy model")

    for index, level in enumerate(encoded.levels):
        if not isinstance(level, EncodedGridLevel):
            raise TypeError("each encoded level must be an EncodedGridLevel")
        if len(level.shape) != 4 or any(
            not isinstance(size, int) or size < 1 for size in level.shape
        ):
            raise ValueError("each Grid shape must contain four positive sizes")
        if level.alphabet_size < 1:
            raise ValueError("alphabet_size must be positive")
        if not isinstance(level.payload, bytes) or not level.payload:
            raise ValueError("each level payload must be non-empty bytes")
        if index > 0 and level.shape[-1] != encoded.levels[0].shape[-1]:
            raise ValueError("all Grid levels must have the same channel count")
    return normalized_steps


def _stats(level_count, payload_bytes):
    header_bits = _HEADER.size * 8
    level_metadata_bits = level_count * _LEVEL_HEADER.size * 8
    payload_bits = payload_bytes * 8
    return GridBitstreamStats(
        header_bits=header_bits,
        level_metadata_bits=level_metadata_bits,
        payload_bits=payload_bits,
        total_bits=header_bits + level_metadata_bits + payload_bits,
    )


def write_grid_bitstream(path, encoded, quant_steps, cdf_precision=16,
                         entropy_model='cross_scale_recursive'):
    """Write a versioned Grid bitstream and return its exact size breakdown."""
    quant_steps = _validate(
        encoded, quant_steps, cdf_precision, entropy_model,
    )
    path = Path(path)
    entropy_model_id = _ENTROPY_MODEL_IDS[entropy_model]

    with path.open('wb') as file:
        file.write(_HEADER.pack(
            _MAGIC,
            _VERSION,
            cdf_precision,
            entropy_model_id,
            len(encoded.levels),
        ))
        for index, (level, quant_step) in enumerate(
            zip(encoded.levels, quant_steps)
        ):
            checksum = zlib.crc32(level.payload) & 0xffffffff
            file.write(_LEVEL_HEADER.pack(
                index,
                *level.shape,
                quant_step,
                level.symbol_offset,
                level.alphabet_size,
                len(level.payload),
                checksum,
            ))
        for level in encoded.levels:
            file.write(level.payload)

    stats = _stats(
        len(encoded.levels),
        sum(len(level.payload) for level in encoded.levels),
    )
    if path.stat().st_size * 8 != stats.total_bits:
        raise RuntimeError("written Grid bitstream size does not match metadata")
    return stats


def read_grid_bitstream(path):
    """Read and validate a Grid bitstream without requiring a checkpoint."""
    path = Path(path)
    with path.open('rb') as file:
        magic, version, precision, model_id, level_count = _HEADER.unpack(
            _read_exact(file, _HEADER.size)
        )
        if magic != _MAGIC:
            raise ValueError("invalid Grid bitstream magic")
        if version != _VERSION:
            raise ValueError("unsupported Grid bitstream version")
        if precision < 1 or precision > 24:
            raise ValueError("invalid CDF precision")
        if model_id not in _ENTROPY_MODEL_NAMES:
            raise ValueError("unsupported entropy model identifier")
        if level_count < 1:
            raise ValueError("Grid bitstream contains no levels")

        metadata = []
        for expected_index in range(level_count):
            values = _LEVEL_HEADER.unpack(
                _read_exact(file, _LEVEL_HEADER.size)
            )
            index = values[0]
            if index != expected_index:
                raise ValueError("Grid levels are not stored in order")
            shape = tuple(values[1:5])
            quant_step = values[5]
            symbol_offset = values[6]
            alphabet_size = values[7]
            payload_size = values[8]
            checksum = values[9]
            if any(size < 1 for size in shape):
                raise ValueError("invalid Grid shape")
            if not math.isfinite(quant_step) or quant_step <= 0:
                raise ValueError("invalid quantization step")
            if alphabet_size < 1 or payload_size < 1:
                raise ValueError("invalid Grid level metadata")
            metadata.append((
                shape,
                quant_step,
                symbol_offset,
                alphabet_size,
                payload_size,
                checksum,
            ))

        levels = []
        quant_steps = []
        payload_bytes = 0
        for (shape, quant_step, symbol_offset, alphabet_size,
             payload_size, checksum) in metadata:
            payload = _read_exact(file, payload_size)
            if zlib.crc32(payload) & 0xffffffff != checksum:
                raise ValueError("Grid payload checksum mismatch")
            levels.append(EncodedGridLevel(
                payload=payload,
                shape=shape,
                symbol_offset=symbol_offset,
                alphabet_size=alphabet_size,
            ))
            quant_steps.append(quant_step)
            payload_bytes += payload_size

        if file.read(1):
            raise ValueError("unexpected trailing data in Grid bitstream")

    stats = _stats(level_count, payload_bytes)
    if path.stat().st_size * 8 != stats.total_bits:
        raise ValueError("Grid bitstream size does not match its metadata")
    return GridBitstream(
        encoded=EncodedEntropyGrids(tuple(levels), ()),
        quant_steps=tuple(quant_steps),
        cdf_precision=precision,
        entropy_model=_ENTROPY_MODEL_NAMES[model_id],
        stats=stats,
    )
