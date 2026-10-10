"""In-memory lossless coding of quantized Grid symbols.

This first-stage codec intentionally does not define an on-disk bitstream.
Shapes, symbol offsets and CDFs are supplied to both encoder and decoder.
"""

from dataclasses import dataclass
import math
import time

import torch

from .arithmetic_coder import ArithmeticDecoder, ArithmeticEncoder


@dataclass(frozen=True)
class EncodedGridLevel:
    payload: bytes
    shape: tuple
    symbol_offset: int
    alphabet_size: int


@dataclass(frozen=True)
class EncodedEntropyGrids:
    levels: tuple
    estimated_bits_by_level: tuple

    @property
    def actual_bits_by_level(self):
        return tuple(len(level.payload) * 8 for level in self.levels)

    @property
    def estimated_bits(self):
        if not self.estimated_bits_by_level:
            return None
        return sum(self.estimated_bits_by_level)

    @property
    def actual_bits(self):
        return sum(self.actual_bits_by_level)


@dataclass(frozen=True)
class GridLevelDecodeTiming:
    level: int
    symbol_count: int
    payload_bits: int
    distribution_seconds: float
    cdf_seconds: float
    arithmetic_seconds: float

    @property
    def total_seconds(self):
        return (
            self.distribution_seconds
            + self.cdf_seconds
            + self.arithmetic_seconds
        )


def _validate_symbols(symbols):
    if not torch.is_tensor(symbols):
        raise TypeError("symbols must be a torch.Tensor")
    if symbols.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("symbols must have an integer dtype")
    if symbols.ndim < 1 or symbols.numel() == 0:
        raise ValueError("symbols must be a non-empty tensor")


def _normalize_cdfs(cdfs, symbols):
    cdfs = torch.as_tensor(cdfs, dtype=torch.int64, device='cpu')
    alphabet = cdfs.shape[-1] - 1 if cdfs.ndim else 0
    if alphabet < 1:
        raise ValueError("cdfs must end with a CDF axis of length at least two")

    value_count = symbols.numel()
    channels = symbols.shape[-1]
    if cdfs.ndim == 1:
        return cdfs.expand(value_count, -1)
    if cdfs.ndim == 2 and cdfs.shape[0] == channels:
        repeats = value_count // channels
        return cdfs.repeat(repeats, 1)
    if cdfs.ndim == 2 and cdfs.shape[0] == value_count:
        return cdfs
    raise ValueError(
        "cdfs must be shared [K], per-channel [C,K], or per-symbol [N,K]"
    )


def encode_grid_level(symbols, cdfs, symbol_offset):
    """Encode one integer Grid tensor and return an in-memory level payload."""
    _validate_symbols(symbols)
    if not isinstance(symbol_offset, int) or isinstance(symbol_offset, bool):
        raise TypeError("symbol_offset must be an integer")

    cpu_symbols = symbols.detach().to(device='cpu', dtype=torch.int64).reshape(-1)
    rows = _normalize_cdfs(cdfs, symbols)
    alphabet_size = rows.shape[-1] - 1
    encoder = ArithmeticEncoder()
    for value, cdf in zip(cpu_symbols.tolist(), rows.tolist()):
        encoder.encode(value - symbol_offset, cdf)
    return EncodedGridLevel(
        payload=encoder.finish(),
        shape=tuple(symbols.shape),
        symbol_offset=symbol_offset,
        alphabet_size=alphabet_size,
    )


def decode_grid_level(encoded, cdfs, dtype=torch.int32):
    """Decode one level using exactly the same CDF sequence as encoding."""
    if not isinstance(encoded, EncodedGridLevel):
        raise TypeError("encoded must be an EncodedGridLevel")
    if dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("dtype must be an integer torch dtype")

    template = torch.empty(encoded.shape, dtype=dtype)
    rows = _normalize_cdfs(cdfs, template)
    decoder = ArithmeticDecoder(encoded.payload)
    values = [
        decoder.decode(cdf) + encoded.symbol_offset
        for cdf in rows.tolist()
    ]
    return torch.tensor(values, dtype=dtype).reshape(encoded.shape)


def encode_grid_symbols(symbols_by_level, cdfs_by_level, symbol_offsets):
    """Encode multiple Grid levels independently in coarse-to-fine order."""
    if not isinstance(symbols_by_level, (tuple, list)) or not symbols_by_level:
        raise ValueError("symbols_by_level must be a non-empty tuple or list")
    if len(cdfs_by_level) != len(symbols_by_level):
        raise ValueError("cdfs_by_level must match symbols_by_level")
    if len(symbol_offsets) != len(symbols_by_level):
        raise ValueError("symbol_offsets must match symbols_by_level")
    return tuple(
        encode_grid_level(symbols, cdfs, offset)
        for symbols, cdfs, offset in zip(
            symbols_by_level, cdfs_by_level, symbol_offsets,
        )
    )


def decode_grid_symbols(encoded_levels, cdfs_by_level, dtype=torch.int32):
    """Decode all levels and preserve their original tensor shapes."""
    if not isinstance(encoded_levels, (tuple, list)) or not encoded_levels:
        raise ValueError("encoded_levels must be a non-empty tuple or list")
    if len(cdfs_by_level) != len(encoded_levels):
        raise ValueError("cdfs_by_level must match encoded_levels")
    return tuple(
        decode_grid_level(encoded, cdfs, dtype=dtype)
        for encoded, cdfs in zip(encoded_levels, cdfs_by_level)
    )


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _validate_recursive_inputs(symbols_by_level, entropy_model):
    if not isinstance(symbols_by_level, (tuple, list)) or not symbols_by_level:
        raise ValueError("symbols_by_level must be a non-empty tuple or list")
    if len(symbols_by_level) != entropy_model.levels:
        raise ValueError("symbols_by_level must match the entropy model levels")
    for symbols in symbols_by_level:
        _validate_symbols(symbols)
        if symbols.ndim != 4 or symbols.shape[-1] != entropy_model.channels:
            raise ValueError("each Grid must have shape [X, Y, T, C]")


def _encode_grid_level_chunks(symbols, cdf_chunks, symbol_offset,
                              alphabet_size):
    values = symbols.detach().to(device='cpu', dtype=torch.int64).reshape(-1)
    encoder = ArithmeticEncoder()
    position = 0
    for chunk in cdf_chunks:
        rows = chunk.tolist()
        chunk_values = values[position:position + len(rows)].tolist()
        if len(chunk_values) != len(rows):
            raise ValueError("CDF rows exceed the Grid symbol count")
        for value, cdf in zip(chunk_values, rows):
            encoder.encode(value - symbol_offset, cdf)
        position += len(rows)
    if position != values.numel():
        raise ValueError("CDF rows do not match the Grid symbol count")
    return EncodedGridLevel(
        encoder.finish(), tuple(symbols.shape), symbol_offset, alphabet_size,
    )


def _decode_grid_level_chunks(encoded, cdf_chunks):
    decoder = ArithmeticDecoder(encoded.payload)
    values = []
    cdf_seconds = 0.0
    arithmetic_seconds = 0.0
    iterator = iter(cdf_chunks)
    while True:
        start = time.perf_counter()
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        cdf_seconds += time.perf_counter() - start
        start = time.perf_counter()
        values.extend(
            decoder.decode(cdf) + encoded.symbol_offset
            for cdf in chunk.tolist()
        )
        arithmetic_seconds += time.perf_counter() - start
    expected = math.prod(encoded.shape)
    if len(values) != expected:
        raise ValueError("CDF rows do not match the encoded Grid shape")
    symbols = torch.tensor(values, dtype=torch.int32).reshape(encoded.shape)
    return symbols, cdf_seconds, arithmetic_seconds


@torch.no_grad()
def encode_recursive_grid_symbols(symbols_by_level, entropy_model, precision=16,
                                  progress=None):
    """Encode Grid symbols using a CumulativeCrossScaleEntropy model."""
    from .entropy_cdf import factorized_cdfs, iter_logistic_cdfs

    _validate_recursive_inputs(symbols_by_level, entropy_model)
    device = _model_device(entropy_model)
    values = tuple(symbols.to(device=device, dtype=torch.float32)
                   for symbols in symbols_by_level)
    likelihoods = entropy_model(values)
    estimated_bits = tuple(
        float((-torch.log2(item)).sum().item()) for item in likelihoods
    )

    encoded_levels = []
    base_min = int(symbols_by_level[0].min().item())
    base_max = int(symbols_by_level[0].max().item())
    base_cdfs = factorized_cdfs(
        entropy_model.base_entropy, base_min, base_max, precision,
    )
    encoded_levels.append(encode_grid_level(
        symbols_by_level[0], base_cdfs, base_min,
    ))
    if progress is not None:
        progress(0, symbols_by_level[0], encoded_levels[0], estimated_bits[0])
    context = entropy_model.initialize_context(values[0])

    for level, symbols in enumerate(symbols_by_level[1:], start=1):
        symbol_min = int(symbols.min().item())
        symbol_max = int(symbols.max().item())
        mean, scale = entropy_model.distribution_parameters(
            level, context, symbols.shape,
        )
        cdf_chunks = iter_logistic_cdfs(
            mean, scale, symbol_min, symbol_max, precision,
        )
        encoded_levels.append(_encode_grid_level_chunks(
            symbols,
            cdf_chunks,
            symbol_min,
            symbol_max - symbol_min + 1,
        ))
        if progress is not None:
            progress(
                level, symbols, encoded_levels[-1], estimated_bits[level],
            )
        context = entropy_model.update_context(level, values[level], context)

    return EncodedEntropyGrids(tuple(encoded_levels), estimated_bits)


@torch.no_grad()
def decode_recursive_grid_symbols(encoded, entropy_model, precision=16,
                                  progress=None, timings=None):
    """Decode recursively, deriving every CDF from previously decoded levels."""
    from .entropy_cdf import factorized_cdfs, iter_logistic_cdfs

    if not isinstance(encoded, EncodedEntropyGrids):
        raise TypeError("encoded must be an EncodedEntropyGrids")
    if len(encoded.levels) != entropy_model.levels:
        raise ValueError("encoded levels must match the entropy model levels")

    device = _model_device(entropy_model)
    base_info = encoded.levels[0]
    base_max = base_info.symbol_offset + base_info.alphabet_size - 1
    _synchronize_device(device)
    start = time.perf_counter()
    base_cdfs = factorized_cdfs(
        entropy_model.base_entropy,
        base_info.symbol_offset,
        base_max,
        precision,
    )
    _synchronize_device(device)
    cdf_seconds = time.perf_counter() - start
    start = time.perf_counter()
    base = decode_grid_level(base_info, base_cdfs)
    arithmetic_seconds = time.perf_counter() - start
    restored = [base]
    item = GridLevelDecodeTiming(
        0,
        base.numel(),
        len(base_info.payload) * 8,
        0.0,
        cdf_seconds,
        arithmetic_seconds,
    )
    if timings is not None:
        timings.append(item)
    if progress is not None:
        progress(item)
    context = entropy_model.initialize_context(
        base.to(device=device, dtype=torch.float32)
    )

    for level, level_info in enumerate(encoded.levels[1:], start=1):
        _synchronize_device(device)
        start = time.perf_counter()
        mean, scale = entropy_model.distribution_parameters(
            level, context, level_info.shape,
        )
        _synchronize_device(device)
        distribution_seconds = time.perf_counter() - start
        symbol_max = level_info.symbol_offset + level_info.alphabet_size - 1
        cdf_chunks = iter_logistic_cdfs(
            mean,
            scale,
            level_info.symbol_offset,
            symbol_max,
            precision,
        )
        symbols, cdf_seconds, arithmetic_seconds = (
            _decode_grid_level_chunks(level_info, cdf_chunks)
        )
        restored.append(symbols)
        item = GridLevelDecodeTiming(
            level,
            symbols.numel(),
            len(level_info.payload) * 8,
            distribution_seconds,
            cdf_seconds,
            arithmetic_seconds,
        )
        if timings is not None:
            timings.append(item)
        if progress is not None:
            progress(item)
        context = entropy_model.update_context(
            level,
            symbols.to(device=device, dtype=torch.float32),
            context,
        )
    return tuple(restored)


def encode_recursive_grid_file(path, symbols_by_level, entropy_model,
                               quant_steps, precision=16, progress=None):
    """Entropy-code Grid symbols and store their payloads in one file."""
    from .bitstream import write_grid_bitstream

    encoded = encode_recursive_grid_symbols(
        symbols_by_level, entropy_model, precision, progress,
    )
    stats = write_grid_bitstream(
        path,
        encoded,
        quant_steps,
        cdf_precision=precision,
        entropy_model='cross_scale_recursive',
    )
    return encoded, stats


def decode_recursive_grid_file(path, entropy_model, progress=None,
                               timings=None):
    """Read a Grid file and reconstruct every integer Grid level."""
    from .bitstream import read_grid_bitstream

    bitstream = read_grid_bitstream(path)
    symbols = decode_recursive_grid_symbols(
        bitstream.encoded,
        entropy_model,
        bitstream.cdf_precision,
        progress,
        timings,
    )
    return symbols, bitstream
