"""Complete, checkpoint-independent Hybrid Grid model container.

Purpose
-------
This module stores everything required for inference in one ``.hgm`` file:
model configuration, non-Grid parameters, FP32 entropy-model parameters and
the arithmetic-coded Grid payload.  It deliberately does not use pickle or
``torch.save``.  Two-dimensional QAT Linear weights are stored as their real
unsigned integer symbols plus FP32 scales and INT32 zero-points; sensitive
one-dimensional parameters and biases remain FP32.

Container layout
----------------
The outer container is an uncompressed ZIP file with CRC protection.  It
contains ``manifest.json``, tensor byte entries and ``grids.hgb``.  The latter
is the versioned Grid bitstream defined by ``compression.bitstream``.

Current scope
-------------
Only the ``cross_scale_recursive`` entropy model is supported because real
coarse-to-fine decoding requires its explicit decoded-context interface.
The root ``codec.py`` exposes this API on the command line.  Full-video
quality, BPP and speed evaluation remain a separate later stage.
"""

from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import torch

from .codec import decode_recursive_grid_file, encode_recursive_grid_file
from .network_quantization import (
    FakeQuantizedParameter,
    network_qat_state,
)
from .quantization import quantize_grid


_FORMAT = 'hybrid_grid_model_bitstream'
_VERSION = 1
_MANIFEST = 'manifest.json'
_GRID_ENTRY = 'grids.hgb'
_CONFIG_KEYS = (
    'grid_levels',
    'grid_feat_dim',
    'base_resolution',
    'finest_resolution',
    'aspect_ratio',
    'time_scale',
    'pe_freq',
    'hidden_dim',
    'quant_step',
    'quant_steps',
    'entropy_model',
    'network_qat',
    'network_quant_bits',
)
_NUMPY_DTYPES = {
    'float32': np.dtype('<f4'),
    'int32': np.dtype('<i4'),
    'uint8': np.dtype('u1'),
    'uint16': np.dtype('<u2'),
}


@dataclass(frozen=True)
class ModelBitstreamStats:
    total_bits: int
    grid_bits: int
    non_grid_parameter_bits: int
    entropy_parameter_bits: int
    manifest_bits: int
    container_overhead_bits: int


@dataclass(frozen=True)
class LoadedModelBitstream:
    model: torch.nn.Module
    config: Namespace
    grid_symbols: tuple
    stats: ModelBitstreamStats
    timing: 'ModelDecodeTiming'


@dataclass(frozen=True)
class ModelDecodeTiming:
    model_build_seconds: float
    parameter_restore_seconds: float
    grid_decode_seconds: float
    total_seconds: float
    grid_levels: tuple


@dataclass(frozen=True)
class ModelBitstreamInfo:
    architecture: str
    config: dict
    grid_shapes: tuple
    quant_steps: tuple
    estimated_grid_bits: float
    actual_grid_payload_bits: int
    stats: ModelBitstreamStats


def _config_dict(config):
    source = vars(config) if not isinstance(config, dict) else config
    missing = [key for key in _CONFIG_KEYS if key not in source]
    if missing:
        raise ValueError(
            'model configuration is missing: {}'.format(', '.join(missing))
        )
    result = {}
    for key in _CONFIG_KEYS:
        value = source[key]
        if isinstance(value, tuple):
            value = list(value)
        result[key] = value
    if result['entropy_model'] != 'cross_scale_recursive':
        raise ValueError(
            'complete bitstream currently requires cross_scale_recursive'
        )
    return result


def _grid_parameter_ids(model):
    return {
        id(level.grid)
        for level in model.reconstruction_model.grid_encoder.levels
    }


def _quantized_parameters(model):
    """Map original parametrized weight IDs to their fake quantizers."""
    result = {}
    for module in model.modules():
        parametrizations = getattr(module, 'parametrizations', {})
        for name in parametrizations:
            chain = parametrizations[name]
            quantizers = [
                item for item in chain
                if isinstance(item, FakeQuantizedParameter)
            ]
            if quantizers:
                result[id(chain.original)] = quantizers[0]
    return result


def _tensor_bytes(tensor, dtype):
    array = tensor.detach().cpu().numpy().astype(_NUMPY_DTYPES[dtype], copy=False)
    return array.tobytes(order='C')


def _tensor_from_bytes(data, dtype, shape, device):
    array = np.frombuffer(data, dtype=_NUMPY_DTYPES[dtype]).copy()
    expected = int(np.prod(shape, dtype=np.int64))
    if array.size != expected:
        raise ValueError('tensor payload size does not match its shape')
    return torch.from_numpy(array.reshape(shape)).to(device=device)


def _parameter_entries(model):
    grid_ids = _grid_parameter_ids(model)
    entropy_ids = {id(parameter) for parameter in model.entropy_models.parameters()}
    quantized = _quantized_parameters(model)
    entries = []
    payloads = {}

    for index, (name, parameter) in enumerate(model.named_parameters()):
        parameter_id = id(parameter)
        if parameter_id in grid_ids:
            continue
        section = 'entropy' if parameter_id in entropy_ids else 'non_grid'
        prefix = 'tensors/{:04d}'.format(index)

        if parameter_id not in quantized:
            entry_name = prefix + '.f32'
            data = _tensor_bytes(parameter.float(), 'float32')
            payloads[entry_name] = data
            entries.append({
                'name': name,
                'section': section,
                'encoding': 'float32',
                'shape': list(parameter.shape),
                'entry': entry_name,
                'payload_bytes': len(data),
            })
            continue

        quantizer = quantized[parameter_id]
        if not quantizer.enabled or quantizer.observer_enabled:
            raise ValueError(
                'network QAT must be enabled and frozen before encoding'
            )
        bits = quantizer.bits
        storage_dtype = 'uint8' if bits <= 8 else 'uint16'
        qmax = 2 ** bits - 1
        scale = quantizer.scale.detach()
        zero_point = quantizer.zero_point.detach()
        integers = torch.round(
            parameter.detach() / scale + zero_point.to(parameter.dtype)
        ).clamp(0, qmax)

        value_name = prefix + '.q'
        scale_name = prefix + '.scale'
        zero_name = prefix + '.zero'
        value_data = _tensor_bytes(integers, storage_dtype)
        scale_data = _tensor_bytes(scale.float(), 'float32')
        zero_data = _tensor_bytes(zero_point, 'int32')
        payloads[value_name] = value_data
        payloads[scale_name] = scale_data
        payloads[zero_name] = zero_data
        entries.append({
            'name': name,
            'section': section,
            'encoding': storage_dtype,
            'bits': bits,
            'shape': list(parameter.shape),
            'entry': value_name,
            'scale_shape': list(scale.shape),
            'scale_entry': scale_name,
            'zero_point_shape': list(zero_point.shape),
            'zero_point_entry': zero_name,
            'payload_bytes': len(value_data) + len(scale_data) + len(zero_data),
        })
    return entries, payloads


def _archive_stats(path, manifest_size, grid_size, records):
    non_grid_bytes = sum(
        record['payload_bytes'] for record in records
        if record['section'] == 'non_grid'
    )
    entropy_bytes = sum(
        record['payload_bytes'] for record in records
        if record['section'] == 'entropy'
    )
    total_bits = Path(path).stat().st_size * 8
    stored_bits = (
        manifest_size + grid_size + non_grid_bytes + entropy_bytes
    ) * 8
    return ModelBitstreamStats(
        total_bits=total_bits,
        grid_bits=grid_size * 8,
        non_grid_parameter_bits=non_grid_bytes * 8,
        entropy_parameter_bits=entropy_bytes * 8,
        manifest_bits=manifest_size * 8,
        container_overhead_bits=total_bits - stored_bits,
    )


@torch.no_grad()
def write_model_bitstream(path, model, config, cdf_precision=16,
                          progress=None):
    """Encode and store a complete inference model in one ``.hgm`` file."""
    config = _config_dict(config)
    if getattr(model, 'entropy_model_type', None) != 'cross_scale_recursive':
        raise ValueError('model must use cross_scale_recursive entropy')
    qat_prepared = bool(_quantized_parameters(model))
    if bool(config['network_qat']) != qat_prepared:
        raise ValueError('network QAT configuration does not match the model')
    if config['network_qat'] and network_qat_state(model) != 'frozen':
        raise ValueError('network QAT must be frozen before encoding')

    quantized = tuple(
        quantize_grid(level.grid, step, mode='symbols')
        for level, step in zip(
            model.reconstruction_model.grid_encoder.levels,
            model.quant_steps,
        )
    )
    symbols = tuple(item.symbols for item in quantized)
    records, tensor_payloads = _parameter_entries(model)

    with tempfile.TemporaryDirectory() as directory:
        grid_path = Path(directory) / _GRID_ENTRY
        encoded, grid_stats = encode_recursive_grid_file(
            grid_path,
            symbols,
            model.entropy_models[0],
            model.quant_steps,
            cdf_precision,
            progress,
        )
        grid_data = grid_path.read_bytes()

    manifest = {
        'format': _FORMAT,
        'version': _VERSION,
        'architecture': model.architecture,
        'config': config,
        'cdf_precision': cdf_precision,
        'parameters': records,
        'estimated_grid_bits': encoded.estimated_bits,
        'actual_grid_payload_bits': encoded.actual_bits,
        'grid_file_bits': grid_stats.total_bits,
    }
    manifest_data = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')

    path = Path(path)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_MANIFEST, manifest_data)
        for entry_name, data in tensor_payloads.items():
            archive.writestr(entry_name, data)
        archive.writestr(_GRID_ENTRY, grid_data)

    return _archive_stats(path, len(manifest_data), len(grid_data), records)


def _read_manifest(archive):
    try:
        data = archive.read(_MANIFEST)
    except KeyError as error:
        raise ValueError('model bitstream has no manifest') from error
    try:
        manifest = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid model bitstream manifest') from error
    if manifest.get('format') != _FORMAT:
        raise ValueError('invalid model bitstream format')
    if manifest.get('version') != _VERSION:
        raise ValueError('unsupported model bitstream version')
    return manifest, len(data)


def inspect_model_bitstream(path):
    """Read configuration and exact storage sizes without decoding the model."""
    from .bitstream import read_grid_bitstream

    path = Path(path)
    try:
        archive = zipfile.ZipFile(path, 'r')
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError('invalid or corrupted model bitstream') from error
    with archive:
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise ValueError('corrupted model bitstream entry: ' + bad_entry)
        manifest, manifest_size = _read_manifest(archive)
        try:
            grid_data = archive.read(_GRID_ENTRY)
        except KeyError as error:
            raise ValueError('model bitstream has no Grid payload') from error
        records = manifest.get('parameters')
        if not isinstance(records, list):
            raise ValueError('invalid parameter manifest')
        stats = _archive_stats(
            path, manifest_size, len(grid_data), records,
        )

    with tempfile.TemporaryDirectory() as directory:
        grid_path = Path(directory) / _GRID_ENTRY
        grid_path.write_bytes(grid_data)
        grid = read_grid_bitstream(grid_path)
    return ModelBitstreamInfo(
        architecture=manifest.get('architecture'),
        config=manifest.get('config'),
        grid_shapes=tuple(level.shape for level in grid.encoded.levels),
        quant_steps=grid.quant_steps,
        estimated_grid_bits=manifest.get('estimated_grid_bits'),
        actual_grid_payload_bits=manifest.get('actual_grid_payload_bits'),
        stats=stats,
    )


def _restore_parameters(model, archive, records, device):
    parameters = dict(model.named_parameters())
    quantized = _quantized_parameters(model)
    expected_names = {
        name for name, parameter in parameters.items()
        if id(parameter) not in _grid_parameter_ids(model)
    }
    saved_names = {record.get('name') for record in records}
    if saved_names != expected_names:
        raise ValueError('saved parameter names do not match the model')

    for record in records:
        name = record['name']
        parameter = parameters[name]
        shape = tuple(record['shape'])
        if shape != tuple(parameter.shape):
            raise ValueError('saved parameter shape mismatch: ' + name)
        try:
            values = _tensor_from_bytes(
                archive.read(record['entry']),
                record['encoding'],
                shape,
                device,
            )
        except KeyError as error:
            raise ValueError('missing tensor payload: ' + name) from error

        if record['encoding'] == 'float32':
            parameter.copy_(values.to(parameter.dtype))
            continue

        quantizer = quantized.get(id(parameter))
        if quantizer is None or quantizer.bits != record['bits']:
            raise ValueError('QAT configuration mismatch: ' + name)
        try:
            scale = _tensor_from_bytes(
                archive.read(record['scale_entry']),
                'float32',
                tuple(record['scale_shape']),
                device,
            )
            zero_point = _tensor_from_bytes(
                archive.read(record['zero_point_entry']),
                'int32',
                tuple(record['zero_point_shape']),
                device,
            )
        except KeyError as error:
            raise ValueError('missing QAT metadata: ' + name) from error
        quantizer.scale.copy_(scale.to(quantizer.scale.dtype))
        quantizer.zero_point.copy_(zero_point.to(quantizer.zero_point.dtype))
        dequantized = (
            values.to(parameter.dtype) - zero_point.to(parameter.dtype)
        ) * scale.to(parameter.dtype)
        parameter.copy_(dequantized)
        quantizer.configure(True, False)


@torch.no_grad()
def load_model_bitstream(path, device='cpu', progress=None):
    """Restore a complete model without reading its training checkpoint."""
    from train_compression import build_compression_model

    path = Path(path)
    device = torch.device(device)
    total_start = time.perf_counter()
    try:
        archive = zipfile.ZipFile(path, 'r')
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError('invalid or corrupted model bitstream') from error

    with archive:
        try:
            bad_entry = archive.testzip()
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError('corrupted model bitstream entry') from error
        if bad_entry is not None:
            raise ValueError('corrupted model bitstream entry: ' + bad_entry)
        manifest, manifest_size = _read_manifest(archive)
        config = Namespace(**manifest['config'])
        start = time.perf_counter()
        model = build_compression_model(config, device)
        model_build_seconds = time.perf_counter() - start
        if model.architecture != manifest.get('architecture'):
            raise ValueError('model architecture does not match the bitstream')
        start = time.perf_counter()
        _restore_parameters(
            model, archive, manifest['parameters'], device,
        )
        parameter_restore_seconds = time.perf_counter() - start

        try:
            grid_data = archive.read(_GRID_ENTRY)
        except KeyError as error:
            raise ValueError('model bitstream has no Grid payload') from error
        records = manifest['parameters']
        stats = _archive_stats(
            path, manifest_size, len(grid_data), records,
        )

    with tempfile.TemporaryDirectory() as directory:
        grid_path = Path(directory) / _GRID_ENTRY
        grid_path.write_bytes(grid_data)
        level_timings = []
        start = time.perf_counter()
        symbols, grid_bitstream = decode_recursive_grid_file(
            grid_path,
            model.entropy_models[0],
            progress,
            level_timings,
        )
        grid_decode_seconds = time.perf_counter() - start

    if tuple(model.quant_steps) != tuple(grid_bitstream.quant_steps):
        raise ValueError('Grid quantization steps do not match the model config')
    for level, level_symbols, step in zip(
        model.reconstruction_model.grid_encoder.levels,
        symbols,
        model.quant_steps,
    ):
        if tuple(level.grid.shape) != tuple(level_symbols.shape):
            raise ValueError('decoded Grid shape does not match the model')
        level.grid.copy_(
            level_symbols.to(device=device, dtype=level.grid.dtype) * step
        )
    model.eval()
    timing = ModelDecodeTiming(
        model_build_seconds=model_build_seconds,
        parameter_restore_seconds=parameter_restore_seconds,
        grid_decode_seconds=grid_decode_seconds,
        total_seconds=time.perf_counter() - total_start,
        grid_levels=tuple(level_timings),
    )
    return LoadedModelBitstream(model, config, symbols, stats, timing)
