"""Analyze gradient conflicts among Base, Mid and Full prefixes.

This diagnostic loads a trained ``PrefixScalableResNet`` checkpoint and
computes three independent L2 gradients on the same video frames. Results are
reported for Grid groups, position encoding, gates, temporal modulation, and
disjoint Base/Mid/Full slices of the nested Decoder parameters.

The script is read-only: it never calls an optimizer or modifies a checkpoint.
"""

import argparse
import json
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

from model import DynamicVideoDataset
from scalable_module_resnet.train import hierarchy_from_sizes
from scalable_prefix_resnet.model import PrefixScalableResNet
from train import seed_everything


PREFIX_NAMES = ("Base", "Mid", "Full")
PAIR_NAMES = ((0, 1), (0, 2), (1, 2))
GRID_PATTERN = re.compile(r"^grid_encoder\.levels\.(\d+)\.")


def _decoder_active_mask(name, parameter, width, model):
    """Return elements used by one nested Decoder width."""
    mask = torch.zeros(parameter.shape, dtype=torch.bool)
    input_dim = model.decoder.input_dim

    if name == "decoder.linear.0.weight":
        mask[:width, :input_dim] = True
    elif name == "decoder.linear.0.bias":
        mask[:width] = True
    elif name in ("decoder.linear.1.weight", "decoder.linear.2.weight"):
        mask[:width, :width] = True
    elif name in ("decoder.linear.1.bias", "decoder.linear.2.bias"):
        mask[:width] = True
    elif name == "decoder.linear.3.weight":
        mask[:, :width] = True
    elif name == "decoder.linear.3.bias":
        mask[:] = True
    elif name == "decoder.skip_proj.weight":
        mask[:width, :input_dim] = True
    elif name in (
        "decoder.output_layer.0.weight",
        "decoder.output_layer.0.bias",
    ):
        mask[:] = True
    else:
        raise ValueError("unrecognized Decoder parameter: " + name)
    return mask


def build_parameter_groups(model):
    """Build disjoint parameter-element groups used in conflict reports."""
    groups = defaultdict(list)
    widths = model.decoder.hidden_widths
    active_decoder_masks = {}
    named_parameters = list(model.named_parameters())

    for parameter_index, (name, parameter) in enumerate(named_parameters):
        if name.startswith("decoder."):
            masks = tuple(
                _decoder_active_mask(name, parameter, width, model)
                for width in widths
            )
            active_decoder_masks[name] = masks
            disjoint_masks = (
                masks[0],
                masks[1] & ~masks[0],
                masks[2] & ~masks[1],
            )
            for group_name, mask in zip(
                ("decoder_base_shared", "decoder_mid_increment", "decoder_full_increment"),
                disjoint_masks,
            ):
                if mask.any():
                    groups[group_name].append((parameter_index, mask))
            continue

        match = GRID_PATTERN.match(name)
        if match:
            level = int(match.group(1))
            if level in model.hierarchy.groups[0]:
                group_name = "grid_base_L0_L6"
            elif level in model.hierarchy.groups[1]:
                group_name = "grid_mid_L7_L8"
            else:
                group_name = "grid_full_L9"
            groups[group_name].append((parameter_index, None))
        elif name.startswith("gate_grid.") or name.startswith("gate_pe."):
            groups["gate_shared"].append((parameter_index, None))
        elif name.startswith("time_mod."):
            groups["time_modulation_shared"].append((parameter_index, None))
        elif name.startswith("pe_encoder."):
            groups["position_encoding_shared"].append((parameter_index, None))
        else:
            groups["other_shared"].append((parameter_index, None))

    return named_parameters, dict(groups)


def _group_vector(grads, parameters, entries):
    parts = []
    for parameter_index, mask in entries:
        gradient = grads[parameter_index]
        parameter = parameters[parameter_index]
        if gradient is None:
            size = parameter.numel() if mask is None else int(mask.sum().item())
            parts.append(torch.zeros(size, dtype=torch.float64))
            continue
        values = gradient.detach().to(device="cpu", dtype=torch.float64)
        parts.append(values.reshape(-1) if mask is None else values[mask])
    if not parts:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(parts)


def extract_group_gradients(model, coords, target, named_parameters, groups):
    """Compute unweighted L2 gradients for all three quality levels."""
    parameters = [parameter for _, parameter in named_parameters]
    prefix_vectors = []
    losses = []
    for quality_level in range(3):
        reconstruction = model.forward_prefix(coords, quality_level).reconstruction
        loss = F.mse_loss(reconstruction, target)
        grads = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        vectors = {
            name: _group_vector(grads, parameters, entries)
            for name, entries in groups.items()
        }
        shared_names = (
            "grid_base_L0_L6",
            "position_encoding_shared",
            "gate_shared",
            "time_modulation_shared",
            "decoder_base_shared",
            "other_shared",
        )
        vectors["shared_core_total"] = torch.cat([
            vectors[name] for name in shared_names if name in vectors
        ])
        vectors["all_parameters"] = torch.cat([
            vectors[name] for name in sorted(vectors)
            if name not in ("shared_core_total", "all_parameters")
        ])
        prefix_vectors.append(vectors)
        losses.append(float(loss.detach().item()))
        del reconstruction, loss, grads
    return tuple(prefix_vectors), tuple(losses)


def vector_pair_statistics(first, second, epsilon=1e-18):
    first_norm = float(torch.linalg.vector_norm(first).item())
    second_norm = float(torch.linalg.vector_norm(second).item())
    if first_norm <= epsilon or second_norm <= epsilon:
        return {
            "cosine": None,
            "opposed_element_ratio": None,
            "joint_nonzero_elements": 0,
        }
    cosine = float(torch.dot(first, second).item() / (first_norm * second_norm))
    jointly_active = (first.abs() > epsilon) & (second.abs() > epsilon)
    jointly_active_count = int(jointly_active.sum().item())
    opposed_ratio = None
    if jointly_active_count:
        opposed = ((first * second) < 0) & jointly_active
        opposed_ratio = float(opposed.sum().item() / jointly_active_count)
    return {
        "cosine": max(-1.0, min(1.0, cosine)),
        "opposed_element_ratio": opposed_ratio,
        "joint_nonzero_elements": jointly_active_count,
    }


class ConflictAccumulator:
    def __init__(self, group_names):
        self.frames = 0
        self.norms = {
            name: [[0.0, 0] for _ in PREFIX_NAMES] for name in group_names
        }
        self.pairs = {
            name: {
                "{}-{}".format(PREFIX_NAMES[a], PREFIX_NAMES[b]): {
                    "cosine_sum": 0.0,
                    "cosine_count": 0,
                    "negative_count": 0,
                    "opposed_sum": 0.0,
                    "opposed_count": 0,
                }
                for a, b in PAIR_NAMES
            }
            for name in group_names
        }

    def update(self, prefix_vectors):
        self.frames += 1
        for group_name in self.norms:
            vectors = [item[group_name] for item in prefix_vectors]
            for quality_level, vector in enumerate(vectors):
                norm = float(torch.linalg.vector_norm(vector).item())
                self.norms[group_name][quality_level][0] += norm
                self.norms[group_name][quality_level][1] += 1
            for first, second in PAIR_NAMES:
                pair_name = "{}-{}".format(
                    PREFIX_NAMES[first], PREFIX_NAMES[second],
                )
                stats = vector_pair_statistics(vectors[first], vectors[second])
                target = self.pairs[group_name][pair_name]
                if stats["cosine"] is not None:
                    target["cosine_sum"] += stats["cosine"]
                    target["cosine_count"] += 1
                    target["negative_count"] += int(stats["cosine"] < 0)
                if stats["opposed_element_ratio"] is not None:
                    target["opposed_sum"] += stats["opposed_element_ratio"]
                    target["opposed_count"] += 1

    def result(self, parameter_counts):
        result = {}
        for group_name, norms in self.norms.items():
            group_result = {
                "parameter_elements": parameter_counts[group_name],
                "mean_gradient_norm": {
                    name: total / max(count, 1)
                    for name, (total, count) in zip(PREFIX_NAMES, norms)
                },
                "pairs": {},
            }
            for pair_name, values in self.pairs[group_name].items():
                cosine_count = values["cosine_count"]
                opposed_count = values["opposed_count"]
                group_result["pairs"][pair_name] = {
                    "mean_cosine": (
                        values["cosine_sum"] / cosine_count
                        if cosine_count else None
                    ),
                    "negative_frame_ratio": (
                        values["negative_count"] / cosine_count
                        if cosine_count else None
                    ),
                    "mean_opposed_element_ratio": (
                        values["opposed_sum"] / opposed_count
                        if opposed_count else None
                    ),
                    "valid_frames": cosine_count,
                }
            result[group_name] = group_result
        return result


def _parameter_counts(parameters, groups):
    counts = {}
    for name, entries in groups.items():
        counts[name] = sum(
            parameters[index].numel() if mask is None else int(mask.sum().item())
            for index, mask in entries
        )
    counts["shared_core_total"] = sum(
        counts.get(name, 0)
        for name in (
            "grid_base_L0_L6",
            "position_encoding_shared",
            "gate_shared",
            "time_modulation_shared",
            "decoder_base_shared",
            "other_shared",
        )
    )
    counts["all_parameters"] = sum(counts.values()) - counts["shared_core_total"]
    return counts


def _sample_indices(length, requested):
    count = min(int(requested), length)
    if count < 1:
        raise ValueError("num_frames must be positive")
    if count == 1:
        return (length // 2,)
    return tuple(sorted(set(
        int(round(index * (length - 1) / (count - 1)))
        for index in range(count)
    )))


def _make_model(checkpoint, device):
    config = checkpoint.get("config", {})
    grid_levels = int(config.get("grid_levels", 10))
    hierarchy = hierarchy_from_sizes(
        config.get("level_group_sizes", (7, 2, 1)), grid_levels,
    )
    model = PrefixScalableResNet(
        grid_levels=grid_levels,
        grid_feat_dim=int(config.get("grid_feat_dim", 6)),
        base_resolution=int(config.get("base_resolution", 20)),
        finest_resolution=int(config.get("finest_resolution", 100)),
        aspect_ratio=tuple(config.get("aspect_ratio", (16, 9))),
        time_scale=float(config.get("time_scale", 0.1)),
        pe_freq=int(config.get("pe_freq", 14)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        width_ratios=tuple(config.get("width_ratios", (0.5, 0.75, 1.0))),
        hierarchy=hierarchy,
    ).to(device)
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError(
            "only PrefixScalableResNet checkpoints are supported; got "
            + str(checkpoint.get("architecture"))
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config


def _format_optional(value, precision=4):
    return "N/A" if value is None else ("{:.%df}" % precision).format(value)


def _write_text_report(path, report):
    lines = [
        "========== Three-Prefix Gradient Conflict Analysis ==========",
        "Checkpoint: " + report["checkpoint"],
        "Checkpoint epoch: " + str(report["checkpoint_epoch"]),
        "Frames: " + str(report["frame_indices"]),
        "Resolution: {} x {}".format(*report["resolution"]),
        "Mean distortion Base/Mid/Full: " + " / ".join(
            "{:.7f}".format(value) for value in report["mean_distortion"]
        ),
        "",
    ]
    for group_name, values in report["groups"].items():
        norms = values["mean_gradient_norm"]
        lines.append(
            "[{}] elements: {} | norm B/M/F: {:.4e} / {:.4e} / {:.4e}".format(
                group_name,
                values["parameter_elements"],
                norms["Base"],
                norms["Mid"],
                norms["Full"],
            )
        )
        for pair_name, pair in values["pairs"].items():
            lines.append(
                "  {} | cosine: {} | negative frames: {} | "
                "opposed elements: {} | valid frames: {}".format(
                    pair_name,
                    _format_optional(pair["mean_cosine"]),
                    _format_optional(pair["negative_frame_ratio"]),
                    _format_optional(pair["mean_opposed_element_ratio"]),
                    pair["valid_frames"],
                )
            )
        lines.append("")
    lines.extend([
        "Interpretation:",
        "  cosine > 0: the two objectives are aligned on average.",
        "  cosine < 0: the two objectives request opposing parameter updates.",
        "  negative frames: fraction of analyzed frames with cosine < 0.",
        "  opposed elements: fraction of jointly non-zero gradient elements with opposite signs.",
        "  N/A: at least one prefix does not use this parameter group.",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(args):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError("checkpoint not found: " + args.checkpoint)
    if not os.path.isdir(args.data_root):
        raise ValueError("data_root is not a directory: " + args.data_root)
    seed_everything(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, config = _make_model(checkpoint, device)

    fixed_res = tuple(args.fixed_res or config.get("fixed_res", (720, 1280)))
    base_res = tuple(config.get("base_res", fixed_res))
    dataset = DynamicVideoDataset(
        args.data_root,
        base_res=base_res,
        fixed_res=fixed_res,
        frame_interval=int(config.get("frame_interval", 1)),
    )
    indices = _sample_indices(len(dataset), args.num_frames)
    named_parameters, groups = build_parameter_groups(model)
    parameters = [parameter for _, parameter in named_parameters]
    parameter_counts = _parameter_counts(parameters, groups)
    analysis_group_names = tuple(groups) + ("shared_core_total", "all_parameters")
    accumulator = ConflictAccumulator(analysis_group_names)
    distortion_sums = [0.0, 0.0, 0.0]

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Frames:", indices)
    print("Resolution:", fixed_res)
    for position, frame_index in enumerate(indices, start=1):
        item = dataset[frame_index]
        coords = item["coords"].unsqueeze(0).to(device)
        target = item["pixels"].unsqueeze(0).to(device)
        prefix_vectors, losses = extract_group_gradients(
            model, coords, target, named_parameters, groups,
        )
        accumulator.update(prefix_vectors)
        distortion_sums = [
            total + value for total, value in zip(distortion_sums, losses)
        ]
        current_bf = vector_pair_statistics(
            prefix_vectors[0]["shared_core_total"],
            prefix_vectors[2]["shared_core_total"],
        )["cosine"]
        print(
            "Frame [{}/{}] source index {} | D B/M/F: {} | shared B-F cosine: {}".format(
                position,
                len(indices),
                frame_index,
                " / ".join("{:.7f}".format(value) for value in losses),
                _format_optional(current_bf),
            )
        )
        del coords, target, prefix_vectors

    mean_distortion = tuple(value / len(indices) for value in distortion_sums)
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "architecture": checkpoint.get("architecture"),
        "training_strategy": checkpoint.get("training_strategy"),
        "frame_indices": indices,
        "resolution": fixed_res,
        "mean_distortion": mean_distortion,
        "groups": accumulator.result(parameter_counts),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    text_path = os.path.join(args.output_dir, args.name + ".txt")
    json_path = os.path.join(args.output_dir, args.name + ".json")
    _write_text_report(text_path, report)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("Saved text report:", os.path.abspath(text_path))
    print("Saved JSON report:", os.path.abspath(json_path))


def build_parser():
    parser = argparse.ArgumentParser(
        description="分析Base、Mid和Full在共享参数上的梯度冲突",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-d", "--data_root", required=True)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--fixed_res", type=int, nargs=2, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", default="./output/prefix_gradient_analysis",
    )
    parser.add_argument("--name", default="prefix_gradient_conflicts")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

