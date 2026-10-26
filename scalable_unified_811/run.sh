#!/usr/bin/env bash
# Usage: bash scalable_unified_811/run.sh GPU DATA_ROOT [EPOCHS=300]
# A separate timestamped log and output namespace; no GPU polling/reservation.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash $0 GPU DATA_ROOT [EPOCHS=300]" >&2
    exit 2
fi
gpu_id=$1
data_root=$2
epochs=${3:-300}
if [[ ! "$gpu_id" =~ ^[0-9]+$ || ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU must be one non-negative index; epochs must be positive." >&2
    exit 2
fi
if [[ ! -d "$data_root" ]]; then
    echo "Dataset directory not found: $data_root" >&2
    exit 2
fi
# Resolve a relative dataset against the caller's directory before changing cwd.
data_root=$(cd "$data_root" && pwd)
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
run_id=$(date +%Y%m%d-%H%M%S)_$$
exp_name="bunny_unified_811_ema_20_30_50_${epochs}epoch"
mkdir -p logs/scalable_unified_811 output/scalable_unified_811
log_file="logs/scalable_unified_811/${exp_name}_${run_id}.log"
echo "Unified 811 | GPU $gpu_id | epochs $epochs | log: $project_root/$log_file"
CUDA_VISIBLE_DEVICES="$gpu_id" "${PYTHON_BIN:-python}" -u \
    -m scalable_unified_811.train \
    -d "$data_root" -b 1 --epochs "$epochs" \
    --lr 5e-3 --weight_decay 0 --warmup 0.2 \
    --grid_levels 10 --grid_feat_dim 6 \
    --base_resolution 20 --finest_resolution 100 --aspect_ratio 16 9 \
    --time_scale 0.1 --pe_freq 14 --hidden_dim 240 \
    --feature_dim 128 --num_blocks 3 --gate_hidden_dim 48 --time_hidden_dim 64 \
    --base_res 720 1280 --fixed_res 720 1280 --frame_interval 1 \
    --level_group_sizes 8 1 1 --width_ratios 0.5 0.75 1.0 \
    --target_shares 0.2 0.3 0.5 --ema_decay 0.99 \
    --loss_type L2 --workers "${WORKERS:-4}" --seed 42 \
    --eval_freq 25 --save_interval 25 --log_interval 50 \
    --exp_name "$exp_name" --out_dir ./output/scalable_unified_811 \
    > "$log_file" 2>&1
echo "Unified 811 completed. See $project_root/$log_file"
