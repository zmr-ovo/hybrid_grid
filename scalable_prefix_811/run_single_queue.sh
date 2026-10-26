#!/usr/bin/env bash
# Run two finite, useful capacity controls on one explicitly selected GPU.
# Usage: bash scalable_prefix_811/run_single_queue.sh GPU DATA_ROOT [EPOCHS]
# Requires the intended Python environment to be activated beforehand.
# No polling, GPU selection, memory reservation, or interaction with other jobs.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash $0 GPU DATA_ROOT [EPOCHS=300]" >&2
    exit 2
fi
gpu_id=$1
data_root=$2
epochs=${3:-300}
if [[ ! "$gpu_id" =~ ^[0-9]+$ || ! "$epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU must be a single non-negative index; EPOCHS must be positive." >&2
    exit 2
fi
if [[ ! -d "$data_root" ]]; then
    echo "Dataset directory not found: $data_root" >&2
    exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
run_id=$(date +%Y%m%d-%H%M%S)_$$
log_dir="logs/prefix_811_single/$run_id"
out_dir="output/scalable_prefix_811_single"
mkdir -p "$log_dir" "$out_dir"
python_bin=${PYTHON_BIN:-python}
common=(
    -d "$data_root" -b 1 --epochs "$epochs"
    --lr 5e-3 --weight_decay 0 --warmup 0.2
    --grid_levels 10 --grid_feat_dim 6
    --base_resolution 20 --finest_resolution 100
    --aspect_ratio 16 9 --time_scale 0.1 --pe_freq 14 --hidden_dim 256
    --base_res 720 1280 --fixed_res 720 1280 --frame_interval 1
    --level_group_sizes 8 1 1 --width_ratios 0.5 0.75 1.0
    --loss_type L2 --workers "${WORKERS:-4}" --seed 42
    --eval_freq 25 --save_interval 25 --log_interval 50
)

for quality in base mid; do
    log_file="$log_dir/${quality}_only.log"
    echo "$(date) | START ${quality}-only | GPU $gpu_id | log: $log_file"
    if CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u \
        -m "scalable_prefix_811.train_${quality}_only" "${common[@]}" \
        --out_dir "$out_dir/${quality}_only" \
        --exp_name "bunny_811_${quality}_only_${epochs}epoch_$run_id" \
        > "$log_file" 2>&1; then
        echo "$(date) | DONE ${quality}-only"
    else
        status=$?
        echo "$(date) | FAILED ${quality}-only (exit $status); queue stopped. See $log_file" >&2
        exit "$status"
    fi
done
echo "$(date) | Both single-prefix experiments completed."
