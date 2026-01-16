#!/bin/bash
# Example evaluation script for B1K policy evaluation
#
# Usage:
#   ./run_eval.sh --task moving_boxes_to_storage --ckpt-dir /path/to/checkpoint
#
# Required environment variables:
#   OPENPI_CKPT_ROOT: Root directory for checkpoints (default: ./outputs/checkpoints)

set -e

# Parse arguments
TASK_NAME="${TASK_NAME:-moving_boxes_to_storage}"
CKPT_DIR="${CKPT_DIR:-${OPENPI_CKPT_ROOT:-./outputs/checkpoints}/pi05_b1k/latest}"
CONTROL_MODE="${CONTROL_MODE:-receeding_horizon}"
NUM_DIFFUSION_STEPS="${NUM_DIFFUSION_STEPS:-10}"
LOG_DIR="${LOG_DIR:-video_outputs}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate behavior

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export B1K_EVAL_TIME=true

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k policy_dir=${CKPT_DIR}"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=${TASK_NAME} \
    eval_on_train_instances=false \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=${NUM_DIFFUSION_STEPS} \
    control_mode=${CONTROL_MODE} \
    log_path="${LOG_DIR}/${TASK_NAME}_${CONTROL_MODE}"
