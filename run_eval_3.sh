#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export CUDA_VISIBLE_DEVICES=2;
export B1K_EVAL_TIME=true;
export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

########################################################
#
# Receeding_temporal but close to current best. Also using hopefully godly 78k steps checkpoint.
#
########################################################

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/78000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/78000/

EXP_NAME="openpi_05_20251115_045832_78k_steps"
LOG_DIR="video_outputs_new/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k_22_TASKS_oversample policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/78000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=false \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=10 \
    max_steps=15000 \
    control_mode=receeding_temporal \
    replan_interval=10 \
    max_predictions=5 \
    exp_k_value=0.05 \
    log_path="${LOG_DIR}/receeding_temporal_close_to_current_best"
