#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export CUDA_VISIBLE_DEVICES=0;
export B1K_EVAL_TIME=true;
# export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

########################################################
#
# Experiment group 2, iteration 0
# - control_mode=receeding_temporal
# - action_horizon=50
# - max_len=250
# - temporal_ensemble_max=5
# - exp_k_value=0.9  # THE THING BEING TESTED HERE
#
########################################################

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/36000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/36000/

EXP_NAME="openpi_05_20251115_045832_36k_steps"
LOG_DIR="video_outputs_lium-ada-v1/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k_oversample_mbts policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k_oversample_mbts/openpi_05_20251115_045832/36000"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=false \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=10 \
    max_steps=18000 \
    control_mode=receeding_temporal \
    action_horizon=50 \
    max_len=250 \
    temporal_ensemble_max=5 \
    exp_k_value=0.9 \
    log_path="${LOG_DIR}/exp_group_2_k_0.9"
