#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export CUDA_VISIBLE_DEVICES=0;
export B1K_EVAL_TIME=true;
export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

########################################################
#
# Just use a more recency-biased exp_k_value with same max_predictions to accurately
# measure the impact of exp_k_value. Also using hopefully godly 78k steps checkpoint.
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
    control_mode=temporal_ensemble \
    replan_interval=1 \
    max_predictions=10 \
    exp_k_value=0.05 \
    log_path="${LOG_DIR}/moderately_recency_biased_temporal_ensemble"
