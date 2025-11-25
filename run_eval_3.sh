#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

# Trying k = 0.05 again but with temporal ensemble here
export CUDA_VISIBLE_DEVICES=2;
export B1K_EVAL_TIME=true;
# export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

export TRAIN_CONFIG_NAME="pi05_b1k_oversample_mbts";
export CKPT_NAME="openpi_05_20251115_045832";
export STEP_COUNT=36000;
export TASK_NAME="moving_boxes_to_storage";

# export CONTROL_MODE="receeding_temporal";
# export MAX_LEN=100;
# export ACTION_HORIZON=20;
# export TEMPORAL_ENSEMBLE_MAX=5;
# export EXP_K_VALUE=0.2;

export CONTROL_MODE="temporal_ensemble";
export MAX_LEN=50;
export ACTION_HORIZON=50;
export TEMPORAL_ENSEMBLE_MAX=50;
export EXP_K_VALUE=0.05;

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/ \
    /workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/

export EXP_NAME="${TASK_NAME}";
export LOG_DIR="mbts_control_modes/${EXP_NAME}";

mkdir -p "${LOG_DIR}";

export POLICY_ARGS="policy=local policy_config=pi05_b1k_inference_final policy_dir=/workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}";

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name="${TASK_NAME}" \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3,4,5,6,7,8,9] \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=10 \
    control_mode=${CONTROL_MODE} \
    action_horizon=${ACTION_HORIZON} \
    max_len=${MAX_LEN} \
    temporal_ensemble_max=${TEMPORAL_ENSEMBLE_MAX} \
    exp_k_value=${EXP_K_VALUE} \
    log_path="${LOG_DIR}/k_${EXP_K_VALUE}"
