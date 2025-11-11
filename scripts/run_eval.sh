#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export B1K_EVAL_TIME=true

# –––––– Scratchpad ––––––
# POLICY_ARGS="policy=lookup"
# POLICY_ARGS="policy=websocket"

# –––––– Model 1 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_single_task_w_us/openpi_05_20251110_180314/21000/ \
    /workspace/openpi/outputs/checkpoints/pi05_single_task_w_us/openpi_05_20251110_180314/21000/

EXP_NAME="pi05_single_task_w_us_FROM_SCRATCH_mbts_256_ah"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_single_task_w_us policy_dir=/workspace/openpi/outputs/checkpoints/pi05_single_task_w_us/openpi_05_20251110_180314/21000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=true \
    eval_instance_ids=[1,2] \
    use_heavy_robot=true \
    max_steps=9000 \
    n_ds_steps=2500 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_mbts_train_0.2_proprio_dropout_again_2500_ds"

# –––––– Model 2 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_single_task_picking_up_trash/openpi_05_20251109_221633/42000/ \
    /workspace/openpi/outputs/checkpoints/pi05_single_task_picking_up_trash/openpi_05_20251109_221633/42000/

EXP_NAME="pi05_single_task_picking_up_trash_256_ah"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_single_task_picking_up_trash policy_dir=/workspace/openpi/outputs/checkpoints/pi05_single_task_picking_up_trash/openpi_05_20251109_221633/42000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=false \
    eval_instance_ids=[1,2] \
    use_heavy_robot=true \
    n_ds_steps=0 \
    max_steps=9000 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_putrash_test_0.2_proprio_dropout"

# –––––– Model 3 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi0_b1k_single_task/openpi_0_20251110_170500/33000/ \
    /workspace/openpi/outputs/checkpoints/pi0_b1k_single_task/openpi_0_20251110_170500/33000/

EXP_NAME="pi0_b1k_single_task_mbts_256_ah"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi0_b1k_single_task policy_dir=/workspace/openpi/outputs/checkpoints/pi0_b1k_single_task/openpi_0_20251110_170500/33000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=true \
    eval_instance_ids=[1,2] \
    use_heavy_robot=true \
    max_steps=7000 \
    n_ds_steps=2500 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_mbts_train_0.2_proprio_dropout_again_2500_ds"

# –––––– Model 4 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251109_182322/45000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251109_182322/45000/

EXP_NAME="pi05_256_ah_3_tasks"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_single_task_w_us policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251109_182322/45000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[1,2] \
    use_heavy_robot=true \
    n_ds_steps=0 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test_0.2_proprio_dropout"

# –––––– Model 5 – OG ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/ \
    /workspace/openpi/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/

EXP_NAME="pi05_single_task_mbts_256_ah"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_single_task_w_us policy_dir=/workspace/openpi/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=false \
    eval_instance_ids=[0] \
    use_heavy_robot=true \
    max_steps=8000 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_picking_up_trash_test_0.2_proprio_dropout"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0] \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_turning_on_radio_test_0.2_proprio_dropout"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=false \
    eval_instance_ids=[4,5,6,7] \
    use_heavy_robot=true \
    max_steps=22000 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_mbts_test_0.2_proprio_dropout_22k_steps"
