#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

# –––––– Scratchpad ––––––
# POLICY_ARGS="policy=lookup"
# POLICY_ARGS="policy=websocket"

# –––––– Model 1 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi0_b1k_2nd/openpi_0_20251109_050108/33000/ \
    /workspace/openpi/outputs/checkpoints/pi0_b1k_2nd/openpi_0_20251109_050108/33000/

EXP_NAME="pi0_on_3_tasks"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi0_b1k_2nd policy_dir=/workspace/openpi/outputs/checkpoints/pi0_b1k_2nd/openpi_0_20251109_050108/33000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3] \
    use_heavy_robot=true \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=true \
    eval_instance_ids=[0,1,2] \
    log_path="${LOG_DIR}/${EXP_NAME}_put_test" \
    use_heavy_robot=true

# –––––– Model 2 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_2nd/openpi_05_20251109_031930/36000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k_2nd/openpi_05_20251109_031930/36000/

EXP_NAME="pi05_from_scratch_on_3_tasks"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k_2nd policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k_2nd/openpi_05_20251109_031930/36000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3] \
    use_heavy_robot=true \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=true \
    eval_instance_ids=[0,1,2] \
    log_path="${LOG_DIR}/${EXP_NAME}_put_test" \
    use_heavy_robot=true

# –––––– Model 3 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251109_025919/33000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251109_025919/33000/

EXP_NAME="pi05_continued_on_3_tasks"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251109_025919/33000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3] \
    use_heavy_robot=true \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=true \
    eval_instance_ids=[0,1,2] \
    log_path="${LOG_DIR}/${EXP_NAME}_put_test" \
    use_heavy_robot=true
