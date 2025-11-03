#!/bin/bash

EXP_NAME="25k_ds_filtering_again_special_prompts"
LOG_DIR="video_outputs/${EXP_NAME}"

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

mkdir -p "${LOG_DIR}"

python OmniGibson/omnigibson/learning/eval.py \
    policy=websocket \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3,4,5] \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test" \
    prompt="Pick up and lift up the radio receiver that's on the table in the living room."

# python OmniGibson/omnigibson/learning/eval.py \
#     policy=websocket \
#     task.name=turning_on_radio \
#     eval_on_train_instances=true \
#     eval_instance_ids=[0,1,2,3,4,5] \
#     log_path="${LOG_DIR}/${EXP_NAME}_tor_train"

# python OmniGibson/omnigibson/learning/eval.py \
#     policy=websocket \
#     task.name=picking_up_trash \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_picking_up_trash_test"

# python OmniGibson/omnigibson/learning/eval.py \
#     policy=websocket \
#     task.name=freeze_pies \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_fp_test"
