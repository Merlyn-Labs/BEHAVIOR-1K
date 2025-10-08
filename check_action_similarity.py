"""
This script is a sanity check to confirm that the base pi05 model outputs "garbage" action vectors,
while the 50k steps finetuned pi05 model outputs correct action vectors.

We see expect that the cosine similarity between the ground truth action and the predicted action
by the base model is low, while the MSE loss is high.

We also expect that the cosine similarity between the ground truth action and the predicted action
by the finetuned model is high, while the MSE loss is low.
"""

import torch.nn.functional as F

from omnigibson.learning.policies import load_policy

from test_inference_equality import OBS_TO_DP_MAPPING, IMAGE_KEYS, load_pickle


def get_obs_from_datapoint(datapoint):
    obs = {
        "robot_r1::proprio": datapoint["observation.state"],
        "robot_r1::cam_rel_poses": datapoint["observation.cam_rel_poses"],
    }
    for img_key in IMAGE_KEYS:
        obs[img_key] = datapoint[OBS_TO_DP_MAPPING[img_key]].permute(1, 2, 0)
    return {k: v.numpy() for k, v in obs.items()}


def main():
    policy_trained = load_policy(
        policy_config="pi05_b1k",
        policy_dir="/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/49999/",
        task_name="turning_on_radio",
    )

    policy_base = load_policy(
        policy_config="pi05",
        policy_dir="gs://openpi-assets/checkpoints/pi05_base/",
        task_name="turning_on_radio",
    )

    datapoint_0 = load_pickle("curr_datapoint.pkl")
    datapoint_49 = load_pickle("curr_datapoint_49.pkl")
    datapoint_240 = load_pickle("curr_datapoint_240.pkl")

    gt_action_0 = datapoint_0["action"][[0]]
    gt_action_49 = datapoint_49["action"][[0]]
    gt_action_240 = datapoint_240["action"][[0]]

    # print("F.cosine_similarity(gt_action_0, gt_action_49)", F.cosine_similarity(gt_action_0, gt_action_49))
    # print("F.cosine_similarity(gt_action_0, gt_action_240)", F.cosine_similarity(gt_action_0, gt_action_240))
    # print("F.cosine_similarity(gt_action_49, gt_action_240)", F.cosine_similarity(gt_action_49, gt_action_240))

    # print("F.mse_loss(gt_action_0, gt_action_49)", F.mse_loss(gt_action_0, gt_action_49))
    # print("F.mse_loss(gt_action_0, gt_action_240)", F.mse_loss(gt_action_0, gt_action_240))
    # print("F.mse_loss(gt_action_49, gt_action_240)", F.mse_loss(gt_action_49, gt_action_240))

    obs_from_dp_0 = get_obs_from_datapoint(datapoint_0)
    obs_from_dp_49 = get_obs_from_datapoint(datapoint_49)
    obs_from_dp_240 = get_obs_from_datapoint(datapoint_240)

    # pred_action_trained_0 = policy_trained.act(obs_from_dp_0).detach().cpu()
    # pred_action_trained_49 = policy_trained.act(obs_from_dp_49).detach().cpu()
    # pred_action_trained_240 = policy_trained.act(obs_from_dp_240).detach().cpu()

    # pred_action_base_0 = policy_base.act(obs_from_dp_0).detach().cpu()
    # pred_action_base_49 = policy_base.act(obs_from_dp_49).detach().cpu()
    # pred_action_base_240 = policy_base.act(obs_from_dp_240).detach().cpu()

    # print("F.cosine_similarity(gt_action_0, pred_action_base_0)", F.cosine_similarity(gt_action_0, pred_action_base_0))
    # print("F.mse_loss(gt_action_0, pred_action_base_0)", F.mse_loss(gt_action_0, pred_action_base_0))

    # print("F.cosine_similarity(gt_action_0, pred_action_trained_0)", F.cosine_similarity(gt_action_0, pred_action_trained_0))
    # print("F.mse_loss(gt_action_0, pred_action_trained_0)", F.mse_loss(gt_action_0, pred_action_trained_0))

    for inference_step, dp_dict in zip([0, 49, 240], [datapoint_0, datapoint_49, datapoint_240]):
        obs_from_dp = get_obs_from_datapoint(dp_dict)
        gt_action = dp_dict["action"][[0]]
        pred_action_base = policy_base.act(obs_from_dp).detach().cpu()
        pred_action_trained = policy_trained.act(obs_from_dp).detach().cpu()
        print(f"Inference step: {inference_step}")
        print(f"F.cosine_similarity(gt_action_{inference_step}, pred_action_base_{inference_step})", F.cosine_similarity(gt_action, pred_action_base))
        print(f"F.mse_loss(gt_action_{inference_step}, pred_action_base_{inference_step})", F.mse_loss(gt_action, pred_action_base))
        print(f"F.cosine_similarity(gt_action_{inference_step}, pred_action_trained_{inference_step})", F.cosine_similarity(gt_action, pred_action_trained))
        print(f"F.mse_loss(gt_action_{inference_step}, pred_action_trained_{inference_step})", F.mse_loss(gt_action, pred_action_trained))
        print("")


if __name__ == "__main__":
    main()
