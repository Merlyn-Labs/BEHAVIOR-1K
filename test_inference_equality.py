import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pickle
import torch
import copy
import torch.nn.functional as F
from omnigibson.learning.policies import load_policy

# dict_keys(['robot_r1::robot_r1:left_realsense_link:Camera:0::rgb', 'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb', 'robot_r1::robot_r1:zed_link:Camera:0::rgb', 'robot_r1::proprio', 'robot_r1::cam_rel_poses'])
# (Pdb) curr_datapoint.keys()
# dict_keys(['index', 'episode_index', 'task_index', 'timestamp', 'observation.state', 'observation.cam_rel_poses', 'action', 'observation.task_info', 'action_is_pad', 'observation.images.rgb.left_wrist', 'observation.images.rgb.right_wrist', 'observation.images.rgb.head', 'task', 'skill_prompts'])

# obs.keys(): curr_datapoint.keys()
OBS_TO_DP_MAPPING = {
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": "observation.images.rgb.left_wrist",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": "observation.images.rgb.right_wrist",
    "robot_r1::robot_r1:zed_link:Camera:0::rgb": "observation.images.rgb.head",
    "robot_r1::proprio": "observation.state",
    "robot_r1::cam_rel_poses": "observation.cam_rel_poses",
}
IMAGE_KEYS = [
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
]

DP_TO_OBS_MAPPING = {v: k for k, v in OBS_TO_DP_MAPPING.items()}

def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def main():
    policy = load_policy(
        policy_config="pi05_b1k",
        policy_dir="/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/49999/", task_name="turning_on_radio",
    )

    # obs = load_pickle("obs.pkl")
    # curr_datapoint = load_pickle("curr_datapoint.pkl")

    obs = load_pickle("obs_240.pkl")
    curr_datapoint = load_pickle("curr_datapoint.pkl")
    ground_truth = curr_datapoint["action"][[0]]

    obs["robot_r1::proprio"] == curr_datapoint["observation.state"]
    divisor = obs["robot_r1::proprio"] / curr_datapoint["observation.state"]

    obs_from_datapoint = {DP_TO_OBS_MAPPING[k]: v for k, v in curr_datapoint.items() if k in DP_TO_OBS_MAPPING}
    assert set(obs.keys()) == set(obs_from_datapoint.keys()), "Keys are not equal!!!"

    for img_key in IMAGE_KEYS:
        obs_from_datapoint[img_key] = obs_from_datapoint[img_key].permute(1, 2, 0)

    obs_from_datapoint_numpy = {k: v.numpy() for k, v in obs_from_datapoint.items()}
    output_from_observation = policy.act(obs).detach().cpu()
    output_from_dataset_version = policy.act(obs_from_datapoint_numpy).detach().cpu()

    print("F.cosine_similarity(output_from_observation, output_from_dataset_version)", F.cosine_similarity(output_from_observation, output_from_dataset_version))
    print("F.cosine_similarity(output_from_observation, ground_truth)", F.cosine_similarity(output_from_observation, ground_truth))
    print("F.cosine_similarity(output_from_dataset_version, ground_truth)", F.cosine_similarity(output_from_dataset_version, ground_truth))

    print("F.mse_loss(output_from_observation, output_from_dataset_version)", F.mse_loss(output_from_observation, output_from_dataset_version))
    print("F.mse_loss(output_from_observation, ground_truth)", F.mse_loss(output_from_observation, ground_truth))
    print("F.mse_loss(output_from_dataset_version, ground_truth)", F.mse_loss(output_from_dataset_version, ground_truth))

    breakpoint()

    # for obs_k in obs.keys():
    #     obs_v = obs[obs_k]
    #     datapoint_v = obs_from_datapoint[obs_k]
    #     print("From observation", obs_k, type(obs_v), obs_v.shape, obs_v.dtype, obs_v.min(), obs_v.max(), obs_v.device)
    #     print("From dataset", obs_k, type(datapoint_v), datapoint_v.shape, datapoint_v.dtype, datapoint_v.min(), datapoint_v.max(), datapoint_v.device)
    #     print("")

    # breakpoint()

    # obs_empty = copy.deepcopy(obs)
    # obs_empty["robot_r1::robot_r1:zed_link:Camera:0::rgb"] = torch.zeros_like(obs_empty["robot_r1::robot_r1:zed_link:Camera:0::rgb"])

if __name__ == "__main__":
    main()
