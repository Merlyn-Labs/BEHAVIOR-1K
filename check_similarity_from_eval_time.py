"""
This script is a sanity check to confirm that the outputted action vector from the 50k steps
finetuned pi05 model, GENERATED DURING EVAL TIME, is close to the ground truth action vector.

We already know that the action vector generated from a datapoint from the dataset is close to
the ground truth action vector, so we expect that the version generated during eval time is also
close to the ground truth action vector.
"""

import os
import json

import torch
import torch.nn.functional as F

from test_inference_equality import load_pickle

ACTION_OUT_DIR = "./action_out"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_action_vector(eval_step: int) -> torch.Tensor:
    path = os.path.join(ACTION_OUT_DIR, "1", f"{eval_step}.json")
    return torch.tensor(load_json(path))


def main():
    for eval_step in [0, 49, 240]:
        print(f"Eval step: {eval_step}")
        action_vector = load_action_vector(eval_step)
        datapoint = load_pickle(f"curr_datapoint_{eval_step}.pkl")
        gt_action = datapoint["action"][[0]]
        breakpoint()
        print(f"F.cosine_similarity(action_vector_{eval_step}, gt_action_{eval_step})", F.cosine_similarity(action_vector, gt_action))
        print(f"F.mse_loss(action_vector_{eval_step}, gt_action_{eval_step})", F.mse_loss(action_vector, gt_action))
        print(action_vector)
        print(gt_action)
        print("")


if __name__ == "__main__":
    main()
