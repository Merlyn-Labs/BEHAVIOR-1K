#!/bin/bash

# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash && source ~/.bashrc && rm Miniconda3-latest-Linux-x86_64.sh

conda create -n behavior python=3.10 -c conda-forge -y
eval "$(conda shell.bash hook)"
conda activate behavior
echo $CONDA_DEFAULT_ENV

python -m pip install uv
python -m uv pip install "numpy<2" "setuptools<=79"
CUDA_VER_SHORT=$(echo $CUDA_VERSION | grep -oE '^[0-9]+\.[0-9]+' | sed 's/\.//')
python -m uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}

./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

python -m uv pip install -e /workspace/openpi/packages/openpi-client/
python -m uv pip install -e /workspace/openpi/
python -m uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}

apt -y update && apt install -y libsm6 libxext6 libxrender-dev libxt6 libglu1-mesa libice6 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxinerama1 libxcursor1 libxi6 libxtst6 libegl1 libopengl0 libglvnd0 libglx0 libgl1 libgtk-3-0 libnss3 xvfb libvulkan1 mesa-vulkan-drivers vulkan-tools
