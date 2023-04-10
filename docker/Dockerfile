FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    ffmpeg \
    unzip \
    lsb-release \
    vim \
    git \
    wget \
    htop \
    byobu \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python package
ENV LANG C.UTF-8
RUN pip3 install gym open3d scipy opencv-python numpy nlopt scipy transforms3d imageio nvitop setuptools opencv-contrib-python tensorboard moviepy h5py --upgrade
RUN pip3 install https://storage1.ucsd.edu/wheels/sapien-dev/sapien-2.0.0.dev20220531-cp38-cp38-manylinux2014_x86_64.whl
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
