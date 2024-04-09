FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "--login", "-c"]

RUN export CUDA_HOME=/usr/local/cuda

RUN apt update

RUN apt install -y wget

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

RUN bash Miniforge3.sh -b -p "/conda"

RUN source "/conda/etc/profile.d/conda.sh"

RUN source "/conda/etc/profile.d/mamba.sh"

RUN /conda/bin/conda init bash

RUN source "/root/.bashrc"

# install build essentials
RUN /conda/bin/conda install -y ninja cmake

# install pytorch v2.0
RUN /conda/bin/pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install pyg
RUN /conda/bin/pip install torch_geometric

# install pyg lib
RUN /conda/bin/pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# install dependencies
RUN /conda/bin/pip install torchmetrics jupyterlab numpy matplotlib pandas ogb

# copy source file
COPY . /spara

WORKDIR /spara

# set environment variable
RUN export CUDA_HOME=/usr/local/cuda

# build project
RUN /conda/bin/conda run -n base bash ./build.sh

RUN /conda/bin/conda run -n base cd ./python && pip install .