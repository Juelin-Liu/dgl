FROM vidia/cuda:11.8.0-devel-ubuntu22.04

RUN export CUDA_HOME=/usr/local/cuda

RUN apt update

RUN apt install -y wget

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

RUN sh Miniforge3.sh -b -p "/conda"

RUN bash "/conda/etc/profile.d/conda.sh"

RUN bash "/conda/etc/profile.d/mamba.sh"

RUN mamba init

RUN source /root/.bashrc

RUN pip install -y torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN mamba install -y -c pyg pyg

RUN mamba install -y ninja cmake

RUN mamba install -y torchmetrics jupyterlab numpy matplotlib

COPY . /spara

RUN cd /spara && ./build.sh
