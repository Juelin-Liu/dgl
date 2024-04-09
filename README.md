# Artifacts for Spara

This repository contains the source code of Spara, an efficient distributed GNN training system. Scripts to reproduce the Figures and Tables in the paper are also provided.

# Project Structure
Spara is highly integrated into DGL. The Python part for Spara is in the `python/dgl/dev` directory whereas the cpp source code are in the `src/spara` directory.

# Software
We assume you use a Linux machine.

# How to build

## Install Docker
Follow the instructions from this [website](https://docs.docker.com/engine/install/ubuntu/) to install docker.

## Install Nividia CUDA Driver
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run --silent --driver
sudo reboot
```

## Install NVIDIA Container Toolkit
We recommend using docker to create the development environment. To do this, you need to [configure docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to use nvidia GPUs inside a container. After installing Docker and the Nvidia driver, you can follow the remaining instructions to install the nvidia container toolkit.

Configure the production repository of nvidia-container-toolkit:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install and configure the container runtime by using the nvidia-ctk command:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Build the Docker Image

```bash
sudo docker build -t spara:latest .
```

## Run the docker image

```bash
sudo docker run --gpus all -it spara:latest
```

## Build Spara from the source

Inside the docker container, you can find all the source code in the `/spara` directory.

Then, you can build the project by running:
```bash
./build.sh
```

# Download Dataset

## Preprocessed Dataset

# Run Experiments