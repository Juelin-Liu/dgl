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

# Option 1: Using Docker
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
$sudo docker run --shm-size=180GB --gpus all -itd spara:latest
<container_id>
$sudo docker exec -it <container_id> /bin/bash
```

# Option 2: Not Using Docker 

```bash
conda create -n spara python=3.10
conda activate spara

# install pytorch v2.0
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install pyg
pip install torch_geometric

# install pyg lib
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# install dependencies
pip install torchmetrics jupyterlab numpy matplotlib pandas ogb
```

Then, install CUDA toolkit 11.8 and set `CUDA_HOME` environment variable accordingly.
Also install a C++ compiler, CMake, and Ninja, which will be used to build the source code.

## Build Spara from the source

Inside the docker container, you can find all the source code in the `/spara` directory.

Then, you can build the project by running:
```bash
./build.sh
```

Then, check if you have installed Spara correctly:
```bash
python -c "import dgl; print(dgl.__path__)"
```
The output should be something like `['/conda/lib/python3.10/site-packages/dgl']`, indicating you have installed Spara correctly.

# Download Dataset
# Option I: Download Raw Dataset and Process Them
Please refer to [README.md](./experiment/prepare_dataset/README.md) for more details.

# Option II: Download Preprocessed Dataset
| Item | Description | Download Size | Uncompressed Size |  
| --- | --- | --- | --- | 
| [Graphs](https://spara-artifact-sc24.s3.us-east-2.amazonaws.com/dataset.tar.gz) | It contains graph topology, label, node and edge weights for produces, paper100M, orkut and friendster. | 75GB | 144GB | \
| [Partition Maps](https://spara-artifact-sc24.s3.us-east-2.amazonaws.com/partition_map.tar.gz) | Contains partition maps for produces, paper100M, orkut and friendster using different combination of nodes and edge weights | 1GB | 4.1GB |

You can use the `download.sh` script to download the dataset and save it to `./dataset` directory. The graph topology data will be in `./dataset/graph` directory whereas the partition maps will be inside `./dataset/graph/partition_map` directory.
You can change the `dataset_dir` variable inside the [`env.sh`](./experiment/script/env.sh) script to change the default downloading folder. 
(Notice that the `partition_map` directory must be inside `graph` directory so the dataloading function can find the maps correctly.)

# Run Experiments
## Experiment Result Generation

## Data Analysis and Visualization


