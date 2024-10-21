#!/bin/bash

python3 -m venv venv

source venv/bin/activate

pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip3 install ninja cmake torchmetrics pandas jupyterlab scipy tqdm