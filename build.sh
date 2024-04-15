#!/bin/bash
export CUDA_HOME="${CUDA_HOME:=/usr/local/cuda}"

echo "CUDA_HOME=$CUDA_HOME"

bash init.sh

cmake -B build -GNinja

cmake --build build -j

cd python && pip install .

cd third_party/torch-quiver && python3 setup.py build_ext --inplace

cd third_party/dist_cache/torch-quiver && python3 setup.py build_ext --inplace
