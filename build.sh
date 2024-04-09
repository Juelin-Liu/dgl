#!/bin/bash
export CUDA_HOME="${CUDA_HOME:=/usr/local/cuda}"

echo "CUDA_HOME=$CUDA_HOME"

bash init.sh

cmake -B build -GNinja

cmake --build build -j

cd python && pip install .