#!/bin/bash
export CUDA_HOME="${CUDA_HOME:=/usr/local/cuda}"

echo "CUDA_HOME=$CUDA_HOME"

cmake -B build -GNinja -DCMAKE_BUILD_TYPE=release -DBUILD_TYPE=release

cmake --build build -j

cd python && pip install . && cd ../