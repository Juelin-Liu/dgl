#!/bin/bash
export CUDA_HOME="${CUDA_HOME:=/usr/local/cuda}"

echo "CUDA_HOME=$CUDA_HOME"

cmake -B build -GNinja -DCMAKE_BUILD_TYPE=debug -DBUILD_TYPE=debug

cmake --build build -j

cd python && pip install . && cd ../