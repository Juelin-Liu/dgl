#!/bin/bash

#rm -rf build

cmake -DBUILD_GRAPHBOLT=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR  -DUSE_CUDA=ON -B build -GNinja

cmake --build build -j
