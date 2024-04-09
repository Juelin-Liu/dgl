#!/bin/bash
bash init.sh
# build nccl for all supported devices
# pushd third_party/nccl && make -j src.build && popd

cmake -B build -GNinja && cmake --build build -j
