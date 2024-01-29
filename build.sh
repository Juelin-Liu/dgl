#!/bin/bash
bash ./experiment/preprocess/PyTorchMetis/init.sh
cmake -B build -GNinja && cmake --build build -j
