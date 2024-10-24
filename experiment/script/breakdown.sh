#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"
echo "Start Time breakdown Benchmark Experiment"
export PYTHONPATH=/home/ubuntu/dgl/third_party/torch-quiver/srcs/python

for graph in orkut; do
    for model in sage; do
        for system in quiver dgl p3; do
            python3 ${python_dir}/train_main.py --system=${system} --model=${model} --graph=${graph} --fanout="15,15,15" --data_dir=${data_dir} --cache_size=2GB --batch_size=1024 --log_file=main.csv
        done
    done
done

for graph in papers100M friendster; do
    for cache_size in 8GB; do
        for system in quiver; do
            python3 ${python_dir}/train_main.py --system=${system} --model=sage --graph=${graph} --fanout="15,15,15" --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=1024 --log_file=main.csv
        done
    done
done