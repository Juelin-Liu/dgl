#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"
echo "Start Main Benchmark Experiment"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for graph in friendster; do
  for model in sage gat; do
    system=split
    cache_sizes=(6G 4G)
   for cache_size in ${cache_sizes[@]}; do
      #python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph}  --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=1024 --log_file=partitioning.csv
      python3 ${python_dir}/train_main.py --sample_mode=gpu --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph}  --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=1024 --log_file=partitioning.csv --node_weight=degree --edge_weight=uniform --bal=bal

     done
  done
done
