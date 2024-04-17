ex#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"
echo "Start Main Benchmark Experiment"

for graph in products papers100M orkut friendster; do
  for model in sage gat; do
    system=split
    cache_sizes=(10G 8G)
   for cache_size in ${cache_sizes[@]}; do
      python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph}  --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --log_file=partitioning.csv
      python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph}  --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --log_file=partitioning.csv --node_weight=degree --edge_weight=uniform --bal=bal

     done
  done
done
