ex#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"
echo "Start Main Benchmark Experiment"

for graph in  friendster; do #	products papers100M orkut friendster; do
  for model in sage ; do #gat
    for system in split; do
	    #dgl quiver dist_cache p3; do
      batch_size=(1024)
      cache_sizes=(10G)
      if [ $system == "split" ]; then
          cache_sizes=(10G 8G 6G 4G)
      	  cache_sizes=(6G)
      fi
      if [ $system == "dist_cache" ]; then
        PYTHONPATH=/spara/third_party/dist_cache/torch-quiver/srcs/python
        cache_sizes=(10G 8G 6G)
      fi
      if [ $system == "quiver" ]; then
        PYTHONPATH=/spara/third_party/torch-quiver/srcs/python
          cache_sizes=(10G 8G 6G)
      fi
      for cache_size in ${cache_sizes[@]}; do
        export PYTHONPATH=$PYTHONPATH
        python3 ${python_dir}/train_main.py --sample_mode=gpu --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph}  --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --log_file=main.csv 
        unset PYTHONPATH
      done
    done
  done
done
