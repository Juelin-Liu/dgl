#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"
graph=friendster
echo "Start Ablation Experiment on $graph"

world_size=4
export PYTHONPATH=/home/sandeep/juelin/dgl/third_party/dist_cache/torch-quiver/srcs/python
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
for model in  gat ; do

  for system in split; do
    batch_size=(1024)
    cache_sizes=(8G)
    sample_mode=uva
    if [ $system == "dgl" ]; then
        cache_sizes=(10G)
    fi
    if [ $model == "sage" ]; then
	cache_sizes=(6G 4G)
	if [ $system == "split" ]; then 
		sample_mode=uva
	fi 
    fi 
    if [ $model == "gat" ]; then
        cache_sizes=(0G)
    fi
    
	    

    for cache_size in ${cache_sizes[@]}; do
      for fanout in  "10,10,10,10"  ; do 
	batch_size=(1024)    
        python3 ${python_dir}/train_main.py --sample_mode=${sample_mode} --system=${system} --model=${model} --fanout=${fanout} --graph=${graph} --world_size=${world_size} --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=1024 --log_file=depth.csv --hid_size=256
      done
      for hidden in 384; do
	batch_size=(1024)
        echo python3 ${python_dir}/train_main.py --sample_mode=${sample_mode} --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=${world_size} --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=1024 --hid_size=${hidden} --log_file=hidden.csv
      done
      for batch_size in 2048; do
        echo python3 ${python_dir}/train_main.py --sample_mode=${sample_mode} --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=${world_size} --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=${batch_size} --log_file=batch_size.csv --hid_size=256
      done
      echo python3 ${python_dir}/train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=8 --data_dir=${data_dir} --cache_size=${cache_size} --batch_size=2048 --log_file=scalability.csv
    done
  done
done
echo "All ablation done"
