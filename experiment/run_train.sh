#!/bin/bash
source ~/.profile
export MACHINE_NAME=Power9

python_script=$1
data_dir=$2
graph_name=$3
num_epoch=$4
batch_size=$5
fanouts=$6
system=$7
model=$8
nvlink=$9
cache_size=${10}

flags="--nvlink=${nvlink} --cache_size=${cache_size} --data_dir=${data_dir} --graph_name=${graph_name} --num_epoch=${num_epoch} --batch_size=${batch_size} --fanouts=${fanouts} --system=${system} --model=${model}"
# echo Flags=$flags
python $python_script $flags
# python $python_script --nvlink=${nvlink} --data_dir=${data_dir} --graph_name=${graph_name} --num_epoch=${num_epoch} --batch_size=${batch_size} --fanouts=${fanouts} --system=${system} --model=${model} 
