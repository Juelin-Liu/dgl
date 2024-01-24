#!/bin/bash

nvlink=0
MACHINE_NAME=jupiter
batch_size=1024
fanouts="15,15,15"
for graph_name in friendster 
do
    for model in sage gat
    do
        for system in p3 dgl
        do
            echo running experiment on $graph_name using $system $model
            python run.py --system=${system} --model=${model} --nvlink=${nvlink} --graph_name=${graph_name} --batch_size=${batch_size} --fanouts=${fanouts}
        done
        for cache_size in 0GB 1GB 2GB #3GB 4GB 5GB 6GB 7GB 8GB 9GB 10GB
        do 
            system=quiver
            echo running experiment on $graph_name using $system $model with cache size $cache_size
            python run.py --cache_size=${cache_size} --system=${system} --model=${model} --nvlink=${nvlink} --graph_name=${graph_name} --batch_size=${batch_size} --fanouts=${fanouts}
        done
    done
done