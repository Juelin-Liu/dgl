#!/bin/bash

nvlink=1
MACHINE_NAME=p3_8xlarge

for graph_name in orkut friendster
do
    for model in sage gat
    do
        for system in quiver p3 dgl
        do
            echo running experiment on $graph_name using $system model $model
            python run.py --system=${system} --model=${model} --nvlink=${nvlink} --graph_name=${graph_name}
        done
    done
done