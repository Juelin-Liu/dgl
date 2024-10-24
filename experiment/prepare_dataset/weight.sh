#!/bin/bash

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${WORK_DIR}/../script/env.sh"

for world_size in 4; do
    for graph_name in orkut papers100M friendster; do
        for num_epoch in 3 5 10; do
            python3 ${WORK_DIR}/get_weight_fast.py --data_dir=$data_dir --graph_name=$graph_name --fanouts=15,15,15 --num_epoch=${num_epoch} --world_size=${world_size}
        done
    done
done
