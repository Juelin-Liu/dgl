#!/bin/bash

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${WORK_DIR}/../script/env.sh"

# Get Weight
for graph_name in products orkut papers100M friendster; do
    python3 ${WORK_DIR}/get_weight.py --data_dir=$data_dir --graph_name=$graph_name --fanouts=20,20,20 --num_epoch=3
done