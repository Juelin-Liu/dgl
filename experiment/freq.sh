#!/bin/bash

fanouts='15,15,15'
batch_size=1024

for graph_name in products orkut papers100M friendster
do
    echo $graph_name
    python3 freq.py --fanouts=${fanouts} --batch_size=${batch_size} --graph_name=${graph_name}
done