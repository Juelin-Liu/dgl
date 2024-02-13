#!/bin/bash

for graph_name in papers100M #friendster products orkut 
do
    for use_bitmap in 0 #1
    do
	for mode in uva
	do
	    basename=${graph_name}_${mode}_${use_bitmap}
        nsys profile -o $basename -f true python3 sample.py --use_bitmap=$use_bitmap --graph_name=$graph_name --mode=$mode >> log.txt
	    # tail -n 3 log.txt
	    # mkdir -p reports/
	    # nsys stats --report cuda_gpu_kern_sum ${basename}.nsys-rep -o reports/${basename}
        done
    done
done

