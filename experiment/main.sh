#!/bin/bash
for graph in products papers100M  orkut friendster;
do
  for model in sage gat;
    do
    for system in split dgl quiver dist_cache p3;
    	do
	batch_size=(1024)
	cache_sizes=(10G 8G 6G)
	if [ $system == "dist_cache" ]; then
           PYTHONPATH=/spara/third_party/dist_cache/quiver/srcs/python
        fi
	if [ $system == "quiver" ]; then
           PYTHONPATH=/spara/third_party/quiver/quiver/srcs/python
        fi
	for cache_size in ${cache_sizes[@]};
	  do
	  export PYTHONPATH=$PYTHONPATH;echo python3 train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=${world_size}  --data_dir=/data/gsplit --cache_size=${cache_size} --batch_size=${batch_size} --log_file=main.csv
	  unset PYTHONPATH  
          done
        done
    done
done
