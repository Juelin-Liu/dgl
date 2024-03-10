#!/bin/bash
for graph in friendster #products papers100M  orkut friendster;
do
for model in sage #gat;
do
for world_size in 4 ;
	do
	batch_size=(1024)
	if [ $world_size -eq 8 ]; then
	   batch_size=(2048)
	fi
	if [ $model == "sage" ]; then
	   cache_sizes=(10G)
	fi
        if [ $model == "gat" ]; then
           cache_sizes=(10G)
        fi
	for cache_size in ${cache_sizes};
do	
	for system in split  #dgl ;
	do
	python3 train_main.py --system=${system} --model=${model} --fanout="15,15,15" --graph=${graph} --world_size=${world_size}  --data_dir=/data/gsplit --cache_size=${cache_size} --batch_size=${batch_size}
  done
done
done
done
done
