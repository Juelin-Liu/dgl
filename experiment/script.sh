for model in gat sage;
do
for world_size in 4 8;
	do
	batch_size=(1024)
	if [ $world_size -eq 8 ]; then
	   batch_size=(2048)
	fi
	if [ $model == "sage" ]; then
	   cache_sizes=(10G)
	fi
        if [ $model == "gat" ]; then
           cache_sizes=(10G 8G)
        fi
	for cache_size in ${cache_sizes};
do	
	python3 train_main.py --model=${model} --fanout="20,20,20" --graph=papers100M --world_size=${world_size}  --data_dir=/data/gsplit --cache_size=${cache_size} --batch_size=${batch_size}

done
done
done
