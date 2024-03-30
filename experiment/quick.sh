
#for graph in orkut friendster;
#do
#for model in sage gat;
#do	
#python3 train_main.py --system=split --model=${model} --fanout=15,15,15 --graph=${graph} --world_size=2 \
#	--data_dir=/data/gsplit --cache_size=8G --batch_size=512 --num_epoch=2 --devices=0,1 --nodes=2
#
#
#done
#done

graph=products
model=sage
python3 train_main.py --system=quiver --model=${model} --fanout=15,15,15 --graph=${graph} --world_size=4 \
	--data_dir=/data/gsplit --cache_size=8G --batch_size=1024 --num_epoch=2  --num-nodes=1 --node-rank=0



