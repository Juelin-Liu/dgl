# python3 batch_sample.py --graph_name ogbn-products --pool_size 1 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 2 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 4 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 8 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 16 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 32 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 64 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-products --pool_size 128 --batch_layer=2

# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 1 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 2 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 4 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 8 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 16 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 32 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 64 --batch_layer=2
# python3 batch_sample.py --graph_name ogbn-papers100M --pool_size 128 --batch_layer=2

python3 dgl_sample.py --graph_name ogbn-products --system dgl-uva
python3 dgl_sample.py --graph_name ogbn-products --system dgl-gpu
python3 dgl_sample.py --graph_name ogbn-papers100M --system dgl-uva
python3 dgl_sample.py --graph_name ogbn-papers100M --system dgl-gpu
python3 base_sample.py --graph_name ogbn-products --system base-uva
python3 base_sample.py --graph_name ogbn-products --system base-gpu
python3 base_sample.py --graph_name ogbn-papers100M --system base-uva
python3 base_sample.py --graph_name ogbn-papers100M --system base-gpu