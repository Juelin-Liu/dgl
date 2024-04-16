from dgl.dev import *
from simulation.simulate import *
from utils import *
import argparse, os

# def get_parser():
#     parser = argparse.ArgumentParser(description='local run script')
#     parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
#     parser.add_argument('--num_epoch', default=20, type=int, help='Number of epochs to be sampled (default 20)')
#     parser.add_argument('--fanouts', default="15,15,15", type=str, help='Input fanouts (15,15,15)')
#     parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
#     parser.add_argument('--data_dir', default="/data/juelin/dataset/gsplit", type=str, help="Input graph directory")
#     parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
#     parser.add_argument('--node_mode', default="uniform", type=str, help="Node weight configuraion", choices=["uniform", "degree", "src", "dst", "input", "random"] )
#     parser.add_argument('--edge_mode', default="uniform", type=str, help="Edge weight configuraion", choices=["uniform", "freq", "random"])
#     parser.add_argument('--bal', default="bal", type=str, help='Balance target idx on each partition or not', choices=["bal", "xbal"])
#     return parser

if __name__ == "__main__":
    
    args = get_args()    
    graph_name = args.graph_name
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    node_weight = args.node_weight
    edge_weight = args.edge_weight
    bal = args.bal
    system = args.system
    sample_mode = args.sample_mode
    world_size = args.world_size
    model = args.model
    cache_size = args.cache_size
    hid_size = args.hid_size
    fanouts = args.fanouts.split(',')

    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    cfg = Config(graph_name=graph_name,
                    world_size=world_size,
                    num_partition=world_size,
                    num_epoch=num_epoch,
                    fanouts=fanouts,
                    batch_size=batch_size,
                    system=system,
                    model=model,
                    cache_size=cache_size,
                    hid_size=hid_size,
                    log_path=log_path,
                    data_dir=data_dir,
                    nvlink=False,
                    partition_type=get_partition_type(node_weight, edge_weight, bal),
                    sample_mode=sample_mode)
           
    simulate(cfg)