from dgl.dev import *
from node.utils import *
from node.trainer import *
import argparse

def get_default_config(graph_name, system, model, log_path, data_dir, nvlink, num_redundant_layer = 0):
    partitioning_graph = ""
    balancing="edge"
    training_nodes="xbal"
    config = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=1,
                       fanouts=[15, 15, 15],
                       batch_size=1024,
                       system=system,
                       model=model,
                       cache_size=0,
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
       
    config.num_redundant_layer = num_redundant_layer
    config.partition_type = f"{partitioning_graph}_w4_{balancing}_{training_nodes}"
    config.nvlink = nvlink
    
    return config

def get_parser():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--batch', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--system', default="dgl", type=str, help='System setting', choices=["dgl", "p3", "quiver", "groot-gpu", "groot-uva", "groot-cache"])
    parser.add_argument('--model', default="sage", type=str, help='Model type', choices=['sage', 'gat'])
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
    parser.add_argument('--data_dir', default="/data/gsplit", type=str, help="Input graph directory")
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--hid_feat', default=256, type=int, help='Size of hidden feature')
    parser.add_argument('--cache_size', default=0, type=float, help="percentage of feature data cached on each gpu")
    parser.add_argument('--num_redundant_layers', default = 0, type = int, help = "number of redundant layers")
    parser.add_argument('--nvlink', default=0, type=int, help="whether server has nvlink", choices=[0, 1])
    parser.add_argument('--test_acc', default=0, type=int, help="whether test model accuracy", choices=[0, 1])
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    
    print(f"{args=}")
    graph_name = str(args.graph_name)
    system = args.system
    data_dir = args.data_dir
    model = args.model
    nvlink = args.nvlink
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    config = get_default_config(graph_name=graph_name, system=system, model=model, log_path=log_path, data_dir=data_dir, nvlink=nvlink)
    if "dgl" in system:
        bench_dgl_batch([config])
    elif "quiver" in system:
        bench_quiver_batch([config])
    elif "p3" in system:
        bench_p3_batch([config])