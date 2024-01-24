from dgl.dev import *
from node.utils import *
from node.trainer import *
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--num_epoch', default=20, type=int, help='Number of epochs (default 20)')
    parser.add_argument('--batch_size', default=1024, type=int, help='Input batch size on each device (default: 1024)')
    parser.add_argument('--fanouts', default="15,15,15", type=str, help='Input fanouts (15,15,15)')
    parser.add_argument('--system', default="dgl", type=str, help='System setting', choices=["dgl", "p3", "quiver", "groot-gpu", "groot-uva", "groot-cache"])
    parser.add_argument('--model', default="sage", type=str, help='Model type', choices=['sage', 'gat'])
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
    parser.add_argument('--data_dir', default="/data/juelin/dataset/gsplit", type=str, help="Input graph directory")
    parser.add_argument('--partition_type', default="", type=str, help="Input graph partition id type")
    parser.add_argument('--cache_size', default="0GB", type=str, help="Amount of GPU ram used for caching")
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--hid_feat', default=256, type=int, help='Size of hidden feature')
    parser.add_argument('--num_redundant_layers', default = 0, type = int, help = "number of redundant layers")
    parser.add_argument('--nvlink', default=0, type=int, help="whether server has nvlink", choices=[0, 1])
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    
    print(f"{args=}")
    graph_name = str(args.graph_name)
    system = args.system
    data_dir = args.data_dir
    model = args.model
    nvlink = args.nvlink
    batch_size = args.batch_size
    fanouts = args.fanouts.split(',')
    cache_size = args.cache_size
    partition_type = args.partition_type
    num_redundant_layers = args.num_redundant_layers
    num_epoch = args.num_epoch

    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    config = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=num_epoch,
                       fanouts=fanouts,
                       batch_size=batch_size,
                       system=system,
                       model=model,
                       cache_size=cache_size,
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
       
    config.num_redundant_layer = num_redundant_layers
    config.partition_type = partition_type
    config.nvlink = nvlink
    config.test_model_acc = 1 if graph_name in ["products"] else 0
    
    if "dgl" in system:
        bench_dgl_batch([config])
    elif "quiver" in system:
        bench_quiver_batch([config])
    elif "p3" in system:
        bench_p3_batch([config])