from dgl.dev import *
from preprocess.partition import *
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='local run script')
    parser.add_argument('--graph_name', default="products", type=str, help="Input graph name", choices=["products", "papers100M", "orkut", "friendster"])
    parser.add_argument('--data_dir', default="/work/pi_mserafini_umass_edu/dataset/gsplit/", type=str, help="Input graph directory")
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--node_mode', default="uniform", type=str, help="Node weight configuraion", choices=["uniform", "degree", "src", "dst", "input"] )
    parser.add_argument('--edge_mode', default="uniform", type=str, help="Edge weight configuraion", choices=["uniform", "freq"])
    parser.add_argument('--bal', default="bal", type=str, help='Balance target idx on each partition or not', choices=["bal", "xbal"])

    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    
    print(f"{args=}")
    graph_name = str(args.graph_name)
    data_dir = args.data_dir
    bal = args.bal
    node_mode = args.node_mode
    edge_mode = args.edge_mode
    world_size= args.world_size
    

    batch_size = 0
    fanouts = [0]
    num_epoch = 0
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    config = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=num_epoch,
                       fanouts=fanouts,
                       batch_size=batch_size,
                       system="dgl-sample",
                       model="none",
                       cache_size="0GB",
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
           
    partition(config, node_mode, edge_mode, bal)