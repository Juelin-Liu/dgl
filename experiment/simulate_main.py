from dgl.dev import *
from simulation.simulate import *
from utils import *
import os

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
    num_partition = args.num_partition
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
                    num_partition=num_partition,
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
    
    if graph_name in ["products", "orkut", "papers100M"]:
        cfg.num_epoch = max(10, cfg.num_epoch)
        
    simulate(cfg)