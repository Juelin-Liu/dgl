import argparse
import os
from utils import get_partition_type, Config, get_args

if __name__ == "__main__":

    args = get_args()

    print(f"input {args=}")
    
    graph_name = args.graph_name
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    nmode = args.nmode
    emode = args.emode
    bal = args.bal
    system = args.system
    sample_mode = args.sample_mode
    partition_type = get_partition_type(nmode, emode, bal)
    world_size = args.world_size
    fanouts = args.fanouts.split(',')
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    
    config = Config(graph_name=graph_name,
                    world_size=world_size,
                    num_partition=world_size,
                    num_epoch=num_epoch,
                    fanouts=fanouts,
                    batch_size=batch_size,
                    system=system,
                    model="none",
                    cache_size="0GB",
                    hid_size=256,
                    log_path=log_path,
                    data_dir=data_dir,
                    nvlink=False,
                    partition_type=partition_type,
                    sample_mode=sample_mode)

    if config.system == "split":
        from sample.split_sample import split_sample
        split_sample(config)
    elif config.system == "dgl":
        from sample.dgl_sample import dgl_sample
        dgl_sample(config)