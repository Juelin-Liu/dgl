import os
from utils import get_partition_type, get_args, Config

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
    model = args.model
    cache_size = args.cache_size
    hid_size = args.hid_size
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
                    model=model,
                    cache_size=cache_size,
                    hid_size=hid_size,
                    log_path=log_path,
                    data_dir=data_dir,
                    nvlink=False,
                    partition_type=partition_type,
                    sample_mode=sample_mode)
    config.test_model_acc = True
    if config.system == "split":
        from nodepred.trainer import bench_split
        bench_split(config)
    elif config.system == "dgl":
        from nodepred.trainer import bench_dgl_batch
        bench_dgl_batch([config])
    elif config.system == "quiver":
        from nodepred.quiver_trainer import bench_quiver_batch
        bench_quiver_batch([config])
    elif config.system == "p3":
        from nodepred.p3_trainer import bench_p3_batch
        bench_p3_batch([config])

