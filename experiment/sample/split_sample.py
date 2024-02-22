
import torch
from torch.multiprocessing import spawn
from utils import *
from dgl.dev.splitloader import *


def split_sample(config: Config):
    graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=True)
    partition_map = None
    if "random " not in config.partition_type:
        partition_map = load_partition_map(config)
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.int8)

    try:
        NcclUniqueId = GetUniqueId()
        spawn(_split_sample, args=(config, NcclUniqueId, graph, train_idx, partition_map),
              nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)


def _split_sample(rank: int, config: Config, NcclUniqueId, graph: dgl.DGLGraph, train_idx: torch.Tensor, partition_map: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    sample_config = SampleConfig(rank=rank, num_partition=config.num_partition, batch_size=config.batch_size,
                                 world_size=config.world_size, mode=config.sample_mode, fanouts=config.fanouts, reindex=True,
                                 drop_last=True)

    dataloader = SplitGraphLoader(graph, partition_map, train_idx, NcclUniqueId, sample_config)
    step = 0
    step_per_epoch = dataloader.num_step_per_epoch
    
    print(f"sampling on device: {device}", flush=True)
    timer = Timer()
    num_edges = 0
    num_input_nodes = 0
    for epoch in range(config.num_epoch):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            num_input_nodes += input_nodes.shape[0]
            for block in blocks:
                num_edges += block.num_edges()

            log_step(rank, epoch, step, step_per_epoch, timer)
    print(f"{rank=} {num_edges=} {num_input_nodes=} duration={timer.duration(2)} secs")
    ddp_exit()
