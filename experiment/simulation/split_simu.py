import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from node.utils import *
from dgl.dev.splitloader import *


def load_partition_map(config: Config, node_mode: str, edge_mode: str, bal: str, num_partitions=4):
    in_dir = os.path.join(config.data_dir, "partition_ids", config.graph_name)
    file_name = f"{config.graph_name}_w{num_partitions}_n{node_mode}_e{edge_mode}_{bal}.pt"
    return torch.load(os.path.join(in_dir, file_name)).type(torch.int8)


def simulate(config: Config, node_mode: str, edge_mode: str, bal: str):
    graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=True)
    partition_map = None
    if node_mode != "random":
        partition_map = load_partition_map(config, node_mode, edge_mode, bal, config.num_partition)
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.int8)

    try:
        unique_id = GetUniqueId()
        spawn(_simulate, args=(config, unique_id, graph, train_idx, partition_map, node_mode, edge_mode, bal),
              nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)


def _simulate(rank: int, config: Config, unique_id, graph: dgl.DGLGraph, train_idx: torch.Tensor, partition_map: torch.Tensor,
              node_mode: str, edge_mode: str, bal: str):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()

    if rank == 0:
        print(config)
    mode = "uva"
    graph = graph.pin_memory_()
    assert(partition_map.shape[0] == graph.num_nodes())
    #if "friendster" in config.graph_name:
    #    graph = graph.pin_memory_()
    #    mode = "uva"
    #else:
    #    graph = graph.to(rank)
    #    mode = "gpu"
    sample_config = SampleConfig(rank=rank, num_partition=config.num_partition, batch_size=config.batch_size,
                                 world_size=config.world_size, mode=mode, fanouts=config.fanouts, reindex=False,
                                 drop_last=True)

    dataloader = SplitGraphLoader(graph, partition_map, train_idx, sample_config)
    InitNccl(rank, config.world_size, unique_id)
    step = 0
    step_per_epoch = dataloader.num_step_per_epoch
    UseBitmap(True)
    print(f"sampling on device: {device}", flush=True)
    timer = Timer()
    epoch_num = config.num_epoch

    v_num = graph.num_nodes()
    e_num = graph.num_edges()

    src_node_cnts = []
    dst_node_cnts = []
    input_node_cnts = []
    crs_edge_cnts = []
    loc_edge_cnts = []

    def get_edge_cnts(src, dst, mapping):
        src_partition = mapping[src]
        dst_partition = mapping[dst]
        loc_mask = src_partition == dst_partition
        crs_mask = src_partition != dst_partition
        loc_cnts = torch.bincount(mapping[src[loc_mask]],
                                  minlength=config.world_size)  # use the src/s partition id to map the local edge (identical)
        crs_cnts = torch.bincount(mapping[src[crs_mask]],
                                  minlength=config.world_size)  # use the dst's partition id to map the cross edge (subject to change)
        return loc_cnts, crs_cnts

    def get_node_cnts(nodes, mapping):
        return torch.bincount(mapping[nodes], minlength=config.world_size)

    def lst_to_tensor(lst):
        return torch.stack(lst)

    num_edges = 0
    for epoch in range(config.num_epoch):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            # input_node_cnts.append(get_node_cnts(input_nodes, partition_map))
            for block in blocks:
                # src, dst = block.all_edges()
                # loc_cnts, crs_cnts = get_edge_cnts(src, dst, partition_map)
                # loc_edge_cnts.append(loc_cnts)
                # crs_edge_cnts.append(crs_cnts)
                #
                # unique_src = block.srcnodes()
                # src_node_cnts.append(get_node_cnts(unique_src, partition_map))
                #
                # unique_dst = block.dstnodes()
                # dst_node_cnts.append(get_node_cnts(unique_dst, partition_map))

                # edge_id = block.edata["_ID"]
                num_edges += block.num_edges()

            log_step(rank, epoch, step, step_per_epoch, timer)
    print(f"{rank=} {num_edges=}")
    # src_node_cnts = lst_to_tensor(src_node_cnts)
    # dst_node_cnts = lst_to_tensor(dst_node_cnts)
    # input_node_cnts = lst_to_tensor(input_node_cnts)
    # loc_edge_cnts = lst_to_tensor(loc_edge_cnts)
    # crs_edge_cnts = lst_to_tensor(crs_edge_cnts)
    #
    # all_src = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank,
    #                       dtype=src_node_cnts.dtype)
    # all_dst = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank,
    #                       dtype=src_node_cnts.dtype)
    # all_loc = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank,
    #                       dtype=src_node_cnts.dtype)
    # all_crs = torch.zeros((src_node_cnts.shape[0] * config.world_size, config.world_size), device=rank,
    #                       dtype=src_node_cnts.dtype)
    # all_input = torch.zeros((input_node_cnts.shape[0] * config.world_size, config.world_size), device=rank,
    #                         dtype=input_node_cnts.dtype)
    #
    # dist.all_gather_into_tensor(all_src, src_node_cnts)
    # dist.all_gather_into_tensor(all_dst, dst_node_cnts)
    # dist.all_gather_into_tensor(all_input, input_node_cnts)
    # dist.all_gather_into_tensor(all_loc, loc_edge_cnts)
    # dist.all_gather_into_tensor(all_crs, crs_edge_cnts)
    #
    # if rank == 0:
    #     all_src = all_src.to("cpu")
    #     all_dst = all_dst.to("cpu")
    #     all_loc = all_loc.to("cpu")
    #     all_crs = all_crs.to("cpu")
    #     all_input = all_input.to("cpu")
    #
    #     out_dir = os.path.join(config.data_dir, "workload", config.graph_name)
    #     print("saving to", out_dir)
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #     state = {
    #         "src": all_src,
    #         "dst": all_dst,
    #         "input": all_input,
    #         "loc": all_loc,
    #         "crs": all_crs
    #     }
    #
    #     file_name = f"{config.graph_name}_w{config.world_size}_n{node_mode}_e{edge_mode}_{bal}.pt"
    #     # torch.save(state, os.path.join(out_dir, file_name))
    #     print(state)

    ddp_exit()