import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from node.utils import *
from dgl.dev.splitloader import *
from dgl.dev.splitmodel import get_distributed_model


def load_partition_map(config: Config, node_mode: str, edge_mode: str, bal: str, num_partitions=4):
    in_dir = os.path.join(config.data_dir, "partition_ids", config.graph_name)
    file_name = f"{config.graph_name}_w{num_partitions}_n{node_mode}_e{edge_mode}_{bal}.pt"
    return torch.load(os.path.join(in_dir, file_name)).type(torch.int8)


def bench_split(config: Config, node_mode: str, edge_mode: str, bal: str):
    # graph, train_idx, valid_idx, test_idx = load_topo(config, is_pinned=True)
    graph, feat, label, train_idx, valid_idx, test_idx, num_label = load_data(config, is_pinned=True)
    config.num_classes = num_label
    
    partition_map = None
    if node_mode != "random":
        partition_map = load_partition_map(config, node_mode, edge_mode, bal, config.num_partition)
    else:
        print(f"using random with {config.num_partition} partitions")
        v_num = graph.num_nodes()
        partition_map = torch.randint(low=0, high=config.num_partition, size=(v_num,), dtype=torch.int8)

    try:
        unique_id = GetUniqueId()
        spawn(train_split_ddp, args=(config, unique_id, graph, feat, label, train_idx, valid_idx, partition_map),
              nprocs=config.world_size)
    except Exception as e:
        print(f"error encountered with {config=}:", e)
        exit(-1)


def train_split_ddp(rank: int, config: Config, unique_id, 
                    graph: dgl.DGLGraph, feat:torch.Tensor, label: torch.Tensor, 
                    train_idx: torch.Tensor, valid_idx: torch.Tensor, 
                    partition_map: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    if rank == 0:
        print(config)
    mode = "uva"
    graph = graph.pin_memory_()
    feat_handle  = pin_memory_inplace(feat)
    label_handle = pin_memory_inplace(label)
    
    assert(partition_map.shape[0] == graph.num_nodes())
    
    model = get_distributed_model(rank, config.world_size, config.num_redundant_layer, 
                                  config.model, feat.shape[1], len(config.fanouts), config.hid_size, config.num_classes)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    #if "friendster" in config.graph_name:
    #    graph = graph.pin_memory_()
    #    mode = "uva"
    #else:
    #    graph = graph.to(rank)
    #    mode = "gpu"
    sample_config = SampleConfig(rank=rank, num_partition=config.num_partition, batch_size=config.batch_size,
                                 world_size=config.world_size, mode=mode, fanouts=config.fanouts, reindex=True,
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
    
    num_edges = 0
    for epoch in range(config.num_epoch):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
            batch_label = gather_pinned_tensor_rows(label, output_nodes)
            batch_pred = model(blocks, batch_feat)
            batch_loss = torch.nn.functional.cross_entropy(batch_pred, batch_label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            log_step(rank, epoch, step, step_per_epoch, timer)
            
    # if config.test_model_acc:
    t2 = Timer()
    print(f"testing model accuracy on {device}")
    model.eval()
    ys = []
    y_hats = []
    dataloader = SplitGraphLoader(graph, partition_map, train_idx, sample_config)
    for input_nodes, output_nodes, blocks in dataloader:
        with torch.no_grad():

            batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
            batch_label = gather_pinned_tensor_rows(label, output_nodes)

            ys.append(batch_label)
            batch_pred = model(blocks, batch_feat)
            y_hats.append(batch_pred)  
    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=config.num_classes)
    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    acc = round(acc.item() * 100 / config.world_size, 2)
    if rank == 0:
        print(f"test accuracy={acc}% in {t2.duration()} secs")
            
    ddp_exit()
