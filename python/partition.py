from libra.util import load_dgl_graph
import dgl
import os, torch
from dgl.partition import metis_partition_assignment_v2

from libra.util import load_idx_split
from libra.timer import Timer
from torch.nn.functional import pad
import time

def get_samp_weight(graph: dgl.DGLGraph, train_idx: torch.Tensor):
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    device = 0    
    graph_sampler = dgl.dataloading.NeighborSampler(fanouts=[20,20,20])
    dataloader = dgl.dataloading.DataLoader(
        graph=graph,               # The graph
        indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
        graph_sampler=graph_sampler,     # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=False, # enable ddp if using mutiple gpus
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        use_uva=graph.is_pinned(),
        num_workers=0,
    )

    timer = Timer()
    epoch_num = 3
    num_edges = 0
    step = 0
    dwgt = torch.zeros(v_num, dtype=torch.int32).to(device)
    dflag = torch.zeros(e_num, dtype=torch.bool).to(device)

    print(f"start sampling")
    for epoch in range(epoch_num):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1            
            for layer, block in enumerate(blocks):
                eid = block.edata["_ID"]
                dflag[eid] = True
                num_edges += block.num_edges()
                src, dst = block.all_edges()
                gids = input_nodes[src]
                cnts = torch.bincount(gids)
                pad_size = v_num - cnts.shape[0]
                cnts = pad(cnts, (0, pad_size))
                dwgt += cnts
                
            if step % 10 == 0:
                print(f"{step=} {num_edges=}")
                
    print(f"duration = {timer.duration()} s")
    return dwgt / epoch_num, dflag

def metis(data_dir: str, graph_name: str, num_partitions = 4):
    in_dir = os.path.join(data_dir, graph_name)
    graph: dgl.DGLGraph = load_dgl_graph(in_dir, is32=False, wsloop=False)
    graph = dgl.remove_self_loop(graph)
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    
    print(f"{graph_name=} {v_num=} {e_num=}")
    cur_dir = os.getcwd()
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)
    for partype in ["samp", "node", "edge"]:
        for bal in ["xbal", "bal"]:
            print(f"start partition on {graph_name} using {partype} {bal}", flush=True)
            node_mask = torch.zeros((v_num), dtype=torch.int32)
            if bal == "bal":
                node_mask[train_idx] = 1
                node_mask[test_idx] = 2
                node_mask[valid_idx] = 3
            wgt = None
            if partype == "node":
                wgt = torch.ones(v_num, dtype=torch.int32)
            elif partype == "edge":
                wgt = graph.in_degrees()
            else:
                graph.create_formats_()
                graph.pin_memory_()
                dwgt, dflag = get_samp_weight(graph, train_idx)
                wgt = dwgt.to("cpu")
            partition_ids = metis_partition_assignment_v2(g=graph, k=num_partitions, mode="k-way", balance_ntypes = node_mask, wgt=wgt, objtype="vol")
            out_dir = os.path.join(cur_dir, "partition_ids")
            file_path = os.path.join(out_dir, f"{graph_name}_w{num_partitions}_{partype}_{bal}.pt")
            torch.save(partition_ids, file_path)

def pruned_metis(data_dir: str, graph_name: str, num_partitions = 4):
    print(f"pruned metis: {graph_name}")
    in_dir = os.path.join(data_dir, graph_name)
    graph: dgl.DGLGraph = load_dgl_graph(in_dir, is32=False, wsloop=False)
    graph = dgl.remove_self_loop(graph)
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    print(f"{graph_name=} {v_num=} {e_num=}", flush=True)
    cur_dir = os.getcwd()
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)
    graph.create_formats_()
    graph.pin_memory_()
    dwgt, dflag = get_samp_weight(graph, train_idx)
    flag = dflag.to("cpu")
    src, dst = graph.all_edges()
    src = src[flag]
    dst = dst[flag]
    graph = dgl.graph(data=(src, dst), num_nodes=v_num)
    for partype in ["samp", "node", "edge"]:
        for bal in ["xbal", "bal"]:
            print(f"start partition on {graph_name} using {partype} {bal}", flush=True)
            start = time.time()
            node_mask = torch.zeros((v_num), dtype=torch.int32)
            if bal == "bal":
                node_mask[train_idx] = 1
                node_mask[test_idx] = 2
                node_mask[valid_idx] = 3
            wgt = None
            if partype == "node":
                wgt = torch.ones(v_num, dtype=torch.int32)
            elif partype == "edge":
                wgt = graph.in_degrees()
            else:
                wgt = dwgt.to("cpu")
            partition_ids = metis_partition_assignment_v2(g=graph, k=num_partitions, mode="k-way", balance_ntypes = node_mask, wgt=wgt, objtype="vol")
            out_dir = os.path.join(cur_dir, "partition_ids")
            file_path = os.path.join(out_dir, f"pruned_{graph_name}_w{num_partitions}_{partype}_{bal}.pt")
            torch.save(partition_ids, file_path)
            print(f"Partition {graph_name} using {partype} {bal} into {num_partitions} partitions using: {round(time.time() - start)} seconds")

dataset = {
    "ogbn-products":   "/data/juelin/dataset/OGBN/processed",
    "ogbn-papers100M": "/data/juelin/dataset/OGBN/processed",
    "com-orkut": "/data/juelin/dataset/SNAP/",
    "com-friendster": "/data/juelin/dataset/SNAP/"
}

for graph_name, data_dir in dataset.items():
    pruned_metis(data_dir, graph_name)
    
# for graph_name, data_dir in dataset.items():
#     metis(data_dir, graph_name)