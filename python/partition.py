from libra.util import load_dgl_graph
import dgl
import os, torch
from dgl.partition import metis_partition_assignment, metis_partition_assignment_v2, metis_partition_assignment_v3
from dgl.dev import Increment, COO2CSR

from libra.util import load_idx_split
from libra.timer import Timer
from torch.nn.functional import pad
import time
import gc

# def load_freqcnt(graph_name):
#     cnt_dir = os.getcwd()
#     eflag_path = os.path.join(cnt_dir, f"freqcnts/{graph_name}_eflag.pt")
#     efreq_path = os.path.join(cnt_dir, f"freqcnts/{graph_name}_efreq.pt")
#     eflag = torch.load(eflag_path)
#     efreq = torch.load(efreq_path)
#     return eflag, efreq

def get_access_threshold(efreq, t):
    e_num = efreq.shape[0]
    agg_efreq = torch.bincount(efreq)
    index = torch.arange(agg_efreq.shape[0])
    cum_efreq_rate = torch.cumsum(agg_efreq, dim = 0) / e_num
    wcum_efreq = torch.cumsum(agg_efreq * index, dim=0)
    wcum_efreq_rate = wcum_efreq / e_num
    for i in range(wcum_efreq_rate.shape[0]):
        if wcum_efreq_rate[i] > t:
            return i, cum_efreq_rate[i]
        
def get_samp_weight(graph: dgl.DGLGraph, train_idx: torch.Tensor):
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    device = "cuda:3"  
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
    epoch_num = 10
    num_edges = 0
    step = 0
    dvwgt = torch.zeros(v_num, dtype=train_idx.dtype).to(device)
    dewgt = torch.zeros(e_num, dtype=train_idx.dtype).to(device)
    dflag = torch.zeros(e_num, dtype=torch.bool).to(device)
    print(f"start sampling", flush=True)
    for epoch in range(epoch_num):
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1            
            for layer, block in enumerate(blocks):
                eid = block.edata["_ID"]
                dflag[eid] = True
                Increment(dewgt, eid)
                num_edges += block.num_edges()
                src, dst = block.all_edges()
                gids = input_nodes[src]
                cnts = torch.bincount(gids)
                pad_size = v_num - cnts.shape[0]
                cnts = pad(cnts, (0, pad_size))
                dvwgt += cnts
                
            if step % 500 == 0:
                print(f"{step=} {num_edges=}", flush=True)
                
    print(f"duration = {timer.duration()} s", flush=True)
    return dvwgt.to("cpu"), dewgt.to("cpu"), dflag.to("cpu")

def metis(data_dir: str, graph_name: str, num_partitions = 4):
    in_dir = os.path.join(data_dir, graph_name)
    graph: dgl.DGLGraph = load_dgl_graph(in_dir, is32=False, wsloop=False)
    v_num = graph.num_nodes()
    e_num = graph.num_edges()
    
    print(f"{graph_name=} {v_num=} {e_num=}")
    cur_dir = os.getcwd()
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)
    for partype in ["edge"]:
        for bal in ["xbal", "bal"]:
            print(f"start partition on {graph_name} using {partype} {bal}", flush=True)
            node_mask = torch.zeros((v_num), dtype=torch.int64)
            if bal == "bal":
                node_mask[train_idx] = 1
                node_mask[test_idx] = 2
                node_mask[valid_idx] = 3
            wgt = None
            partition_ids = None
            if partype == "node":
                partition_ids = metis_partition_assignment(g=graph, k=num_partitions, mode="k-way", balance_ntypes=node_mask,balance_edges=False, objtype="vol")
            elif partype == "edge":
                partition_ids = metis_partition_assignment(g=graph, k=num_partitions, mode="k-way", balance_ntypes=node_mask,balance_edges=True, objtype="vol")
            else:
                graph.create_formats_()
                graph.pin_memory_()
                dvwgt, _, _ = get_samp_weight(graph, train_idx)
                wgt = dvwgt.to("cpu")
                partition_ids = metis_partition_assignment_v2(g=graph, k=num_partitions, mode="k-way", balance_ntypes = node_mask, wgt=wgt, objtype="vol")
            out_dir = os.path.join(cur_dir, "partition_ids")
            file_path = os.path.join(out_dir, f"{graph_name}_w{num_partitions}_{partype}_{bal}.pt")
            torch.save(partition_ids, file_path)

# def pruned_metis(data_dir: str, graph_name: str, num_partitions = 4):
#     print(f"pruned metis: {graph_name}")
#     in_dir = os.path.join(data_dir, graph_name)
#     graph: dgl.DGLGraph = load_dgl_graph(in_dir, is32=False, wsloop=False)
#     graph = dgl.remove_self_loop(graph)
#     v_num = graph.num_nodes()
#     e_num = graph.num_edges()
#     print(f"{graph_name=} {v_num=} {e_num=}", flush=True)
#     cur_dir = os.getcwd()
#     train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)
#     graph.create_formats_()
#     graph.pin_memory_()
#     vwgt, _, flag = get_samp_weight(graph, train_idx)
#     src, dst = graph.all_edges()
#     src = src[flag]
#     dst = dst[flag]
#     graph = dgl.graph(data=(src, dst), num_nodes=v_num)
#     for partype in ["samp", "node", "edge"]:
#         for bal in ["xbal", "bal"]:
#             print(f"start partition on {graph_name} using {partype} {bal}", flush=True)
#             start = time.time()
#             node_mask = torch.zeros((v_num), dtype=torch.int64)
#             if bal == "bal":
#                 node_mask[train_idx] = 1
#                 node_mask[test_idx] = 2
#                 node_mask[valid_idx] = 3
#             wgt = None
#             if partype == "node":
#                 wgt = torch.ones(v_num, dtype=torch.int64)
#             elif partype == "edge":
#                 wgt = graph.in_degrees().type(torch.int64)
#             else:
#                 wgt = vwgt.type(torch.int64)
#             partition_ids = metis_partition_assignment_v2(g=graph, k=num_partitions, mode="k-way", balance_ntypes = node_mask, wgt=wgt, objtype="vol")
#             out_dir = os.path.join(cur_dir, "partition_ids")
#             file_path = os.path.join(out_dir, f"pruned_{graph_name}_w{num_partitions}_{partype}_{bal}.pt")
#             torch.save(partition_ids, file_path)
#             print(f"Partition {graph_name} using {partype} {bal} into {num_partitions} partitions using: {round(time.time() - start)} seconds")

# def tpruned_metis(data_dir: str, graph_name: str, num_partitions = 4):
#     print(f"tpruned metis: {graph_name}")
#     in_dir = os.path.join(data_dir, graph_name)
#     org_graph: dgl.DGLGraph = load_dgl_graph(in_dir, is32=False, wsloop=False)
#     v_num = org_graph.num_nodes()
#     e_num = org_graph.num_edges()
#     print(f"{graph_name=} {v_num=} {e_num=}", flush=True)
#     cur_dir = os.getcwd()
#     train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=False)
#     org_graph.create_formats_()
#     org_graph.pin_memory_()
#     t_list = [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]
#     vwgt, ewgt, _ = get_samp_weight(org_graph, train_idx)
#     org_src, org_dst = org_graph.all_edges()

#     for t in t_list:
#         i, c = get_access_threshold(ewgt, t)
#         flag = ewgt >= i
#         src = org_src[flag]
#         dst = org_dst[flag]
#         cur_graph = dgl.graph(data=(src, dst), num_nodes=v_num)
#         pruned_percentage = c * 100
#         for partype in ["samp", "node", "edge"]:
#             for bal in ["xbal", "bal"]:
#                 print(f"start partitioning {graph_name} using {partype} {bal} with {round(pruned_percentage.item(), 1)} % edges pruned", flush=True)
#                 start = time.time()
#                 node_mask = torch.zeros((v_num), dtype=torch.int64)
#                 if bal == "bal":
#                     node_mask[train_idx] = 1
#                     node_mask[test_idx] = 2
#                     node_mask[valid_idx] = 3
#                 wgt = None
#                 if partype == "node":
#                     wgt = torch.ones(v_num, dtype=torch.int64)
#                 elif partype == "edge":
#                     wgt = cur_graph.in_degrees().type(torch.int64)
#                 else:
#                     wgt = vwgt.type(torch.int64)
#                 partition_ids = metis_partition_assignment_v2(g=cur_graph, k=num_partitions, mode="k-way", balance_ntypes = node_mask, wgt=wgt, objtype="vol")
#                 out_dir = os.path.join(cur_dir, "partition_ids")
#                 file_path = os.path.join(out_dir, f"t{t}_{graph_name}_w{num_partitions}_{partype}_{bal}.pt")
#                 torch.save(partition_ids, file_path)
#                 print(f"Partition {graph_name} using {partype} {bal} into {num_partitions} partitions using: {round(time.time() - start)} seconds")
                   
                   
def wprune_metis(data_dir: str, graph_name: str, num_partitions = 4):
    print(f"wpruned metis: {graph_name}")
    in_dir = os.path.join(data_dir, graph_name)
    cur_dir = os.getcwd()
    org_graph: dgl.DGLGraph = load_dgl_graph(in_dir, is32=True, wsloop=False)
    v_num = org_graph.num_nodes()
    e_num = org_graph.num_edges()
    print(f"{graph_name=} {v_num=} {e_num=}", flush=True)
    train_idx, test_idx, valid_idx = load_idx_split(in_dir, is32=True)
    org_graph.create_formats_()
    org_graph.pin_memory_()

    vwgt, ewgt, eflag = get_samp_weight(org_graph, train_idx)
    org_graph = org_graph.to("cpu")
    org_src, org_dst = org_graph.all_edges()
    src = org_src[eflag].clone()
    dst = org_dst[eflag].clone()
    cur_ewgt = ewgt[eflag].clone()
    del org_graph
    del org_src
    del org_dst
    del ewgt
    
    # src = src.to(torch.int64)
    # dst = dst.to(torch.int64)
    # cur_ewgt = cur_ewgt.to(torch.int64)
    degree, indptr, indices, data = COO2CSR(v_num=v_num, src=src, dst=dst, data=cur_ewgt)
    del src
    del dst
    del cur_ewgt
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    assert(indices.shape[0] == data.shape[0])
    cur_vwgt = vwgt.to(torch.int64)
    percentage = indices.shape[0] / 2 / e_num * 100
    for partype in ["samp", "node", "edge"]:
        for bal in ["xbal", "bal"]:
            print(f"start partitioning {graph_name} using {partype} {bal} {round(percentage, 1)} % edges", flush=True)
            start = time.time()
            node_mask = torch.zeros((v_num), dtype=torch.int64)
            if bal == "bal":
                node_mask[train_idx] = 1
                node_mask[test_idx] = 2
                node_mask[valid_idx] = 3
            wgt = None
            if partype == "node":
                wgt = torch.ones(v_num, dtype=torch.int64)
            elif partype == "edge":
                wgt = degree
            else:
                wgt = cur_vwgt
            partition_ids = metis_partition_assignment_v3(indptr=indptr, indices=indices, k=num_partitions, mode="k-way", balance_ntypes=node_mask, wgt=wgt, ewgt=data, objtype="vol")
            out_dir = os.path.join(cur_dir, "partition_ids")
            file_path = os.path.join(out_dir, f"wpruned_{graph_name}_w{num_partitions}_{partype}_{bal}.pt")
            torch.save(partition_ids, file_path)
            print(f"Partition {graph_name} using {partype} {bal} into {num_partitions} partitions using: {round(time.time() - start)} seconds")
                   
dataset = {
    "com-friendster": "/data/juelin/dataset/SNAP/",
    # "com-orkut": "/data/juelin/dataset/SNAP/",
    # "ogbn-papers100M": "/data/juelin/dataset/OGBN/processed",
    # "ogbn-products":   "/data/juelin/dataset/OGBN/processed",
}

for graph_name, data_dir in dataset.items():
    wprune_metis(data_dir, graph_name)
    
# for graph_name, data_dir in dataset.items():
#     metis(data_dir, graph_name)