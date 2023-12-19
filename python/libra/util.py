import os, dgl, torch, time, csv, gc
from ogb.nodeproppred import DglNodePropPredDataset
import pandas as pd
import numpy as np

def _build_dgl_graph(indptr, indices, edges) -> dgl.DGLGraph:
    graph = dgl.graph(("csc", (indptr, indices, edges)))
    return graph

def preprocess(graph_name, in_dir, out_dir) -> None:
    out_dir = os.path.join(out_dir, graph_name)
    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)
    
    id_type = torch.int64
    idtype_str = "64"
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    graph = dataset[0][0]
    if graph_name == "ogbn-proteins":
        feat = graph.edata.pop("feat")
        torch.save(feat, os.path.join(out_dir, "feat.pt"))
        species = graph.ndata["species"]
        torch.save(species, os.path.join(out_dir, "species.pt"))
    else:
        feat: torch.Tensor = graph.dstdata.pop("feat")
        torch.save(feat, os.path.join(out_dir, "feat.pt"))
        del feat

    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    
    torch.save(node_labels, os.path.join(out_dir, "label.pt"))

    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"].type(id_type)
    valid_idx = idx_split["valid"].type(id_type)
    test_idx = idx_split["test"].type(id_type)

    ntype = torch.zeros(graph.num_nodes(), dtype = torch.int64)
    count = 0
    for k in ["train", "valid", "test"]:
        ids = idx_split[k].type(id_type)
        ntype[ids] = count
        count  = count + 1
    torch.save(ntype, os.path.join(out_dir, f'ntype.pt'))


    torch.save(train_idx, os.path.join(out_dir, f"train_idx_{idtype_str}.pt"))
    torch.save(valid_idx, os.path.join(out_dir, f"valid_idx_{idtype_str}.pt"))
    torch.save(test_idx, os.path.join(out_dir, f"test_idx_{idtype_str}.pt"))

    indptr, indices, edges = graph.adj_tensors("csc")
    indptr = indptr.type(id_type)
    indices = indices.type(id_type)
    edges = edges.type(id_type)
    
    torch.save(indptr, os.path.join(out_dir, f"indptr_{idtype_str}.pt"))
    torch.save(indices, os.path.join(out_dir, f"indices_{idtype_str}.pt"))
    torch.save(edges, os.path.join(out_dir, f"edges_{idtype_str}.pt"))
    add_self_loop(out_dir, out_dir)

def add_self_loop(in_dir, out_dir=None):
    id_type = torch.int64
    idtype_str = "64"
    graph = load_dgl_graph(in_dir)
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    indptr, indices, edges = graph.adj_tensors("csc")
    indptr = indptr.type(id_type)
    indices = indices.type(id_type)
    edges = edges.type(id_type)
    if out_dir == None:
        out_dir = in_dir
    torch.save(indptr, os.path.join(out_dir, f"indptr_{idtype_str}_wsloop.pt"))
    torch.save(indices, os.path.join(out_dir, f"indices_{idtype_str}_wsloop.pt"))
    torch.save(edges, os.path.join(out_dir, f"edges_{idtype_str}_wsloop.pt"))
    
def load_graph(in_dir, is32=False, wsloop=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    idtype_str = "64"
    indptr = None
    indices = None
    if not wsloop: 
        indptr = torch.load(os.path.join(in_dir, f"indptr_{idtype_str}.pt"))
        indices = torch.load(os.path.join(in_dir, f"indices_{idtype_str}.pt"))
        # edges = torch.load(os.path.join(in_dir, f"edges_{idtype_str}.pt"))
        edges = torch.empty(0, dtype=indices.dtype)
    else:
        # with self loop
        indptr = torch.load(os.path.join(in_dir, f"indptr_{idtype_str}_wsloop.pt"))
        indices = torch.load(os.path.join(in_dir, f"indices_{idtype_str}_wsloop.pt"))
        # edges = torch.load(os.path.join(in_dir, f"edges_{idtype_str}_wsloop.pt"))
        edges = torch.empty(0, dtype=indices.dtype)
    if is32:
        return indptr.type(torch.int32), indices.type(torch.int32), edges.type(torch.int32)
    else:
        return indptr, indices, edges

def load_partition_map(in_dir, world_size=4, edge_balanced=False, is32=False):
    partype_str = "edge" if edge_balanced else "node"
    partition_map = torch.load(os.path.join(in_dir, f"partition_map_{partype_str}_w4.pt"))
    if is32:
        return partition_map.type(torch.int32)
    else:
        return partition_map.type(torch.int64)
    
def load_idx_split(in_dir, is32=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    idtype_str = "64"
    train_idx = torch.load(os.path.join(in_dir, f"train_idx_{idtype_str}.pt"))
    valid_idx = torch.load(os.path.join(in_dir, f"valid_idx_{idtype_str}.pt"))
    test_idx = torch.load(os.path.join(in_dir, f"test_idx_{idtype_str}.pt"))
    if is32:
        return train_idx.type(torch.int32), valid_idx.type(torch.int32), test_idx.type(torch.int32)
    else:
        return train_idx, valid_idx, test_idx

def load_feat_label(in_dir) -> (torch.Tensor, torch.Tensor, int):
    feat = torch.load(os.path.join(in_dir, f"feat.pt"))
    label = torch.load(os.path.join(in_dir, f"label.pt"))
    num_labels = torch.unique(label).shape[0]
    return feat, label, num_labels

def load_dgl_graph(in_dir, is32=False, wsloop=False) -> dgl.DGLGraph:
    indptr, indices, edges = load_graph(in_dir, is32, wsloop)
    graph = _build_dgl_graph(indptr, indices, edges)
    if is32:
        return graph.int()
    else:
        return graph
    
def get_dataset(graph_name, in_dir):
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    return dataset

def get_metis_partition(in_dir, v_num, config):
    assert config.partition_type in ["edge_balanced", "node_balanced", "random"]
    if config.partition_type == "random":
        return torch.randint(0, config.world_size, (v_num,), dtype = torch.int32)
    if config.partition_type == "edge_balanced":
        return torch.load(f'{in_dir}/partition_map_edge_w{config.world_size}.pt').to(torch.int32)
    if config.partition_type == "node_balanced":
        return torch.load(f'{in_dir}/partition_map_node_w{config.world_size}.pt').to(torch.int32)

def get_dgl_sampler(graph: dgl.DGLGraph, train_idx: torch.Tensor, graph_samler: dgl.dataloading.Sampler, system:str = "cpu", batch_size:int=1024, use_dpp=False) -> dgl.dataloading.dataloader.DataLoader:
    device = torch.cuda.current_device()
    dataloader = None
    drop_last = True
    
    if device == torch.cuda.device(0):
        print(f"before dataloader init graph formats: {graph.formats()}")

    if system == "cpu":
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx,         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device="cpu",      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=False,
            num_workers=1,
        )
    elif "uva" in system:
        graph.pin_memory_()
        assert(graph.is_pinned())
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=True,
            num_workers=0,
        )
    elif "gpu" in system:
        graph = graph.to(device)
        dataloader = dgl.dataloading.DataLoader(
            graph=graph,               # The graph
            indices=train_idx.to(device),         # The node IDs to iterate over in minibatches
            graph_sampler=graph_samler,     # The neighbor sampler
            device=device,      # Put the sampled MFGs on CPU or GPU
            use_ddp=use_dpp, # enable ddp if using mutiple gpus
            # The following arguments are inherited from PyTorch DataLoader.
            batch_size=batch_size,    # Batch size
            shuffle=True,       # Whether to shuffle the nodes for every epoch
            drop_last=drop_last,    # Whether to drop the last incomplete batch
            use_uva=False,
            num_workers=0,
        )
    if device == torch.cuda.device(0):
        print(f"after dataloader init graph formats: {graph.formats()}")
    return dataloader, graph

def get_memory_info(device=torch.cuda.current_device(), rd=0):
    allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
    allocated_mb = round(allocated_mb, rd)
    reserved_mb = round(reserved_mb, rd)
    return allocated_mb, reserved_mb