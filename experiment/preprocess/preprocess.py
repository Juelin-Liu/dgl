import torch
import os
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dev import LoadSNAP

def load_graph(in_dir, is32=False, wsloop=False, is_sym=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    symtype_str = "sym" if is_sym else "xsym"
    indptr = torch.load(os.path.join(in_dir, f"indptr_64_{symtype_str}.pt"))
    indices = torch.load(os.path.join(in_dir, f"indices_64_{symtype_str}.pt"))
    edges = torch.empty(0, dtype=indices.dtype)
    if wsloop and not is_sym:
        graph: dgl.DGLGraph = dgl.graph(("csc", (indptr, indices, edges)))
        graph = dgl.add_self_loop(graph)
        indptr, indices, edges = graph.adj_tensors("csc")
    if is32:
        return indptr.type(torch.int32), indices.type(torch.int32), edges.type(torch.int32)
    else:
        return indptr, indices, edges

def load_idx_split(in_dir, is32=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    train_idx = torch.load(os.path.join(in_dir, f"train_idx_64.pt"))
    valid_idx = torch.load(os.path.join(in_dir, f"valid_idx_64.pt"))
    test_idx = torch.load(os.path.join(in_dir, f"test_idx_64.pt"))
    if is32:
        return train_idx.type(torch.int32), valid_idx.type(torch.int32), test_idx.type(torch.int32)
    else:
        return train_idx, valid_idx, test_idx
    
def prep_snap_graph(in_dir, out_dir, filename, to_sym):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    in_path = f"{in_dir}/{filename}"
    indptr, indices = LoadSNAP(in_path, to_sym)
    if to_sym:
        torch.save(indptr, f"{out_dir}/indptr_64_sym.pt")
        torch.save(indices, f"{out_dir}/indices_64_sym.pt")
    else:
        torch.save(indptr, f"{out_dir}/indptr_64_xsym.pt")
        torch.save(indices, f"{out_dir}/indices_64_xsym.pt")
        
def prep_ogbn_graph(in_dir, out_dir, graph_name):
    assert(graph_name in ["ogbn-products", "ogbn-papers100M"])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataset = DglNodePropPredDataset(graph_name, in_dir)
    graph = dataset[0][0]
    
    feat: torch.Tensor = graph.dstdata.pop("feat")
    torch.save(feat, os.path.join(out_dir, "feat.pt"))
    del feat
        
    indptr, indices, _ = graph.adj_tensors("csc")
    torch.save(indptr, f"{out_dir}/indptr_64_xsym.pt")
    torch.save(indices, f"{out_dir}/indices_64_xsym.pt")
    
    # generate idx split
    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"]
    valid_idx = idx_split["valid"]
    test_idx = idx_split["test"]
    
    torch.save(train_idx, os.path.join(out_dir, f"train_idx_64.pt"))
    torch.save(valid_idx, os.path.join(out_dir, f"valid_idx_64.pt"))
    torch.save(test_idx, os.path.join(out_dir, f"test_idx_64.pt"))
    
    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    torch.save(node_labels, os.path.join(out_dir, "label.pt"))

def gen_rand_feat(v_num, hid_size):
    return torch.empty((v_num, hid_size))

def gen_rand_label(v_num, num_classes):
    return torch.randint(low=0, high=num_classes, size=(v_num,))

def generate_idx_split(v_num, out_dir):
    # generate idx split
    num_train = int(v_num * 0.1)
    num_val   = int(v_num * 0.05)
    num_test  = int(v_num * 0.05)
    rand_idx  = torch.randperm(v_num)
    train_idx = rand_idx[0 : num_train].clone()
    test_idx  = rand_idx[num_train : num_train + num_test].clone()
    val_idx   = rand_idx[num_train + num_test : num_train + num_test + num_val].clone()
    torch.save(train_idx, f"{out_dir}/train_idx_64.pt")
    torch.save(test_idx,  f"{out_dir}/test_idx_64.pt")
    torch.save(val_idx,   f"{out_dir}/valid_idx_64.pt")