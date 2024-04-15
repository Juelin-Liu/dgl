import torch
import os
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dev import LoadSNAP
import os
    
def prep_snap_graph(in_dir, out_dir, filename, to_sym):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    in_path = f"{in_dir}/{filename}"
    indptr, indices = LoadSNAP(in_path, to_sym)
    if to_sym:
        torch.save(indptr, f"{out_dir}/indptr_sym.pt")
        torch.save(indices, f"{out_dir}/indices_sym.pt")
    else:
        torch.save(indptr, f"{out_dir}/indptr_xsym.pt")
        torch.save(indices, f"{out_dir}/indices_xsym.pt")
        
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
    torch.save(indptr, f"{out_dir}/indptr_xsym.pt")
    torch.save(indices, f"{out_dir}/indices_xsym.pt")
    
    # generate idx split
    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"]
    valid_idx = idx_split["valid"]
    test_idx = idx_split["test"]
    
    torch.save(train_idx, os.path.join(out_dir, f"train_idx.pt"))
    torch.save(valid_idx, os.path.join(out_dir, f"valid_idx.pt"))
    torch.save(test_idx, os.path.join(out_dir, f"test_idx.pt"))
    
    node_labels: torch.Tensor = dataset[0][1]
    node_labels = node_labels.flatten().clone()
    torch.nan_to_num_(node_labels, nan=0.0)
    node_labels: torch.Tensor = node_labels.type(torch.int64)
    torch.save(node_labels, os.path.join(out_dir, "label.pt"))

def gen_rand_feat(v_num, feat_dim):
    return torch.empty((v_num, feat_dim))

def gen_rand_label(v_num, num_classes):
    return torch.randint(low=0, high=num_classes, size=(v_num,))

def generate_idx_split(v_num, ratio, out_dir):
    # generate idx split
    num_train = int(v_num * ratio)
    num_val   = int(v_num * 0.5 * ratio)
    num_test  = int(v_num * 0.5 * ratio)
    rand_idx  = torch.randperm(v_num)
    train_idx = rand_idx[0 : num_train].clone()
    test_idx  = rand_idx[num_train : num_train + num_test].clone()
    val_idx   = rand_idx[num_train + num_test : num_train + num_test + num_val].clone()
    
    os.makedirs(f"{out_dir}", exist_ok=True)
    torch.save(train_idx, f"{out_dir}/train_idx.pt")
    torch.save(test_idx,  f"{out_dir}/test_idx.pt")
    torch.save(val_idx,   f"{out_dir}/valid_idx.pt")
    
if __name__ ==  "__main__":
    data_dir = "./dataset"
    try:
        data_dir = os.environ["data_dir"]
    except:
        data_dir = "/data/gsplit"
        
    print(f"{data_dir=}")
    
    for graph_name in ["orkut", "friendster"]:
        filedir = os.path.join(data_dir, graph_name)
        prep_snap_graph(in_dir=filedir, out_dir=filedir, filename=f"{graph_name}.txt", to_sym=True)

    for graph_name in ["ogbn-products", "ogbn-papers100M"]:
        prep_ogbn_graph(in_dir=data_dir, out_dir=data_dir, graph_name=graph_name)
