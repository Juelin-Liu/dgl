from dgl.dev import CompactCSR, MakeSym, ExpandIndptr
from dgl.partition import metis_partition_assignment_capi
import torch, os, time, dgl

class Timer:
    def __init__(self):
        self.start = time.time()
    def duration(self, rd=3):
        return round(time.time() - self.start, rd)
    def reset(self):
        self.start = time.time()

class Config:
    def __init__(self, graph_name, world_size, num_epoch, fanouts,
                 batch_size, system, model, hid_size, cache_size, log_path, data_dir):
        try:
            self.machine_name = os.environ['MACHINE_NAME']
        except Exception as e:
            self.machine_name = "jupiter"
        self.graph_name = graph_name
        self.world_size = world_size
        self.num_epoch = num_epoch
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.system = system
        self.model = model
        self.cache_size = cache_size
        self.hid_size = hid_size
        self.log_path = log_path
        self.data_dir = data_dir
        self.num_redundant_layer = len(self.fanouts)
        self.nvlink = False
        self.in_feat = -1
        self.num_classes = -1
        self.partition_type = "edge_balanced"
        self.test_model_acc = False
        
def load_idx_split(in_dir, is32=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    # graph_name = in_dir.split("/")[-1]
    # if graph_name in ["orkut", "friendster"]:
    #     data_dir = "/data/juelin/project/scratch/dgl/experiment/dataset"
    #     in_dir = os.path.join(data_dir, graph_name)
    
    print("load idx split from", in_dir)
    train_idx = torch.load(os.path.join(in_dir, f"train_idx.pt"))
    valid_idx = torch.load(os.path.join(in_dir, f"valid_idx.pt"))
    test_idx = torch.load(os.path.join(in_dir, f"test_idx.pt"))
    if is32:
        return train_idx.type(torch.int32), valid_idx.type(torch.int32), test_idx.type(torch.int32)
    else:
        return train_idx, valid_idx, test_idx

def load_graph(in_dir, is32=False, wsloop=False, is_sym=False, load_edge_weight=False) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    symtype_str = "sym" if is_sym else "xsym"
    indptr = torch.load(os.path.join(in_dir, f"indptr_{symtype_str}.pt"))
    indices = torch.load(os.path.join(in_dir, f"indices_{symtype_str}.pt"))
    edges = torch.empty(0, dtype=indices.dtype)
    if load_edge_weight:
        edges = torch.load(os.path.join(in_dir, "edge_weight.pt")).type(torch.int64)
    if wsloop and is_sym == False:
        graph: dgl.DGLGraph = dgl.graph(("csc", (indptr, indices, edges)))
        graph = dgl.add_self_loop(graph)
        indptr, indices, edges = graph.adj_tensors("csc")
    if is32:
        return indptr.type(torch.int32), indices.type(torch.int32), edges.type(torch.int32)
    else:
        return indptr, indices, edges
    
def load_metis_graph(config:Config, node_mode: str, edge_mode: str):
    print("Start graph topology loading", flush=True)
    timer = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    wsloop = False
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    
    load_edge_weight = True if edge_mode == "freq" else False
    indptr, indices, edge_weight = load_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym, load_edge_weight=load_edge_weight)
    v_num = indptr.shape[0] - 1
    e_num = indices.shape[0]
    print(f"load graph topology in {timer.duration()} secs", flush=True)
    timer.reset()
    degree = indptr[1:] - indptr[:-1]
    if load_edge_weight:
        # print("prunning edges")
        flag = edge_weight > 0
        indices = indices[flag].clone()
        edge_weight = edge_weight[flag].clone()
        indptr = CompactCSR(indptr, flag.type(torch.uint8))
        remain_ratio = flag.sum() / flag.shape[0] * 100
        e_num = flag.sum()
        print(f"remove {round(100 - remain_ratio.item())}% edges in {timer.duration()} secs", flush=True)
        
    timer.reset()
    if is_sym == False or load_edge_weight:
        # remove self edges
        src = ExpandIndptr(indptr)
        flag = src != indices
        indices = indices[flag].clone()
        edge_weight = edge_weight[flag].clone()
        indptr = CompactCSR(indptr, flag.type(torch.uint8))
        print(f"aftre remove self edges in {timer.duration()} secs", flush=True)

        indptr, indices, edge_weight = MakeSym(indptr=indptr, indices=indices, data=edge_weight)
        graph = dgl.graph(("csr", (indptr, indices, edge_weight)))
        graph = dgl.remove_self_loop(graph)

        assert(edge_weight.min() > 0)
        print(f"convert graph org_enum={e_num} to sym_enum={graph.num_edges()} in {timer.duration()} secs", flush=True)
    
    graph = dgl.graph(("csr", (indptr, indices, edge_weight)))

    assert(graph.num_nodes() == v_num)
    node_weight = None
    if node_mode == "uniform":
        node_weight = torch.ones((v_num,), dtype=torch.int64)
    elif node_mode == "degree":
        node_weight = degree
    elif node_mode in ["src", "dst", "input"]:
        node_weight = torch.load(f"{in_dir}/{node_mode}_node_weight.pt")
        node_weight = node_weight * 8 + 1
    return node_weight.type(torch.int64), graph

def partition(config: Config, node_mode:str, edge_mode:str, bal: str):
    assert node_mode in ["uniform", "degree", "src", "dst", "input"]    
    assert edge_mode in ["uniform", "freq"]
    assert bal in ["bal", "xbal"]
    
    timer = Timer()
    node_weight, graph = load_metis_graph(config, node_mode, edge_mode)
    print(f"load graph in {timer.duration()} secs", flush=True)
    if bal == "bal":
        v_num = graph.num_nodes()
        vwgts = [node_weight]
        in_dir = os.path.join(config.data_dir, config.graph_name)
        train, valid, test = load_idx_split(in_dir)
        for target_idx in [train, valid, test]:
            wgt = torch.zeros((v_num,), dtype=torch.int64)
            wgt[target_idx] = node_weight[target_idx]
            vwgts.append(wgt)
        node_weight = torch.concat(vwgts).clone()
    assert(node_weight.is_contiguous())
    assignment = metis_partition_assignment_capi(sym_g=graph, k=config.world_size, vwgt=node_weight, mode="k-way", objtype="cut", use_edge_weight=edge_mode != "uniform")
    file_name = f"{config.graph_name}_w{config.world_size}_n{node_mode}_e{edge_mode}_{bal}.pt"
    out_dir = os.path.join(config.data_dir, "partition_ids", config.graph_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"saving file to {out_dir}/{file_name}", flush=True)
    torch.save(assignment, os.path.join(out_dir, file_name))