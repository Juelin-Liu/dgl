from dgl.dev import *
from dgl.partition import metis_partition_assignment_capi, make_symmetric
from node.utils import *

def load_metis_graph(config:Config, node_mode: str, edge_mode: str):
    print("Start graph topology loading")
    timer = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    wsloop = False
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    
    load_edge_weight = True if edge_mode == "freq" else False
    indptr, indices, edge_weight = load_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym, load_edge_weight=load_edge_weight)
    v_num = indptr.shape[0] - 1
    print(f"load graph topology in {timer.duration()} secs")
    timer.reset()
    
    if load_edge_weight:
        print("prunning edges")
        flag = edge_weight > 0
        indices = indices[flag].clone()
        edge_weight = edge_weight[flag].clone()
        indptr = CompactCSR(indptr, flag)
        remain_ratio = flag.sum() / flag.shape[0] * 100
        print(f"prunning {round(100 - remain_ratio)}% edges in {timer.duration()} secs")

    timer.reset()
    graph = dgl.graph(("csr", (indptr, indices, edge_weight)))
    if is_sym == False or load_edge_weight:
        graph = make_symmetric(graph)

    assert(graph.num_nodes() == v_num)
    node_weight = None
    if node_mode == "uniform":
        node_weight = torch.ones((v_num,), dtype=torch.int64)
    elif node_mode == "degree":
        node_weight = indptr[1:] - indptr[:-1]
    elif node_mode in ["src", "dst", "input"]:
        node_weight = torch.load(f"{in_dir}/{node_mode}_node_weight.pt")
    
    return node_weight.type(torch.int64), graph

def partition(config: Config, node_mode:str, edge_mode:str, bal: str):
    assert node_mode in ["uniform", "degree", "src", "dst", "input"]    
    assert edge_mode in ["uniform", "freq"]
    assert bal in ["bal", "xbal"]
    node_weight, graph = load_metis_graph(config, node_mode, edge_mode)
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
    print(f"saving file to {out_dir}/{file_name}")
    torch.save(assignment, os.path.join(out_dir, file_name))