import torch
import os
import dgl
import time
import csv
import pandas as pd
import torch.distributed as dist
from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows

PREHEAT_STEP=100

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_exit():
    dist.destroy_process_group()

def get_memory_info(device=None, rd=0):
    if device == None:
        device = torch.cuda.current_device()
    allocated_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved(device) / 1024 / 1024
    allocated_mb = round(allocated_mb, rd)
    reserved_mb = round(reserved_mb, rd)
    return allocated_mb, reserved_mb

class Timer:
    def __init__(self):
        self.start = time.time()
    def duration(self, rd=3):
        return round(time.time() - self.start, rd)
    def reset(self):
        self.start = time.time()

class CudaTimer:
    def __init__(self, stream=torch.cuda.current_stream()):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream = stream
        self.end_recorded = False
        self.start_event.record(stream=self.stream)

    def start(self):
        self.start_event.record(stream=self.stream)
        
    def end(self):
        self.end_event.record(stream=self.stream)
        self.end_recorded = True
        
    def reset(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_recorded = False
        
    def duration(self):
        assert(self.end_recorded)
        self.start_event.synchronize()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)
        duration_s = duration_ms / 1000
        return duration_s
    
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
        
    def get_file_name(self):
        if "groot" not in self.system:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                     f"{len(self.fanouts)}x{self.fanouts[0]}_{self.cache_size}")
        else:
            return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
                    f"{len(self.fanouts)}x{self.fanouts[0]}_{self.num_redundant_layer}_{self.cache_size}")

    def header(self):
        return ["timestamp","machine_name", "graph_name", "world_size", "num_epoch", "fanouts", "num_redundant_layers", \
                "batch_size", "system", \
                    "model", "hid_size", "cache_size", "partition_type"]
    
    def content(self):
        connection = "_nvlink" if self.nvlink else "_pcie"
        machine_name = self.machine_name + connection
        return [pd.Timestamp('now'), machine_name, self.graph_name, self.world_size, self.num_epoch, self.fanouts, self.num_redundant_layer, \
                    self.batch_size, self.system, self.model, self.hid_size, self.cache_size, self.partition_type]

    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += f"num_classes={self.num_classes}"
        res += "\n"
        return res    

class Profiler:
    def __init__(self, duration: float, sampling_time : float, feature_time: float,\
                 forward_time: float, backward_time: float, test_acc):
        self.duration = duration
        self.sampling_time = sampling_time
        self.feature_time = feature_time
        self.forward_time = forward_time
        self.backward_time = backward_time
        self.test_acc = test_acc
        self.allocated_mb, self.reserved_mb = get_memory_info()
        self.edges_computed = 0
        self.edges_computed_min = 0
        self.edges_computed_max = 0
        self.edge_skew = 0
        self.run_time = 0
    def header(self):
        header = ["duration (s)", "sampling (s)", "feature (s)", "forward (s)", "backward (s)",\
                    "allocated (MB)", "reserved (MB)", "test accuracy %", "edges_computed", "edge_skew", "min_edge", "max_edge","run_time"]
        return header
    
    def content(self):
        content = [self.duration, self.sampling_time, self.feature_time, self.forward_time,\
                   self.backward_time, self.allocated_mb, self.reserved_mb, self.test_acc, \
                   self.edges_computed, self.edge_skew, self.edges_computed_min, self.edges_computed_max, self.run_time]
        return content
    
    def __repr__(self):
        res = ""
        header = self.header()
        content = self.content()
        for header, ctn in zip(header, content):
            res += f"{header}={ctn} | "
        res += "\n"
        return res

def profile_edge_skew(edges_computed: int, profiler:Profiler, rank:int, dist: any):
    profiler.edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed = sum(edges_computed)/len(edges_computed)
    edges_computed_max  = torch.tensor(edges_computed).to(rank)
    edges_computed_min  = torch.tensor(edges_computed).to(rank)
    edges_computed_avg  = torch.tensor(edges_computed).to(rank)
    dist.all_reduce(edges_computed_max, op = dist.ReduceOp.MAX)
    dist.all_reduce(edges_computed_min, op = dist.ReduceOp.MIN)
    dist.all_reduce(edges_computed_avg, op = dist.ReduceOp.SUM)
    
    profiler.edges_computed_min = edges_computed_min.item()
    profiler.edges_computed_max = edges_computed_max.item()
    profiler.edges_computed = edges_computed_avg.item() / 4
    profiler.edge_skew = (edges_computed_max.item() - edges_computed_min.item()) / profiler.edges_computed

def get_default_config(graph_name, system, log_path, data_dir, num_redundant_layer = 0):
    configs = []
    partitioning_graph = ""
    balancing="edge"
    training_nodes="xbal"
    for model in ["sage", "gat"]:
       config = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=5,
                       fanouts=[20,20,20],
                       batch_size=1024,
                       system=system,
                       model=model,
                       cache_size = 0,
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
       
       config.num_redundant_layer = num_redundant_layer
       config.partition_type = f"{partitioning_graph}_w4_{balancing}_{training_nodes}"
       configs.append(config)
    return configs

def empty_profiler():
    empty = -1
    profiler = Profiler(duration=empty, sampling_time=empty, feature_time=empty, forward_time=empty, backward_time=empty, test_acc=empty)
    return profiler

def oom_profiler():
    oom = "oom"
    profiler = Profiler(duration=oom, sampling_time=oom, feature_time=oom, forward_time=oom, backward_time=oom, test_acc=oom)
    return profiler


def get_duration(timers: list[CudaTimer], rb=3)->float:
    res = 0.0
    for timer in timers:
        res += timer.duration()
    return round(res, rb)

def write_to_csv(out_path, configs: list[Config], profilers: list[Profiler]):
    assert(len(configs) == len(profilers))
    def get_row(header, content):
        res = {}
        for k, v in zip(header, content):
            res[k] = v
        return res
    
    has_header = os.path.isfile(out_path)
    with open(out_path, 'a') as f:
        header = configs[0].header() + profilers[0].header()
        writer = csv.DictWriter(f, fieldnames=header)        
        if not has_header:
            writer.writeheader()
        for config, profiler in zip(configs, profilers):
            row = get_row(config.header() + profiler.header(), config.content() + profiler.content())
            writer.writerow(row)
    print("Experiment result has been written to: ", out_path)

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

def load_dgl_graph(in_dir, is32=False, wsloop=False, is_sym=False, load_edge_weight=False)->dgl.DGLGraph:
    indptr, indices, edges = load_graph(in_dir, is32, wsloop, is_sym, load_edge_weight)
    return dgl.graph(("csc", (indptr, indices, edges)))

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

def load_feat_label(in_dir) -> (torch.Tensor, torch.Tensor, int):
    feat = torch.load(os.path.join(in_dir, f"feat.pt"))
    label = torch.load(os.path.join(in_dir, f"label.pt"))
    num_labels = torch.unique(label).shape[0]
    return feat, label, num_labels

def gen_rand_feat(v_num, feat_dim):
    return torch.empty((v_num, feat_dim))

def gen_rand_label(v_num, num_classes):
    return torch.randint(low=0, high=num_classes, size=(v_num,))

def gen_feat_label(v_num, feat_dim, num_classes=10):
    feat = gen_rand_feat(v_num, feat_dim)
    label = gen_rand_label(v_num, num_classes)
    return feat, label, num_classes

def load_topo(config: Config, is_pinned=False):
    print("Start graph topology loading")
    t1 = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    wsloop = "gat" in config.model
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    graph = load_dgl_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym)
    train_idx, valid_idx, test_idx = load_idx_split(in_dir, is32=is32)
    if is_pinned:
        graph = graph.pin_memory_()
        
    return graph, train_idx, valid_idx, test_idx

def get_feat_dim(config: Config):
    if config.graph_name == "orkut":
        return 1280
    elif config.graph_name == "friendster":
        return 128
    elif config.graph_name == "papers100M":
        return 128
    elif config.graph_name == "products":
        return 100

def get_feat_bytes(feat:torch.Tensor):
    res = feat.element_size()
    for dim in feat.shape:
        res *= dim
    return res        

def get_feat_bytes_str(feat:torch.Tensor):
    feat_bytes = get_feat_bytes(feat)
    num_gb = round(feat_bytes / 1024 / 1024 / 1024)
    return f"{num_gb}GB"

def load_data(config: Config, is_pinned=False):
    print("Start data loading")
    t1 = Timer()
    in_dir = os.path.join(config.data_dir, config.graph_name)
    wsloop = "gat" in config.model
    is32 = False
    is_sym = config.graph_name in ["orkut", "friendster"]
    graph = load_dgl_graph(in_dir, is32=is32, wsloop=wsloop, is_sym=is_sym)
    train_idx, valid_idx, test_idx = load_idx_split(in_dir, is32=is32)    
    feat = None
    label = None
    num_label = None
    if config.graph_name in ["products"]:
        feat, label, num_label = load_feat_label(os.path.join(config.data_dir, config.graph_name))
    else:
        v_num = graph.num_nodes()
        feat_dim = get_feat_dim(config)
        feat = gen_rand_feat(v_num, feat_dim)
        num_label = 10
        label = gen_rand_label(v_num, 10)
        
    print(f"Data loading total time {t1.duration()} secs")
    
    if is_pinned:
        print("pining data")
        nd_feat = pin_memory_inplace(feat)
        nd_label = pin_memory_inplace(label)
        graph = graph.pin_memory_()
    return graph, feat, label, train_idx, valid_idx, test_idx, num_label

def log_step(rank, epoch, step, step_per_epoch, timer):
    if rank == 0 and (step % step_per_epoch) % 100 == 0:
        cur_step = step % step_per_epoch
        if cur_step == 0:
            cur_step = step_per_epoch
        print(f"{epoch=} {cur_step=} / {step_per_epoch} time={timer.duration()} secs")