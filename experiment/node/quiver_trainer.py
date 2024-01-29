import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
import quiver
from node.quiver_sampler import *
from node.model import *
from node.utils import *
from dgl.dev import *

def bench_quiver_batch(configs: list[Config]):
    for config in configs:
        assert(config.system == "quiver")
        assert(config.graph_name == configs[0].graph_name)
        assert(config.model == configs[0].model)
    
    device_list = [i for i in range(config.world_size)]
    quiver.init_p2p(device_list=device_list)    
    
    graph, feat, label, train_idx, valid_idx, test_idx, num_label = load_data(configs[0], is_pinned=False)
    indptr, indices, edges = graph.adj_tensors("csc")

    csr_topo = quiver.CSRTopo(indptr=indptr, indices=indices)    
    cache_policy = "p2p_clique_replicate" if config.nvlink else "device_replicate"
    device_cache_size = config.cache_size # TODO: dynamically determine cache size
    sampling_mode = "UVA"
    quiver_feat = quiver.Feature(rank=0, device_list=device_list, device_cache_size=device_cache_size, cache_policy=cache_policy, csr_topo=csr_topo)
    quiver_feat.from_cpu_tensor(feat)
    quiver_sampler = quiver.GraphSageSampler(csr_topo=csr_topo, sizes=config.fanouts, device=0, mode=sampling_mode)
    del feat
    gc.collect()
    
    for config in configs:
        config.num_classes = num_label
        try:
            spawn(train_quiver_ddp, args=(config, quiver_sampler, quiver_feat, label, num_label, train_idx, test_idx), nprocs=config.world_size)
        except Exception as e:
            print(f"error encountered with {config=}:", e)
            if "out of memory"in str(e):
                write_to_csv(config.log_path, [config], [oom_profiler()])
            else:
                write_to_csv(config.log_path, [config], [empty_profiler()])
                with open(f"exceptions/{config.get_file_name()}",'w') as fp:
                    fp.write(str(e))
        gc.collect()
        torch.cuda.empty_cache()
            
def train_quiver_ddp(rank: int, config: Config, quiver_sampler: quiver.pyg.GraphSageSampler, feat: quiver.Feature, label: torch.Tensor, num_label: int, train_idx: torch.Tensor, test_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    if rank == 0:
        print(config)
    config.in_feat = feat.shape[1]
    dataloader = QuiverDglSageSample(rank, world_size=config.world_size, batch_size=config.batch_size, target_idx=train_idx, sampler=quiver_sampler)

    model = None
    if config.model == "gat":
        model = Gat(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label, num_heads=4)
    elif config.model == "sage":
        model = Sage(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"pre-heating model on device: {device}")
    label = label.to(device)
    
    step = 0
    for input_nodes, output_nodes, blocks in dataloader:
        step += 1
        batch_feat = feat[input_nodes]
        batch_label = label[output_nodes]
        batch_pred = model(blocks, batch_feat)
        batch_loss = torch.nn.functional.cross_entropy(batch_pred, batch_label)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if step > PREHEAT_STEP:
            dataloader.reset()
            break
    dist.barrier()
    print(f"training model on device: {device}")        
    timer = Timer()
    sampling_timers = []
    feature_timers = []
    forward_timers = []
    backward_timers = []
    edges_computed = []
    step_per_epoch = dataloader.target_idx.shape[0] // dataloader.batch_size + 1
    step = 0
    for epoch in range(config.num_epoch):
        edges_computed_epoch = 0
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            sampling_timer.end()
            
            feat_timer = CudaTimer()
            batch_feat = feat[input_nodes]
            batch_label = label[output_nodes]
            dist.barrier()
            feat_timer.end()            
            
            forward_timer = CudaTimer()
            for block in blocks:
                block:dgl.DGLGraph
                edges_computed_epoch += block.num_edges()
            batch_pred = model(blocks, batch_feat)
            batch_loss = torch.nn.functional.cross_entropy(batch_pred, batch_label)
            forward_timer.end()
            
            backward_timer = CudaTimer()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            backward_timer.end()
            

            sampling_timers.append(sampling_timer)
            feature_timers.append(feat_timer)
            forward_timers.append(forward_timer)
            backward_timers.append(backward_timer)

            sampling_timer = CudaTimer()
            
            log_step(rank, epoch, step, step_per_epoch, timer)
                
        edges_computed.append(edges_computed_epoch)
    torch.cuda.synchronize()
    duration = timer.duration()
    sampling_time = get_duration(sampling_timers)
    feature_time = get_duration(feature_timers)
    forward_time = get_duration(forward_timers)
    backward_time = get_duration(backward_timers)
    profiler = Profiler(duration=duration, sampling_time=sampling_time, feature_time=feature_time, forward_time=forward_time, backward_time=backward_time, test_acc=0)
    profile_edge_skew(edges_computed, profiler, rank, dist)
    
    dist.barrier()
    
    if config.test_model_acc:
        t2 = Timer()
        print(f"testing model accuracy on {device}")
        model.eval()
        ys = []
        y_hats = []
        dataloader = QuiverDglSageSample(rank=rank, world_size=config.world_size, batch_size=config.batch_size, target_idx=test_idx, sampler=quiver_sampler)
        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():
                batch_feat = feat[input_nodes]
                batch_label = label[output_nodes]
                ys.append(batch_label)
                batch_pred = model(blocks, batch_feat)
                y_hats.append(batch_pred)  
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_label)
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc = round(acc.item() * 100 / config.world_size, 2)
        profiler.test_acc = acc  
        if rank == 0:
            print(f"test accuracy={acc}% in {t2.duration()} secs")
            
    if rank == 0:
        write_to_csv(config.log_path, [config], [profiler])
    ddp_exit()