import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from node.model import *
from node.utils import *
from dgl.dev import *

def bench_dgl_batch(configs: list[Config]):
    for config in configs:
        assert("dgl" in config.system)
        assert(config.graph_name == configs[0].graph_name)
        assert(config.model == configs[0].model)
        
    graph, feat, label, train_idx, valid_idx, test_idx, num_label = load_data(configs[0])
    for config in configs:
        config.num_classes = num_label
        try:
            spawn(train_dgl_ddp, args=(config, graph, feat, label, num_label, train_idx, test_idx), nprocs=config.world_size)
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
            
def train_dgl_ddp(rank: int, config: Config, graph: dgl.DGLGraph, feat: torch.Tensor, label: torch.Tensor, num_label: int, train_idx: torch.Tensor, test_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    if rank == 0:
        print(config)
    mode = "uva"
    feat_handle  = pin_memory_inplace(feat)
    label_handle = pin_memory_inplace(label)
    graph = graph.pin_memory_()    
    sample_config = SampleConfig(rank=rank, world_size=config.world_size, mode=mode, fanouts=config.fanouts)
    dataloader = GraphDataloader(graph, train_idx, sample_config)

    model = None
    if config.model == "gat":
        model = Gat(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label, num_heads=4)
    elif config.model == "sage":
        model = Sage(in_feats=feat.shape[1], hid_feats=config.hid_size, num_layers=len(config.fanouts), out_feats=num_label)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"preheating trainer on device: {device}")
    step = 0
    for input_nodes, output_nodes, blocks in dataloader:
        step += 1
        batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
        batch_label = gather_pinned_tensor_rows(label, output_nodes)
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
            batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
            batch_label = gather_pinned_tensor_rows(label, output_nodes)
            # dist.barrier()
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
        dataloader = GraphDataloader(graph, test_idx, sample_config)
        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():

                batch_feat = gather_pinned_tensor_rows(feat, input_nodes)
                batch_label = gather_pinned_tensor_rows(label, output_nodes)

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