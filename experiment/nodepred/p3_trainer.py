import torchmetrics.functional as MF
import torch.distributed as dist
import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from node.model import *
from node.utils import *
from dgl.dev import *

# This function split the feature data horizontally
# each node's data is partitioned into 'world_size' chunks
# return the partition corresponding to the 'rank'
# Input args:
# rank: [0, world_size - 1]
# Output: feat
def get_p3_local_feat(rank: int, world_size:int, feat: torch.Tensor, padding=True) -> torch.Tensor:
    org_feat_width = feat.shape[1]
    if padding and org_feat_width % world_size != 0:
        step = int(org_feat_width / world_size)
        pad = world_size - org_feat_width + step * world_size
        padded_width = org_feat_width + pad
        assert(padded_width % world_size == 0)
        step = int(padded_width / world_size)
        start_idx = rank * step
        end_idx = start_idx + step
        local_feat = None
        if rank == world_size - 1:
            # padding is required for P3 to work correctly
            local_feat = feat[:, start_idx : org_feat_width]
            zeros = torch.zeros((local_feat.shape[0], pad), dtype=local_feat.dtype)
            local_feat = torch.concatenate([local_feat, zeros], dim=1)
        else:
            local_feat = feat[:, start_idx : end_idx]
        return local_feat
    else:
        step = int(feat.shape[1] / world_size)
        start_idx = rank * step
        end_idx = min(start_idx + step, feat.shape[1])
        if rank == world_size - 1:
            end_idx = feat.shape[1]
        local_feat = feat[:, start_idx : end_idx]
        return local_feat

def get_p3_model(config: Config):
    # print(f"get_p3_model: {config=}")
    device = torch.cuda.current_device()
    if config.model == "sage":
        return create_sage_p3(device, config.in_feat, config.hid_size, config.num_classes, len(config.fanouts))
    elif config.model == "gat":
        return create_gat_p3(device, config.in_feat, config.hid_size, config.num_classes, len(config.fanouts))
    else:
        print(f"invalid model type {config.model}")
        exit(-1) 
    
def bench_p3_batch(configs: list[Config]):
    for config in configs:
        assert("p3" in config.system)
        assert(config.graph_name == configs[0].graph_name)
        assert(config.model == configs[0].model)
        
    graph, train_idx, valid_idx, test_idx = load_topo(configs[0], is_pinned=True)
    if config.graph_name in ["products"]:
        feat, label, num_label = load_feat_label(os.path.join(config.data_dir, config.graph_name))
    else:
        v_num = graph.num_nodes()
        feat = None
        num_label = 10
        label = gen_rand_label(v_num, 10)

    for config in configs:
        config.num_classes = num_label
        try:
            spawn(train_p3_ddp, args=(config, graph, feat, label, num_label, train_idx, test_idx), nprocs=config.world_size)
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
     
def train_p3_ddp(rank: int, config: Config, graph: dgl.DGLGraph, global_feat: torch.Tensor, label: torch.Tensor, num_label: int, train_idx: torch.Tensor, test_idx: torch.Tensor):
    ddp_setup(rank, config.world_size)
    device = torch.cuda.current_device()
    e2eTimer = Timer()
    if rank == 0:
        print(config)
        
    if global_feat is not None:
        feat = get_p3_local_feat(rank, config.world_size, global_feat).clone()
    else:
        feat_dim = get_feat_dim(config) // config.world_size
        feat = gen_rand_feat(v_num=graph.num_nodes(), feat_dim=feat_dim)

    config.in_feat = feat.shape[1]    
    assert(config.in_feat > 0)
    feat_handle = None
    label_handle = None
    if config.graph_name == "orkut":
        config.cache_size = get_tensor_size(feat)
        print(f"caching all feature data {config.cache_size} on {device=}")
        feat = feat.to(device)
        label = label.to(device)
    else:
        feat_handle  = pin_memory_inplace(feat)
        label_handle = pin_memory_inplace(label)
    graph = graph.pin_memory_()    
    mode = "uva"

        
    sample_config = SampleConfig(rank=rank, batch_size=config.batch_size, world_size=config.world_size, mode=mode, fanouts=config.fanouts)
    dataloader = GraphDataloader(graph, train_idx, sample_config)
    
    local_model, global_model = get_p3_model(config)
    # print(f"creating ddp on device: {device}")
    global_model = DDP(global_model, device_ids=[rank], output_device=rank)
    local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    # print(f"creating buffers on device: {device}")
    edge_size_lst: list = [torch.zeros((4,),dtype=torch.int64,device=rank) for _ in range(config.world_size)] #(rank, num_edges, num_dst_nodes, num_src_nodes)
    est_node_size = config.batch_size * 20
    local_feat_width = feat.shape[1]
    input_node_buffer_lst: list[torch.Tensor] = [] # input nodes 
    input_feat_buffer_lst: list[torch.Tensor] = [] # input feats 
    src_edge_buffer_lst: list[torch.Tensor] = [] # src nodes
    dst_edge_buffer_lst: list[torch.Tensor] = [] # dst nodes 
    global_grad_lst: list[torch.Tensor] = [] # feature data gathered for other gpus
    local_hid_buffer_lst: list[torch.Tensor] = [None] * config.world_size # storing feature data gathered from other gpus
    for idx in range(config.world_size):    
        nid_dtype = train_idx.dtype   
        input_node_buffer_lst.append(torch.zeros(est_node_size, dtype=nid_dtype, device=device))
        src_edge_buffer_lst.append(torch.zeros(est_node_size, dtype=nid_dtype, device=device))
        dst_edge_buffer_lst.append(torch.zeros(est_node_size, dtype=nid_dtype, device=device))
        global_grad_lst.append(torch.zeros([est_node_size, config.hid_size], dtype=torch.float32, device=device, requires_grad=False))
        input_feat_buffer_lst.append(torch.zeros([est_node_size, config.hid_size], dtype=torch.float32, device=device, requires_grad=False))

    shuffle = P3Shuffle.apply
    
    print(f"preheat model on device: {device}")
    step = 0
    for input_nodes, output_nodes, blocks in dataloader:
        step += 1       
        print(f"{step=}", flush=True)
        # 2. feature extraction and shuffling
        top_block: dgl.DGLGraph = blocks[0]
        src, dst = top_block.adj_tensors("coo")
        edge_size = torch.tensor((rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes()),dtype=torch.int64,device=rank)
        dist.all_gather(tensor_list=edge_size_lst, tensor=edge_size, async_op=False)            
        for r, edge_size, src_node_size, dst_node_size in edge_size_lst:
            src_edge_buffer_lst[r].resize_(edge_size)
            dst_edge_buffer_lst[r].resize_(edge_size)
            input_node_buffer_lst[r].resize_(src_node_size)
            
        batch_feat = None
        handle1 = dist.all_gather(tensor_list=input_node_buffer_lst, tensor=input_nodes, async_op=True)
        handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst, tensor=src, async_op=True)
        handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst, tensor=dst, async_op=True)
        batch_label = gather_tensor(label, output_nodes)

            
        handle1.wait()
        for r, _input_nodes in enumerate(input_node_buffer_lst):
            input_feat_buffer_lst[r] = gather_tensor(feat, _input_nodes)

        handle2.wait()
        handle3.wait()
        # print(f"{rank=} {epoch=} {iter_idx=} input_feat_shapes={[x.shape for x in input_feat_buffer_lst]} start computing first hidden layer")
        torch.cuda.synchronize()

        # 3. compute hid tensor for all ranks
        block = None
        for r in range(config.world_size):
            input_nodes = input_node_buffer_lst[r]
            input_feats = input_feat_buffer_lst[r]
            if r == rank:
                block = top_block
            else:
                src = src_edge_buffer_lst[r]
                dst = dst_edge_buffer_lst[r]
                src_node_size = edge_size_lst[r][2].item()
                dst_node_size = edge_size_lst[r][3].item()
                block = dgl.create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=device)
            
            # print(f"{rank=} {block.num_src_nodes()=} {input_feats.shape[0]=}", flush=True)
            local_hid_buffer_lst[r] = local_model(block, input_feats)
            global_grad_lst[r].resize_([block.num_dst_nodes(), config.hid_size])
    
    
        agg_hid: torch.Tensor = shuffle(rank, config.world_size, 
                                        local_hid_buffer_lst[rank], 
                                        local_hid_buffer_lst, 
                                        global_grad_lst)
        # 6. Compute forward pass locally
        batch_pred = global_model(blocks[1:], agg_hid)
        batch_loss = torch.nn.functional.cross_entropy(batch_pred, batch_label)
        torch.cuda.synchronize()

        # backward
        global_optimizer.zero_grad()
        local_optimizer.zero_grad()
        batch_loss.backward()
        

        #global_optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        for r, global_grad in enumerate(global_grad_lst):
            if r != rank:
                local_optimizer.zero_grad()
                local_hid_buffer_lst[r].backward(global_grad)
                local_optimizer.step()
        
        global_optimizer.step()
        torch.cuda.synchronize()
        if step > PREHEAT_STEP:
            dataloader.reset()
            break
            
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
        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"start epoch {epoch}")
        edges_computed_epoch = 0
        sampling_timer = CudaTimer()
        for input_nodes, output_nodes, blocks in dataloader:
            step += 1
            sampling_timer.end()
            
            # 2. feature extraction and shuffling
            feat_timer = CudaTimer()

            top_block: dgl.DGLGraph = blocks[0]
            src, dst = top_block.adj_tensors("coo")
            edge_size = torch.tensor((rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes()),dtype=torch.int64,device=rank)
            dist.all_gather(tensor_list=edge_size_lst, tensor=edge_size, async_op=False)            
            for r, edge_size, src_node_size, dst_node_size in edge_size_lst:
                src_edge_buffer_lst[r].resize_(edge_size)
                dst_edge_buffer_lst[r].resize_(edge_size)
                input_node_buffer_lst[r].resize_(src_node_size)
                
            batch_feat = None
            handle1 = dist.all_gather(tensor_list=input_node_buffer_lst, tensor=input_nodes, async_op=True)
            handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst, tensor=src, async_op=True)
            handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst, tensor=dst, async_op=True)

            batch_label = gather_tensor(label, output_nodes)

                
            handle1.wait()
            for r, _input_nodes in enumerate(input_node_buffer_lst):
                input_feat_buffer_lst[r] = gather_tensor(feat, _input_nodes)

                    
            handle2.wait()
            handle3.wait()
            # print(f"{rank=} {epoch=} {iter_idx=} input_feat_shapes={[x.shape for x in input_feat_buffer_lst]} start computing first hidden layer")
            torch.cuda.synchronize()
            feat_timer.end()            

            # 3. compute hid tensor for all ranks
            forward_timer = CudaTimer()
            block = None
            for r in range(config.world_size):
                input_nodes = input_node_buffer_lst[r]
                input_feats = input_feat_buffer_lst[r]
                if r == rank:
                    block = top_block
                else:
                    src = src_edge_buffer_lst[r]
                    dst = dst_edge_buffer_lst[r]
                    src_node_size = edge_size_lst[r][2].item()
                    dst_node_size = edge_size_lst[r][3].item()
                    block = dgl.create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=device)
                
                # print(f"{rank=} {block.num_src_nodes()=} {input_feats.shape[0]=}", flush=True)
                local_hid_buffer_lst[r] = local_model(block, input_feats)
                global_grad_lst[r].resize_([block.num_dst_nodes(), config.hid_size])
        
        
            agg_hid: torch.Tensor = shuffle(rank, config.world_size, 
                                            local_hid_buffer_lst[rank], 
                                            local_hid_buffer_lst, 
                                            global_grad_lst)
            # 6. Compute forward pass locally
            batch_pred = global_model(blocks[1:], agg_hid)
            batch_loss = torch.nn.functional.cross_entropy(batch_pred, batch_label)
            torch.cuda.synchronize()
            forward_timer.end()

            # backward
            backward_timer = CudaTimer()
            global_optimizer.zero_grad()
            local_optimizer.zero_grad()
            batch_loss.backward()
            
            global_optimizer.step()
            for r, global_grad in enumerate(global_grad_lst):
                if r != rank:
                    local_optimizer.zero_grad()
                    local_hid_buffer_lst[r].backward(global_grad)
                    local_optimizer.step()
                    
            torch.cuda.synchronize()
            backward_timer.end()
            for block in blocks:
                edges_computed_epoch += block.num_edges()
            sampling_timers.append(sampling_timer)
            feature_timers.append(feat_timer)
            forward_timers.append(forward_timer)
            backward_timers.append(backward_timer)
            sampling_timer = CudaTimer()
            # torch.cuda.synchronize()
            # dist.barrier()
            
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
    if rank == 0:
        print(f"train for {config.num_epoch} epochs in {duration}s")
        if not config.test_model_acc:
            write_to_csv(config.log_path, [config], [profiler])
    
    dist.barrier()
    
    if config.test_model_acc:
        print(f"testing model accuracy on {device}")
        dataloader = GraphDataloader(graph, test_idx, sample_config)
        local_model.eval()
        global_model.eval()
        ys = []
        y_hats = []
        for input_nodes, output_nodes, blocks in dataloader:
            with torch.no_grad():
                top_block: dgl.DGLGraph = blocks[0]
                src, dst = top_block.adj_tensors("coo")
                edge_size = torch.tensor((rank, src.shape[0], top_block.num_src_nodes(), top_block.num_dst_nodes()), dtype=torch.int64, device=rank)
                dist.all_gather(edge_size_lst, edge_size)
                
                for r, edge_size, src_node_size, dst_node_size in edge_size_lst:
                    src_edge_buffer_lst[r].resize_(edge_size)
                    dst_edge_buffer_lst[r].resize_(edge_size)
                    input_node_buffer_lst[r].resize_(src_node_size)
                    
                batch_feat = None
                handle1 = dist.all_gather(tensor_list=input_node_buffer_lst, tensor=input_nodes, async_op=True)
                handle2 = dist.all_gather(tensor_list=src_edge_buffer_lst, tensor=src, async_op=True)
                handle3 = dist.all_gather(tensor_list=dst_edge_buffer_lst, tensor=dst, async_op=True)

                batch_label = gather_tensor(label, output_nodes)
                    
                handle1.wait()
                for r, _input_nodes in enumerate(input_node_buffer_lst):
                    input_feat_buffer_lst[r] = gather_tensor(feat, _input_nodes)

                        
                handle2.wait()
                handle3.wait()
                torch.cuda.synchronize()
                block = None
                for r in range(config.world_size):
                    input_nodes = input_node_buffer_lst[r]
                    input_feats = input_feat_buffer_lst[r]
                    if r == rank:
                        block = top_block
                    else:
                        src = src_edge_buffer_lst[r]
                        dst = dst_edge_buffer_lst[r]
                        src_node_size = edge_size_lst[r][2].item()
                        dst_node_size = edge_size_lst[r][3].item()
                        block = dgl.create_block(('coo', (src, dst)), num_dst_nodes=dst_node_size, num_src_nodes=src_node_size, device=device)
                                            
                    local_hid_buffer_lst[r] = local_model(block, input_feats)
                    global_grad_lst[r].resize_([block.num_dst_nodes(), config.hid_size])
            
            
                agg_hid: torch.Tensor = shuffle(rank, config.world_size, 
                                                local_hid_buffer_lst[rank], 
                                                local_hid_buffer_lst, 
                                                global_grad_lst)
                # 6. Compute forward pass locally
                ys.append(batch_label)
                batch_pred = global_model(blocks[1:], agg_hid)
                y_hats.append(batch_pred)
                # torch.cuda.synchronize()
                
        acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys), task="multiclass", num_classes=num_label)
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc = round(acc.item() * 100 / config.world_size, 2)
        profiler.test_acc = acc  
        if rank == 0:
            print(f"test accuracy={acc}%")
            write_to_csv(config.log_path, [config], [profiler])
    ddp_exit()
