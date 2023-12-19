
    
# class Config:
#     def __init__(self, graph_name, world_size, num_epoch, fanouts,
#                  batch_size, system, model, hid_size, cache_size, log_path, data_dir, pool_size, batch_layer, replace):
#         try:
#             self.machine_name = os.environ['MACHINE_NAME']
#         except Exception as e:
#             self.machine_name = "Elba"
#         self.graph_name = graph_name
#         self.world_size = world_size
#         self.num_epoch = num_epoch
#         self.fanouts = fanouts
#         self.batch_size = batch_size
#         self.system = system
#         self.model = model
#         self.in_feat = -1
#         self.num_classes = -1
#         self.cache_size = cache_size
#         self.hid_size = hid_size
#         self.log_path = log_path
#         self.data_dir = data_dir
#         self.num_redundant_layer = len(self.fanouts)
#         self.partition_type = "edge_balanced"
#         self.pool_size = pool_size
#         self.batch_layer = batch_layer
#         self.replace = replace
        
#     def get_file_name(self):
#         if "groot" not in self.system:
#             return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
#                      f"{len(self.fanouts)}x{self.fanouts[0]}_{self.cache_size}")
#         else:
#             return (f"{self.system}_{self.graph_name}_{self.model}_{self.batch_size}_{self.hid_size}_" + \
#                     f"{len(self.fanouts)}x{self.fanouts[0]}_{self.num_redundant_layer}_{self.cache_size}")

#     def header(self):
#         return ["machine_name", "system", "graph_name", "world_size", "num_epoch", "fanouts", "batch_size", "pool_size", "batch_layer", "replace"]
    
#     def content(self):
#         return [ self.machine_name, self.system, self.graph_name, self.world_size, self.num_epoch, self.fanouts, self.batch_size, self.pool_size, self.batch_layer, self.replace]

#     def __repr__(self):
#         res = ""
#         header = self.header()
#         content = self.content()
        
#         for header, ctn in zip(header, content):
#             res += f"{header}={ctn} | "
#         res += f"num_classes={self.num_classes}"
#         res += "\n"
#         return res    

# class Profiler:
#     def __init__(self, num_epoch: int, duration: float, sampling_time : float, feature_time: float,\
#                  forward_time: float, backward_time: float, test_acc: float):
#         self.num_epoch = num_epoch
#         self.duration = duration
#         self.sampling_time = sampling_time
#         self.feature_time = feature_time
#         self.forward_time = forward_time
#         self.backward_time = backward_time
#         self.test_acc = test_acc
#         self.allocated_mb, self.reserved_mb = get_memory_info()
#     def header(self):
#         header = ["epoch (s)", "sampling (s)", "feature (s)", "forward (s)", "backward (s)", \
#                   "allocated (MB)", "reserved (MB)", "test accuracy %", ]
#         return header
    
#     def content(self):
#         content = [self.duration / self.num_epoch, \
#                    self.sampling_time / self.num_epoch, \
#                    self.feature_time / self.num_epoch, \
#                    self.forward_time / self.num_epoch, \
#                    self.backward_time / self.num_epoch, \
#                    self.allocated_mb, \
#                    self.reserved_mb, \
#                    self.test_acc]
#         return content
    
#     def __repr__(self):
#         res = ""
#         header = self.header()
#         content = self.content()
#         for header, ctn in zip(header, content):
#             res += f"{header}={ctn} | "
#         res += "\n"
#         return res

# def empty_profiler():
#     empty = -1
#     profiler = Profiler(num_epoch=1, duration=empty, sampling_time=empty, feature_time=empty, forward_time=empty, backward_time=empty, test_acc=empty)
#     return profiler

# def oom_profiler():
#     oom = "oom"
#     profiler = Profiler(num_epoch=1, duration=oom, sampling_time=oom, feature_time=oom, forward_time=oom, backward_time=oom, test_acc=oom)
#     return profiler

# def get_duration(timers: list[CudaTimer], rb=3)->float:
#     res = 0.0
#     for timer in timers:
#         res += timer.duration()
#     return round(res, rb)

# def write_to_csv(out_path, configs: list[Config], profilers: list[Profiler]):
#     assert(len(configs) == len(profilers))
#     def get_row(header, content):
#         res = {}
#         for k, v in zip(header, content):
#             res[k] = v
#         return res
    
#     has_header = os.path.isfile(out_path)
#     with open(out_path, 'a') as f:
#         header = configs[0].header() + profilers[0].header()
#         writer = csv.DictWriter(f, fieldnames=header)        
#         if not has_header:
#             writer.writeheader()
#         for config, profiler in zip(configs, profilers):
#             row = get_row(config.header() + profiler.header(), config.content() + profiler.content())
#             writer.writerow(row)
#     print("Experiment result has been written to: ", out_path)

# def get_configs(graph_name, system, log_path, data_dir, pool_size, batch_layer, replace):
#     fanouts = [[10, 10, 10]]
#     pool_sizes = [pool_size]
#     configs = []
#     for fanout in fanouts:
#         for pool_size in pool_sizes:
#             config = Config(graph_name=graph_name, 
#                             world_size=1, 
#                             num_epoch=1, 
#                             fanouts=fanout, 
#                             batch_size=1024, 
#                             system=system, 
#                             model="sage",
#                             hid_size=128, 
#                             cache_size=0, 
#                             log_path=log_path,
#                             data_dir=data_dir,
#                             pool_size=pool_size,
#                             batch_layer=batch_layer,
#                             replace=replace)
#             configs.append(config)
#     return configs