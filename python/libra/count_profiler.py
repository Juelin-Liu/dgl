import torch
from torch import Tensor

class CountProfiler:
    def __init__(self, bal_map: Tensor, xbal_map: Tensor, rand_map: Tensor):
        self.bal_map = bal_map
        self.xbal_map = xbal_map
        self.rand_map = rand_map
        self.world_size = 4
        
        self.crs_edge_bal = []
        self.crs_edge_xbal = []
        self.crs_edge_rand = []
        
        self.loc_edge_bal = []
        self.loc_edge_xbal = []
        self.loc_edge_rand = []
        
        self.src_bal = []
        self.src_xbal = []
        self.src_rand = []
        
        self.dst_bal = []
        self.dst_xbal = []
        self.dst_rand = []
        
    def add_edges(self, src, dst):
        def get_cnts(src, dst, mapping):
            src_pid = mapping[src]
            dst_pid = mapping[dst]
            loc_mask = src_pid == dst_pid
            crs_mask = src_pid != dst_pid
            loc_cnts = torch.bincount(mapping[src[loc_mask]]) # use the src's partition id to map the local edge (identical)
            crs_cnts = torch.bincount(mapping[src[crs_mask]]) # use the src's partition id to map the cross edge (subject to change)
            return loc_cnts, crs_cnts
        
        
        loc_cnts, crs_cnts = get_cnts(src, dst, self.bal_map)
        self.loc_edge_bal.append(loc_cnts.tolist())
        self.crs_edge_bal.append(crs_cnts.tolist())
        
        loc_cnts, crs_cnts = get_cnts(src, dst, self.xbal_map)
        self.loc_edge_xbal.append(loc_cnts.tolist())
        self.crs_edge_xbal.append(crs_cnts.tolist())
        
        loc_cnts, crs_cnts = get_cnts(src, dst, self.rand_map)
        self.loc_edge_rand.append(loc_cnts.tolist())
        self.crs_edge_rand.append(crs_cnts.tolist())
        
    def add_src(self, unique_src):
        _bal = torch.bincount(self.bal_map[unique_src])
        _xbal = torch.bincount(self.xbal_map[unique_src])
        _rand = torch.bincount(self.rand_map[unique_src])
        self.src_bal.append(_bal.tolist())
        self.src_xbal.append(_xbal.tolist())
        self.src_rand.append(_rand.tolist())
        
    def add_dst(self, unique_dst):
        _bal = torch.bincount(self.bal_map[unique_dst])
        _xbal = torch.bincount(self.xbal_map[unique_dst])
        _rand = torch.bincount(self.rand_map[unique_dst])
        
        self.dst_bal.append(_bal.tolist())
        self.dst_xbal.append(_xbal.tolist())
        self.dst_rand.append(_rand.tolist()) 
        
    def get_map(self, partition_type: str):
        assert(partition_type in ["bal", "xbal", "random"])
        if partition_type == "bal":
            return self.bal_map
        
        elif partition_type == "xbal":
            return self.xbal_map
        
        elif partition_type == "random":
            return self.rand_map
        
        else:
            return None
        
        
    def get_cnts(self, partition_type: str, layer: int, total_layer: int):
        assert(partition_type in ["bal", "xbal", "random"])
        
        start = layer
        step = total_layer

        def slice_row(row):
            return row[start:len(row):step]
        
        if partition_type == "bal":
            return (slice_row(self.src_bal),slice_row(self.dst_bal),slice_row(self.loc_edge_bal),slice_row(self.crs_edge_bal))
        elif partition_type == "xbal":
            return (slice_row(self.src_xbal),slice_row(self.dst_xbal),slice_row(self.loc_edge_xbal),slice_row(self.crs_edge_xbal))
        elif partition_type == "random":
            return (slice_row(self.src_rand),slice_row(self.dst_rand),slice_row(self.loc_edge_rand),slice_row(self.crs_edge_rand))
        else:
            return (None, None, None, None)
        
    def save(self, path):
        def toTensor(rows):
            for i in range(len(rows)):
                while len(rows[i]) != self.world_size:
                    rows[i].append(0)
                    
            return torch.Tensor(rows).type(torch.int)
        
        state = {"bal_map": self.bal_map, "xbal_map": self.xbal_map, "rand_map": self.rand_map,
                 "crs_edge_bal": toTensor(self.crs_edge_bal), "crs_edge_xbal": toTensor(self.crs_edge_xbal), "crs_edge_rand": toTensor(self.crs_edge_rand),
                 "loc_edge_bal": toTensor(self.loc_edge_bal), "loc_edge_xbal": toTensor(self.loc_edge_xbal), "loc_edge_rand": toTensor(self.loc_edge_rand),
                 "src_bal": toTensor(self.src_bal), "src_xbal": toTensor(self.src_xbal), "src_rand": toTensor(self.src_rand),
                 "dst_bal": toTensor(self.dst_bal), "dst_xbal": toTensor(self.dst_xbal), "dst_rand": toTensor(self.dst_rand)}
        torch.save(state, path)
    
    def load(self, path):
        state = torch.load(path)        
        self.bal_map = state["bal_map"]
        self.xbal_map = state["xbal_map"]
        self.rand_map = state["rand_map"]
        
        self.crs_edge_bal = state["crs_edge_bal"].tolist()
        self.crs_edge_xbal = state["crs_edge_xbal"].tolist()
        self.crs_edge_rand = state["crs_edge_rand"].tolist()
        
        self.loc_edge_bal = state["loc_edge_bal"].tolist()
        self.loc_edge_xbal = state["loc_edge_xbal"].tolist()
        self.loc_edge_rand = state["loc_edge_rand"].tolist()
        
        self.src_bal = state["src_bal"].tolist()
        self.src_xbal = state["src_xbal"].tolist()
        self.src_rand = state["src_rand"].tolist()
        
        self.dst_bal = state["dst_bal"].tolist()
        self.dst_xbal = state["dst_xbal"].tolist()
        self.dst_rand = state["dst_rand"].tolist()