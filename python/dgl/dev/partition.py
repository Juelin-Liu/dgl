from dgl.backend import to_dgl_nd, from_dgl_nd
from .._ffi.function import _init_api

_init_api("dgl.dev", __name__)

def mtmetis_partition(num_partition:int, num_iteration:int, num_initpart:int, \
    unbalance_val:float, obj_cut:bool, \
    indptr, indices, node_weight, edge_weight):
    ret = _CAPI_MTMetisPartition(num_partition, num_iteration, num_initpart, \
        unbalance_val, obj_cut, \
        to_dgl_nd(indptr), to_dgl_nd(indices), to_dgl_nd(node_weight), to_dgl_nd(edge_weight))
    return from_dgl_nd(ret)
