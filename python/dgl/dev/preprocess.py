from .util import *

# this is an inplace function the content in indices will be overwriten
def ReindexCSR(indptr: Tensor, indices: Tensor):
    indptr = F.to_dgl_nd(indptr)
    indices = F.to_dgl_nd(indices)
    ret = _CAPI_ReindexCSR(indptr, indices)
    indptr = F.from_dgl_nd(ret(0))
    indices = F.from_dgl_nd(ret(1))
    return (indptr, indices)

# compact indptr to remove the edges not in the flag
def CompactCSR(indptr: Tensor, flag: Tensor):
    indptr = F.to_dgl_nd(indptr)
    flag = F.to_dgl_nd(flag)
    _indptr = _CAPI_CompactCSR(indptr, flag)
    return F.from_dgl_nd(_indptr)

def LoadSNAP(in_file: str, to_sym=True):
    ret = _CAPI_LoadSNAP(in_file, to_sym)
    indptr = F.from_dgl_nd(ret(0))
    indices = F.from_dgl_nd(ret(1))
    return indptr, indices