//
// Created by juelinliu on 12/11/23.
//

#ifndef DGL_MAP_EDGES_CUH
#define DGL_MAP_EDGES_CUH

#include <assert.h>

#include "../../graph/unit_graph.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"
using namespace dgl::runtime::cuda;
namespace dgl::dev {
template <typename IdType>
void GPUMapEdges(
    aten::COOMatrix& mat, const runtime::cuda::OrderedHashTable<IdType>& hash);
}  // namespace dgl::dev
#endif  // DGL_MAP_EDGES_CUH
