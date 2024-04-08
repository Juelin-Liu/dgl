//
// Created by juelinliu on 12/11/23.
//

#ifndef DGL_MAP_EDGES_CUH
#define DGL_MAP_EDGES_CUH

#include "../../graph/unit_graph.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"

using namespace dgl::runtime::cuda;
namespace dgl::dev {
// return the unique elements in the arr
NDArray getUnique(const std::vector<NDArray> &rows);

template <typename IdType>
void GPUMapEdges(
    aten::COOMatrix& mat, const runtime::cuda::OrderedHashTable<IdType>& hash,
    cudaStream_t stream);

template <typename IdType>
void GPUMapEdges(
    const NDArray &in_rows, NDArray &ret_row,
    const runtime::cuda::OrderedHashTable<IdType> &hash_table,
    cudaStream_t stream);
}  // namespace dgl::dev
// namespace dgl::dev
#endif  // DGL_MAP_EDGES_CUH
