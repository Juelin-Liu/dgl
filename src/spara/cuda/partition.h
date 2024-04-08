//
// Created by juelin on 2/14/24.
//

#ifndef DGL_PARTITION_H
#define DGL_PARTITION_H
#include <dgl/array.h>

namespace dgl::dev
{

template <DGLDeviceType XPU, typename IdType, typename IndexType>
std::tuple<IdArray, IdArray, IdArray> compute_partition_continuous_indices(
    IdArray partition_idx, int num_partitions, cudaStream_t stream);

//template <DGLDeviceType XPU, typename PIdType>
//std::tuple<IdArray, IdArray, IdArray>
//compute_partition_continuous_indices_strawman(
//    IdArray partition_map, int num_partitions, cudaStream_t stream);

}
#endif  // DGL_PARTITION_H
