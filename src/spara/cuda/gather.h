//
// Created by juelin on 2/14/24.
//

#ifndef DGL_GATHER_H
#define DGL_GATHER_H
#include <dgl/array.h>

namespace dgl::dev
{
template <DGLDeviceType XPU, typename IdType, typename IndexType>
IdArray gather_atomic_accumulate(
    IdArray accumulated_grads, IdArray idx_unique_to_shuffled,
    IdArray grad_shuffled_reshape, cudaStream_t stream);
}
#endif  // DGL_GATHER_H