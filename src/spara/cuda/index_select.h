//
// Created by juelin on 8/15/23.
//

#ifndef DGL_CUDA_INDEX_SELECT_CUH
#define DGL_CUDA_INDEX_SELECT_CUH

#include <dgl/array.h>
#include <cuda_runtime.h>
namespace dgl::dev {
NDArray IndexSelect(
    const NDArray &array, const IdArray &index, cudaStream_t stream);

void IndexSelect(
    const NDArray &array, const IdArray &index, void *out_buff,
                 cudaStream_t stream);
void IndexSelect(
    const NDArray &array, const IdArray &index, const IdArray &out_idx,
                 NDArray &out_buff, cudaStream_t stream);
} // namespace dgl::dev

#endif // DGL_CUDA_INDEX_SELECT_CUH