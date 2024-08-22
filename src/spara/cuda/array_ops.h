//
// Created by juelinliu on 12/11/23.
//

#ifndef DGL_ARRAY_OPS_CUH
#define DGL_ARRAY_OPS_CUH
#include <dgl/array.h>

namespace dgl::dev {
int64_t NumItem(const NDArray& bitmap);

void Reset(NDArray& bitmap);

template <DGLDeviceType XPU, typename IdType>
void Mask(NDArray& bitmap, NDArray row);

template <DGLDeviceType XPU, typename IdType>
NDArray Flagged(const NDArray& bitmap, DGLContext ctx);

// template <DGLDeviceType XPU, typename IdType>
// void Increment(NDArray& count, const NDArray& indices);

template <DGLDeviceType XPU, typename CounterType, typename IndexType>
void Increment(NDArray& count, const NDArray& indices);
}  // namespace dgl::dev
#endif  // DGL_ARRAY_OPS_CUH