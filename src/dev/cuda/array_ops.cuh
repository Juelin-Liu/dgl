//
// Created by juelinliu on 12/11/23.
//

#ifndef DGL_ARRAY_OPS_CUH
#define DGL_ARRAY_OPS_CUH
#include <dgl/array.h>

namespace dgl::dev
{
    int64_t NumItem(const NDArray& bitmap, int64_t capacity);

    void Reset(NDArray& bitmap, int64_t capacity);

    template<DGLDeviceType XPU, typename IdType>
    void Mask(NDArray& bitmap, int64_t capacity, NDArray row);

    template<DGLDeviceType XPU, typename IdType>
    NDArray Flagged(const NDArray& bitmap, int64_t capacity, DGLContext ctx);
}
#endif  // DGL_ARRAY_OPS_CUH
