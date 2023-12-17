//
// Created by juelinliu on 12/11/23.
//

#ifndef DGL_ARRAY_OPS_CUH
#define DGL_ARRAY_OPS_CUH
#include <dgl/array.h>

namespace dgl::dev
{
    int64_t NumItem(const NDArray& bitmap);

    void Reset(NDArray& bitmap);

    template<DGLDeviceType XPU, typename IdType>
    void Mask(NDArray& bitmap, NDArray row);

    template<DGLDeviceType XPU, typename IdType>
    NDArray Flagged(const NDArray& bitmap, DGLContext ctx);
}
#endif  // DGL_ARRAY_OPS_CUH
