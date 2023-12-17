//
// Created by juelinliu on 12/16/23.
//

#ifndef DGL_BATCH_ROWWISE_SAMPLING_CUH
#define DGL_BATCH_ROWWISE_SAMPLING_CUH

#include "../../graph/unit_graph.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"
#include "bitmap.h"

namespace  dgl::dev
{

template<DGLDeviceType XPU, typename IdType>
std::vector<aten::COOMatrix> CSRRowWiseSamplingUniformBatch(const aten::CSRMatrix& mat, const std::vector<NDArray>& rows,
                                                            const int64_t num_picks, const bool replace);

}
#endif  // DGL_BATCH_ROWWISE_SAMPLING_CUH
