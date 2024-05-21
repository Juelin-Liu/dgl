//
// Created by juelin on 2/22/24.
//

#include "preprocess.h"
#include <dgl/packed_func_ext.h>
#include "../graph/unit_graph.h"
#include "cuda/array_ops.cuh"

#define ATEN_COUNTER_BITS_SWITCH(bits, IdType, ...)                      \
  do {                                                              \
    CHECK((bits) == 32 || (bits) == 64 || (bits) == 16) << "bits must be 16, 32 or 64"; \
    if ((bits) == 16) {                                             \
      typedef int16_t IdType;                                       \
      { __VA_ARGS__ }                                               \
    } else if ((bits) == 32) {                                             \
      typedef int32_t IdType;                                       \
      { __VA_ARGS__ }                                               \
    } else if ((bits) == 64) {                                      \
      typedef int64_t IdType;                                       \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Counter can only be int16_t, int32 or int64";                \
    }                                                               \
  } while (0)


using namespace dgl::runtime;
namespace dgl::dev
{
DGL_REGISTER_GLOBAL("dev._CAPI_ReindexCSR")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray in_indptr = args[0];
      NDArray in_indices = args[1];
      const auto [out_indptr, out_indices] = ReindexCSR(in_indptr, in_indices);
      *rv = ConvertNDArrayVectorToPackedFunc({out_indptr, out_indices});
    });

DGL_REGISTER_GLOBAL("dev._CAPI_PartitionCSR")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray in_indptr = args[0];
      NDArray in_indices = args[1];
      NDArray flag = args[2];
      const auto [out_indptr, out_indices] = PartitionCSR(in_indptr, in_indices, flag);
      *rv = ConvertNDArrayVectorToPackedFunc({out_indptr, out_indices});
    });

DGL_REGISTER_GLOBAL("dev._CAPI_CompactCSR")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray in_indptr = args[0];
      NDArray in_indices_flag = args[1];
      const auto out_indptr = CompactCSR(in_indptr, in_indices_flag);
      *rv = out_indptr;
    });

DGL_REGISTER_GLOBAL("dev._CAPI_ExpandIndptr")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray in_indptr = args[0];
      *rv = ExpandIndptr(in_indptr);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Increment")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray count = args[0];
      NDArray row = args[1];
      ATEN_COUNTER_BITS_SWITCH(count->dtype.bits, CounterType, {
        ATEN_ID_TYPE_SWITCH(row->dtype, IndexType, {
          Increment<kDGLCUDA, CounterType, IndexType>(count, row);
        });
      });
    });
}