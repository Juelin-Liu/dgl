//
// Created by juelin on 2/22/24.
//

#include "preprocess.h"
#include <dgl/packed_func_ext.h>
#include "../graph/unit_graph.h"
#include "cuda/array_ops.cuh"

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
      ATEN_ID_TYPE_SWITCH(count->dtype, CounterType, {
        ATEN_ID_TYPE_SWITCH(row->dtype, IndexType, {
          Increment<kDGLCUDA, CounterType, IndexType>(count, row);
        });
      });
    });
}