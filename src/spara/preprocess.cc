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
DGL_REGISTER_GLOBAL("dev._CAPI_LoadSNAP")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string in_file = args[0];
      bool to_sym = args[1];
      const auto [out_indptr, out_indices] = LoadSNAP(in_file, to_sym);
      *rv = ConvertNDArrayVectorToPackedFunc({out_indptr, out_indices});
    });

DGL_REGISTER_GLOBAL("dev._CAPI_MakeSym")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray in_indptr = args[0];
      NDArray in_indices = args[1];
      NDArray in_data = args[2];
      CHECK(in_indices->dtype.bits == 32);
      CHECK(in_indptr->dtype.bits == 64);
      if (in_data.NumElements() != 0) {
        CHECK(in_data->dtype.bits == 32);
        CHECK_EQ(in_data.NumElements(), in_indices.NumElements());
        const auto [indptr, indices, data] =
            MakeSym(in_indptr, in_indices, in_data);
        *rv = ConvertNDArrayVectorToPackedFunc({indptr, indices, data});
      } else {
        const auto [indptr, indices] = MakeSym(in_indptr, in_indices);
        *rv = ConvertNDArrayVectorToPackedFunc({indptr, indices, in_data});
      }
    });

DGL_REGISTER_GLOBAL("dev._CAPI_MetisPartition")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t num_partition = args[0];
      int64_t num_iteration = args[1];
      int64_t num_initpart = args[2];
      double unbalance_val = args[3];
      bool obj_cut = args[4];

      NDArray indptr = args[5];
      NDArray indices = args[6];
      NDArray node_weight = args[7];
      NDArray edge_weight = args[8];

      CHECK(indptr->dtype.bits == 64);
      CHECK(indices->dtype.bits == 32);;
#if !defined(_WIN32)
      *rv = MetisPartition(num_partition, num_iteration, num_initpart, (float) unbalance_val, obj_cut, indptr, indices, node_weight, edge_weight);
#else
      LOG(FATAL) << "Metis partition does not support Windows.";
#endif  // !defined(_WIN32)
    });

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