//
// Created by juelin on 12/26/23.
//

#include "coo2csr.h"
#include "../graph/unit_graph.h"
#include <dgl/packed_func_ext.h>
using namespace dgl::runtime;

namespace dgl::dev
{
DGL_REGISTER_GLOBAL("dev._CAPI_COO2CSR")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t v_num = args[0];
      NDArray src = args[1];
      NDArray dst = args[2];
      NDArray in_data = args[3];
      CHECK_EQ(dst.NumElements(), src.NumElements());
      if (in_data.NumElements() != 0) {
        CHECK_EQ(in_data.NumElements(), src.NumElements());
        const auto [indptr, indices, data] = COO2CSR(v_num, src, dst, in_data);
        *rv=ConvertNDArrayVectorToPackedFunc({indptr, indices, data});
      } else {
        const auto [indptr, indices] = COO2CSR(v_num, src, dst);
        *rv=ConvertNDArrayVectorToPackedFunc({indptr, indices, in_data});
      }
    });

    DGL_REGISTER_GLOBAL("dev._CAPI_CompactCSR")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray in_indptr = args[0];
      NDArray in_indices = args[1];
      const auto [out_indptr, out_indices] = CompactCSR(in_indptr, in_indices);
      *rv=ConvertNDArrayVectorToPackedFunc({out_indptr, out_indices});
    });

    DGL_REGISTER_GLOBAL("dev._CAPI_LoadSNAP")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      std::string in_file = args[0];
      const auto [out_indptr, out_indices] = LoadSNAP(in_file);
      *rv=ConvertNDArrayVectorToPackedFunc({out_indptr, out_indices});
    });
}