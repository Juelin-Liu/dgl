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
      CHECK_EQ(in_data.NumElements(), src.NumElements());
      CHECK_EQ(dst.NumElements(), src.NumElements());
      const auto [degree, indptr, indices, data] = COO2CSR(v_num, src, dst, in_data);
      *rv=ConvertNDArrayVectorToPackedFunc({degree, indptr, indices, data});
    });

DGL_REGISTER_GLOBAL("dev._CAPI_COO2CSRNoData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t v_num = args[0];
      NDArray src = args[1];
      NDArray dst = args[2];
      CHECK_EQ(dst.NumElements(), src.NumElements());
      const auto [degree, indptr, indices] = COO2CSR(v_num, src, dst);
      *rv=ConvertNDArrayVectorToPackedFunc({degree, indptr, indices});
    });
}