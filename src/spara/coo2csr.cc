////
//// Created by juelin on 12/26/23.
////
//
//#include "coo2csr.h"
//#include <dgl/packed_func_ext.h>
//#include "../graph/unit_graph.h"
//using namespace dgl::runtime;
//
//namespace dgl::dev {
////DGL_REGISTER_GLOBAL("dev._CAPI_COO2CSR")
////    .set_body([](DGLArgs args, DGLRetValue *rv) {
////      int64_t v_num = args[0];
////      NDArray src = args[1];
////      NDArray dst = args[2];
////      NDArray in_data = args[3];
////      bool to_undirected = args[4];
////      CHECK_EQ(dst.NumElements(), src.NumElements());
////      if (in_data.NumElements() != 0) {
////        CHECK_EQ(in_data.NumElements(), src.NumElements());
////        const auto [indptr, indices, data] =
////            COO2CSR(v_num, src, dst, in_data, to_undirected);
////        *rv = ConvertNDArrayVectorToPackedFunc({indptr, indices, data});
////      } else {
////        const auto [indptr, indices] = COO2CSR(v_num, src, dst, to_undirected);
////        *rv = ConvertNDArrayVectorToPackedFunc({indptr, indices, in_data});
////      }
////    });
//
//
//}  // namespace dgl::dev