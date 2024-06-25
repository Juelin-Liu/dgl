////
//// Created by juelin on 12/26/23.
////
//
//#ifndef DGL_DEVCOO2CSR_H
//#define DGL_DEVCOO2CSR_H
//#include <dgl/array.h>
//#include <dgl/aten/array_ops.h>
//#include <dgl/runtime/c_runtime_api.h>
//#include <dgl/runtime/container.h>
//#include <oneapi/tbb/parallel_sort.h>
//
//#include <algorithm>
//#include <atomic>
//#include <cstdint>
//#include <cstring>
//#include <filesystem>
//#include <fstream>
//#include <numeric>
//#include <tuple>
//
//#include "dgl/aten/macro.h"
//#include "dgl/runtime/ndarray.h"
//#include "metis.h"
//
//namespace dgl::dev {
//
//static std::tuple<IdArray, IdArray, IdArray> COO2CSR(
//    int64_t v_num, NDArray src, NDArray dst, NDArray data, bool to_undirected) {
//  int64_t e_num = src.NumElements();
//
//  LOG(INFO) << "COO2CSR v_num: " << v_num << " | e_num: " << e_num;
//  typedef std::vector<EdgeWithData> EdgeVec;
//  EdgeVec edge_vec;
//  edge_vec.resize(e_num + to_undirected * e_num);
//  auto src_ptr = src.Ptr<int32_t >();
//  auto dst_ptr = dst.Ptr<int32_t >();
//  auto data_ptr = data.Ptr<int32_t >();
//  tbb::parallel_for(
//      tbb::blocked_range<int64_t>(0, e_num),
//      [&](tbb::blocked_range<int64_t> r) {
//        for (int64_t i = r.begin(); i < r.end(); i++) {
//          if (to_undirected) {
//            edge_vec.at(i * 2) = {src_ptr[i], dst_ptr[i], data_ptr[i]};
//            edge_vec.at(i * 2 + 1) = {dst_ptr[i], src_ptr[i], data_ptr[i]};
//          } else {
//            edge_vec.at(i) = {src_ptr[i], dst_ptr[i], data_ptr[i]};
//          }
//        }
//      });
//  LOG(INFO) << "COO2CSR start sorting";
//  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
//  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
//  edge_vec.shrink_to_fit();
//  int64_t cur_e_num = edge_vec.size();
//  IdArray indptr = aten::NewIdArray(v_num + 1);
//  IdArray indices = aten::NewIdArray(cur_e_num);
//  IdArray retdata = aten::NewIdArray(cur_e_num);
//
//  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
//  memset(retdata.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
//  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
//  std::vector<std::atomic<int64_t>> degree(v_num + 1);
//  int64_t *indices_ptr = indices.Ptr<int64_t>();
//  int64_t *retdata_ptr = retdata.Ptr<int64_t>();
//  LOG(INFO) << "COO2CSR compute degree";
//
//  tbb::parallel_for(
//      tbb::blocked_range<int64_t>(0, edge_vec.size()),
//      [&](tbb::blocked_range<int64_t> r) {
//        for (int64_t i = r.begin(); i < r.end(); i++) {
//          const auto &e = edge_vec.at(i);
//          degree.at(e._src)++;
//          indices_ptr[i] = e._dst;
//          retdata_ptr[i] = e._data;
//        }
//      });
//
//  LOG(INFO) << "COO2CSR compute indptr";
//  auto out_start = indptr.Ptr<int64_t>();
//  auto out_end =
//      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
//  LOG(INFO) << "COO2CSR e_num after convert " << cur_e_num;
//
//  CHECK_EQ(out_end - out_start, v_num + 1);
//  CHECK_EQ(out_start[v_num], edge_vec.size());
//  return {indptr, indices, retdata};
//}  // COO2CSR
//
//
////static std::tuple<IdArray, IdArray> MakeSym(
////    NDArray in_indptr, NDArray in_indices) {
////  auto dtype = in_indices->dtype;
////  int64_t e_num = in_indices.NumElements();
////  int64_t v_num = in_indptr.NumElements() - 1;
////  LOG(INFO) << "MakeSym v_num: " << v_num << " | e_num: " << e_num;
////  typedef std::vector<Edge> EdgeVec;
////  EdgeVec edge_vec;
////  edge_vec.resize(e_num * 2);
////  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
////    auto _in_indptr = in_indptr.Ptr<IdType>();
////    auto _in_indices = in_indices.Ptr<IdType>();
////    tbb::parallel_for(
////        tbb::blocked_range<int64_t>(0, v_num),
////        [&](tbb::blocked_range<int64_t> r) {
////          for (int64_t v = r.begin(); v < r.end(); v++) {
////            int64_t start = _in_indptr[v];
////            int64_t end = _in_indptr[v + 1];
////            for (int64_t i = start; i < end; i++) {
////              int64_t u = _in_indices[i];
////              edge_vec.at(i * 2) = {v, u};
////              edge_vec.at(i * 2 + 1) = {u, v};
////            }
////          }
////        });
////  });
////  LOG(INFO) << "MakeSym start sorting";
////  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
////  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
////  edge_vec.shrink_to_fit();
////  int64_t cur_e_num = edge_vec.size();
////  IdArray indptr = aten::NewIdArray(v_num + 1);
////  IdArray indices = aten::NewIdArray(cur_e_num);
////  // IdArray retdata = aten::NewIdArray(cur_e_num);
////
////  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
////  // memset(retdata.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
////  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
////  std::vector<std::atomic<int64_t>> degree(v_num + 1);
////  int64_t *indices_ptr = indices.Ptr<int64_t>();
////  // int64_t * retdata_ptr = retdata.Ptr<int64_t>();
////  LOG(INFO) << "MakeSym compute degree";
////
////  tbb::parallel_for(
////      tbb::blocked_range<int64_t>(0, edge_vec.size()),
////      [&](tbb::blocked_range<int64_t> r) {
////        for (int64_t i = r.begin(); i < r.end(); i++) {
////          const auto &e = edge_vec.at(i);
////          degree.at(e._src)++;
////          indices_ptr[i] = e._dst;
////          // retdata_ptr[i] = e._data;
////        }
////      });
////
////  LOG(INFO) << "COO2CSR compute indptr";
////  auto out_start = indptr.Ptr<int64_t>();
////  auto out_end =
////      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
////  LOG(INFO) << "COO2CSR e_num after convert " << cur_e_num;
////
////  CHECK_EQ(out_end - out_start, v_num + 1);
////  CHECK_EQ(out_start[v_num], edge_vec.size());
////  return {indptr, indices};
////}  // MakeSym
//
//static std::tuple<IdArray, IdArray> COO2CSR(
//    int64_t v_num, NDArray src, NDArray dst, bool to_undirected) {
//  auto dtype = src->dtype;
//  int64_t e_num = src.NumElements();
//  LOG(INFO) << "COO2CSR v_num: " << v_num << " | e_num: " << e_num;
//  typedef std::vector<Edge> EdgeVec;
//  EdgeVec edge_vec;
//  edge_vec.resize(e_num + to_undirected * e_num);
//  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
//    auto src_ptr = src.Ptr<IdType>();
//    auto dst_ptr = dst.Ptr<IdType>();
//    tbb::parallel_for(
//        tbb::blocked_range<int64_t>(0, e_num),
//        [&](tbb::blocked_range<int64_t> r) {
//          for (int64_t i = r.begin(); i < r.end(); i++) {
//            if (to_undirected) {
//              edge_vec.at(i * 2) = {src_ptr[i], dst_ptr[i]};
//              edge_vec.at(i * 2 + 1) = {dst_ptr[i], src_ptr[i]};
//            } else {
//              edge_vec.at(i) = {src_ptr[i], dst_ptr[i]};
//            }
//          }
//        });
//  });
//  LOG(INFO) << "COO2CSR start sorting";
//  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
//  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
//  edge_vec.shrink_to_fit();
//  int64_t cur_e_num = edge_vec.size();
//  std::vector<std::atomic<int64_t>> degree(v_num + 1);
//  IdArray indptr = aten::NewIdArray(v_num + 1);
//  IdArray indices = aten::NewIdArray(cur_e_num);
//  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
//  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
//  int64_t *indices_ptr = indices.Ptr<int64_t>();
//  int64_t *out_start = indptr.Ptr<int64_t>();
//  LOG(INFO) << "COO2CSR compute degree";
//  tbb::parallel_for(
//      tbb::blocked_range<int64_t>(0, edge_vec.size()),
//      [&](tbb::blocked_range<int64_t> r) {
//        for (int64_t i = r.begin(); i < r.end(); i++) {
//          const Edge &e = edge_vec.at(i);
//          degree.at(e._src)++;
//          indices_ptr[i] = e._dst;
//        }
//      });
//
//  LOG(INFO) << "COO2CSR compute indptr";
//  auto out_end =
//      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
//  LOG(INFO) << "COO2CSR e_num after convert " << cur_e_num;
//
//  CHECK_EQ(out_end - out_start, v_num + 1);
//  CHECK_EQ(out_start[v_num], edge_vec.size());
//  return {indptr, indices};
//
//}  // COO2CSR
//
//
//}  // namespace dgl::dev
//
//#endif  // DGL_DEVCOO2CSR_H
