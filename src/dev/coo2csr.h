//
// Created by juelin on 12/26/23.
//

#ifndef DGL_DEVCOO2CSR_H
#define DGL_DEVCOO2CSR_H
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/container.h>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <oneapi/tbb/parallel_sort.h>

namespace dgl::dev
{
  std::tuple<IdArray, IdArray, IdArray, IdArray> COO2CSR(int64_t v_num, NDArray src, NDArray dst, NDArray data) {
    auto dtype = src->dtype;
    auto ctx = src->ctx;
    CHECK_EQ(src.NumElements(), dst.NumElements());
    CHECK_EQ(src.NumElements(), data.NumElements());
    int64_t e_num = src.NumElements();
    LOG(INFO) << "COO2CSR edges: " << e_num << " | vertices: " << v_num;
    typedef std::tuple<int64_t, int64_t, int64_t > Edge; // src, dst, data
    typedef std::vector<Edge> EdgeVec;
    EdgeVec edge_vec;
    edge_vec.reserve(e_num * 2);
    ATEN_ID_TYPE_SWITCH(dtype, IdType, {
      auto src_ptr = src.Ptr<IdType>();
      auto dst_ptr = dst.Ptr<IdType>();
      auto data_ptr = data.Ptr<IdType>();
      for (int64_t i = 0; i < e_num; i++){
        edge_vec.emplace_back(src_ptr[i], dst_ptr[i], data_ptr[i]);
        edge_vec.emplace_back(dst_ptr[i], src_ptr[i], data_ptr[i]);
      }
    });
    CHECK_EQ(edge_vec.size(), e_num * 2);
    LOG(INFO) << "COO2CSR start sorting";

    tbb::parallel_sort(edge_vec.begin(), edge_vec.end(), [](const Edge& a, const Edge& b)->bool {
      if (std::get<0>(a) == std::get<0>(b)) {
        return std::get<1>(a)< std::get<1>(b);
      } else {
        return std::get<0>(a) < std::get<0>(b);
      }
    });

    edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
    edge_vec.shrink_to_fit();
    int64_t cur_e_num = edge_vec.size();
    IdArray degrees = aten::NewIdArray(v_num);
    IdArray indptr = aten::NewIdArray(v_num + 1);
    IdArray indices = aten::NewIdArray(cur_e_num);
    IdArray retdata = aten::NewIdArray(cur_e_num);

    int64_t * indices_ptr = indices.Ptr<int64_t>();
    int64_t * retdata_ptr = retdata.Ptr<int64_t>();
    int64_t * degrees_ptr = degrees.Ptr<int64_t>();
    memset(degrees.Ptr<void>(), 0, sizeof(int64_t) * v_num);
    for (int64_t i = 0; i < edge_vec.size(); i++) {
      const Edge& e = edge_vec.at(i);
      degrees_ptr[std::get<0>(e)]++;
      indices_ptr[i] = std::get<1>(e);
      retdata_ptr[i] = std::get<2>(e);
    }
    LOG(INFO) << "COO2CSR compute indptr";
    auto out_start = indptr.Ptr<int64_t>();
    auto out_end = std::exclusive_scan(degrees_ptr, degrees_ptr + v_num + 1, out_start, 0);

    CHECK_EQ(out_end - out_start, v_num + 1);
    CHECK_EQ(out_start[v_num], edge_vec.size());
    return {degrees, indptr, indices, retdata};
  } // COO2CSR
} // dgl::dev

#endif  // DGL_DEVCOO2CSR_H

