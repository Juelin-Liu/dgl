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
  struct EdgeWithData
  {
    int64_t _src{0}; 
    int64_t _dst{0};
    int64_t _data{0};
    EdgeWithData() = default;
    EdgeWithData(int64_t src, int64_t dst, int64_t data): _src{src}, _dst{dst}, _data{data} {};
    bool operator == (const EdgeWithData &other) const {
      return other._src == _src && other._dst == _dst;
    }
    bool operator < (const EdgeWithData& other) const {
      if (_src == other._src) {
        return _dst < other._dst;
      } else {
        return _src < other._src;
      }
    }
  };

  struct Edge
  {
    int64_t _src{0};
    int64_t _dst{0};
    Edge(int64_t src, int64_t dst): _src{src}, _dst{dst} {};
    Edge() = default;
    bool operator == (const Edge &other) const {
      return other._src == _src && other._dst == _dst;
    }
    bool operator < (const Edge& other) const {
      if (_src == other._src) {
        return _dst < other._dst;
      } else {
        return _src < other._src;
      }
    }
  };

  std::tuple<IdArray, IdArray, IdArray, IdArray> COO2CSR(int64_t v_num, NDArray src, NDArray dst, NDArray data) {
    auto dtype = src->dtype;
    auto ctx = src->ctx;
    int64_t e_num = src.NumElements();
    LOG(INFO) << "COO2CSR edges: " << e_num << " | vertices: " << v_num;
    typedef std::vector<EdgeWithData> EdgeVec;
    EdgeVec edge_vec;
    edge_vec.resize(2 * e_num);
    ATEN_ID_TYPE_SWITCH(dtype, IdType, {
      auto src_ptr = src.Ptr<IdType>();
      auto dst_ptr = dst.Ptr<IdType>();
      auto data_ptr = data.Ptr<IdType>();

      tbb::parallel_for( tbb::blocked_range<int64_t >(0, e_num),
                       [&](tbb::blocked_range<int64_t > r)
      {
          for (int64_t i=r.begin(); i<r.end(); i++)
          {
            edge_vec.at(i * 2) = {src_ptr[i], dst_ptr[i], data_ptr[i]};
            edge_vec.at(i * 2 + 1) = {dst_ptr[i], src_ptr[i], data_ptr[i]};
          }
      });
    });
    LOG(INFO) << "COO2CSR start sorting";

    // tbb::parallel_sort(edge_vec.begin(), edge_vec.end(), [](const EdgeWithData a, const EdgeWithData b)->bool {
    //   if (a._src == b._src) {
    //     return a._dst < b._dst;
    //   } else {
    //     return a._src < b._dst;
    //   }
    // });
    tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
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
    memset(indptr.Ptr<void>(), 0, sizeof(int64_t) * (v_num + 1));
    for (int64_t i = 0; i < edge_vec.size(); i++) {
      const EdgeWithData& e = edge_vec.at(i);
      degrees_ptr[e._src]++;
      indices_ptr[i] = e._dst;
      retdata_ptr[i] = e._data;
    }
    LOG(INFO) << "COO2CSR compute indptr";
    auto out_start = indptr.Ptr<int64_t>();
    auto out_end = std::exclusive_scan(degrees_ptr, degrees_ptr + v_num + 1, out_start, 0);

    CHECK_EQ(out_end - out_start, v_num + 1);
    CHECK_EQ(out_start[v_num], edge_vec.size());
    return {degrees, indptr, indices, retdata};
  } // COO2CSR

  std::tuple<IdArray, IdArray, IdArray> COO2CSR(int64_t v_num, NDArray src, NDArray dst) {
    auto dtype = src->dtype;
    auto ctx = src->ctx;
    int64_t e_num = src.NumElements();
    LOG(INFO) << "COO2CSR edges: " << e_num << " | vertices: " << v_num;
    typedef std::vector<Edge> EdgeVec;
    EdgeVec edge_vec;
    edge_vec.resize(2 * e_num);
    ATEN_ID_TYPE_SWITCH(dtype, IdType, {
      auto src_ptr = src.Ptr<IdType>();
      auto dst_ptr = dst.Ptr<IdType>();
      tbb::parallel_for( tbb::blocked_range<int64_t >(0, e_num),
                       [&](tbb::blocked_range<int64_t > r)
      {
          for (int64_t i=r.begin(); i<r.end(); i++)
          {
            edge_vec.at(i * 2) = {src_ptr[i], dst_ptr[i]};
            edge_vec.at(i * 2 + 1) = {dst_ptr[i], src_ptr[i]};
          }
      });
    });
    LOG(INFO) << "COO2CSR start sorting";

    // tbb::parallel_sort(edge_vec.begin(), edge_vec.end(), [](const Edge& a, const Edge& b)->bool {
    //   if (a._src == b._src) {
    //     return a._dst < b._dst;
    //   } else {
    //     return a._src < b._dst;
    //   }
    // });
    tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
    edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
    edge_vec.shrink_to_fit();
    int64_t cur_e_num = edge_vec.size();
    IdArray degrees = aten::NewIdArray(v_num);
    IdArray indptr = aten::NewIdArray(v_num + 1);
    IdArray indices = aten::NewIdArray(cur_e_num);

    int64_t * indices_ptr = indices.Ptr<int64_t>();
    int64_t * degrees_ptr = degrees.Ptr<int64_t>();
    int64_t * out_start = indptr.Ptr<int64_t>();
    memset(degrees.Ptr<void>(), 0, sizeof(int64_t) * v_num);
    memset(indptr.Ptr<void>(), 0, sizeof(int64_t) * (v_num + 1));
    for (int64_t i = 0; i < edge_vec.size(); i++) {
      const Edge& e = edge_vec.at(i);
      degrees_ptr[e._src]++;
      indices_ptr[i] = e._dst;
    }
    LOG(INFO) << "COO2CSR compute indptr";
    auto out_end = std::exclusive_scan(degrees_ptr, degrees_ptr + v_num + 1, out_start, 0);

    CHECK_EQ(out_end - out_start, v_num + 1);
    CHECK_EQ(out_start[v_num], edge_vec.size());
    return {degrees, indptr, indices};
  } // COO2CSR

} // dgl::dev

#endif  // DGL_DEVCOO2CSR_H

