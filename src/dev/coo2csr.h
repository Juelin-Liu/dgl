//
// Created by juelin on 12/26/23.
//

#ifndef DGL_DEVCOO2CSR_H
#define DGL_DEVCOO2CSR_H
#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/container.h>
#include <oneapi/tbb/parallel_sort.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <tuple>

#include "dgl/aten/macro.h"
#include "dgl/runtime/ndarray.h"
#include "metis.h"

namespace dgl::dev {
struct EdgeWithData {
  int64_t _src{0};
  int64_t _dst{0};
  int64_t _data{0};
  EdgeWithData() = default;
  EdgeWithData(int64_t src, int64_t dst, int64_t data)
      : _src{src}, _dst{dst}, _data{data} {};
  bool operator==(const EdgeWithData &other) const {
    return other._src == _src && other._dst == _dst;
  }
  bool operator<(const EdgeWithData &other) const {
    if (_src == other._src) {
      return _dst < other._dst;
    } else {
      return _src < other._src;
    }
  }
};

struct Edge {
  int64_t _src{0};
  int64_t _dst{0};
  Edge(int64_t src, int64_t dst) : _src{src}, _dst{dst} {};
  Edge() = default;
  bool operator==(const Edge &other) const {
    return other._src == _src && other._dst == _dst;
  }
  bool operator<(const Edge &other) const {
    if (_src == other._src) {
      return _dst < other._dst;
    } else {
      return _src < other._src;
    }
  }
};

static std::tuple<IdArray, IdArray, IdArray> COO2CSR(
    int64_t v_num, NDArray src, NDArray dst, NDArray data, bool to_undirected) {
  auto dtype = src->dtype;
  int64_t e_num = src.NumElements();
  LOG(INFO) << "COO2CSR v_num: " << v_num << " | e_num: " << e_num;
  typedef std::vector<EdgeWithData> EdgeVec;
  EdgeVec edge_vec;
  edge_vec.resize(e_num + to_undirected * e_num);
  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
    auto src_ptr = src.Ptr<IdType>();
    auto dst_ptr = dst.Ptr<IdType>();
    auto data_ptr = data.Ptr<IdType>();
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, e_num),
        [&](tbb::blocked_range<int64_t> r) {
          for (int64_t i = r.begin(); i < r.end(); i++) {
            if (to_undirected) {
              edge_vec.at(i * 2) = {src_ptr[i], dst_ptr[i], data_ptr[i]};
              edge_vec.at(i * 2 + 1) = {dst_ptr[i], src_ptr[i], data_ptr[i]};
            } else {
              edge_vec.at(i) = {src_ptr[i], dst_ptr[i], data_ptr[i]};
            }
          }
        });
  });
  LOG(INFO) << "COO2CSR start sorting";
  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t cur_e_num = edge_vec.size();
  IdArray indptr = aten::NewIdArray(v_num + 1);
  IdArray indices = aten::NewIdArray(cur_e_num);
  IdArray retdata = aten::NewIdArray(cur_e_num);

  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
  memset(retdata.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  int64_t *indices_ptr = indices.Ptr<int64_t>();
  int64_t *retdata_ptr = retdata.Ptr<int64_t>();
  LOG(INFO) << "COO2CSR compute degree";

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const auto &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
          retdata_ptr[i] = e._data;
        }
      });

  LOG(INFO) << "COO2CSR compute indptr";
  auto out_start = indptr.Ptr<int64_t>();
  auto out_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
  LOG(INFO) << "COO2CSR e_num after convert " << cur_e_num;

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices, retdata};
}  // COO2CSR

static std::tuple<IdArray, IdArray, IdArray> MakeSym(
    NDArray in_indptr, NDArray in_indices, NDArray data) {
  auto dtype = in_indices->dtype;
  int64_t e_num = in_indices.NumElements();
  int64_t v_num = in_indptr.NumElements() - 1;
  LOG(INFO) << "MakeSym v_num: " << v_num << " | e_num: " << e_num;
  typedef std::vector<EdgeWithData> EdgeVec;
  EdgeVec edge_vec;
  edge_vec.resize(e_num * 2);
  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
    auto _in_indptr = in_indptr.Ptr<IdType>();
    auto _in_indices = in_indices.Ptr<IdType>();
    auto data_ptr = data.Ptr<IdType>();
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, v_num),
        [&](tbb::blocked_range<int64_t> r) {
          for (int64_t v = r.begin(); v < r.end(); v++) {
            IdType start = _in_indptr[v];
            IdType end = _in_indptr[v + 1];
            for (IdType i = start; i < end; i++) {
              IdType u = _in_indices[i];
              edge_vec.at(i * 2) = {v, u, data_ptr[i]};
              edge_vec.at(i * 2 + 1) = {u, v, data_ptr[i]};
            }
          }
        });
  });
  LOG(INFO) << "MakeSym start sorting";
  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t cur_e_num = edge_vec.size();
  IdArray indptr = aten::NewIdArray(v_num + 1);
  IdArray indices = aten::NewIdArray(cur_e_num);
  IdArray retdata = aten::NewIdArray(cur_e_num);

  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
  memset(retdata.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  int64_t *indices_ptr = indices.Ptr<int64_t>();
  int64_t *retdata_ptr = retdata.Ptr<int64_t>();
  LOG(INFO) << "MakeSym compute degree";

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const auto &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
          retdata_ptr[i] = e._data;
        }
      });

  LOG(INFO) << "MakeSym compute indptr";
  auto out_start = indptr.Ptr<int64_t>();
  auto out_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
  LOG(INFO) << "MakeSym e_num after convert " << cur_e_num;

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices, retdata};
}  // MakeSym

static std::tuple<IdArray, IdArray> MakeSym(
    NDArray in_indptr, NDArray in_indices) {
  auto dtype = in_indices->dtype;
  int64_t e_num = in_indices.NumElements();
  int64_t v_num = in_indptr.NumElements() - 1;
  LOG(INFO) << "MakeSym v_num: " << v_num << " | e_num: " << e_num;
  typedef std::vector<Edge> EdgeVec;
  EdgeVec edge_vec;
  edge_vec.resize(e_num * 2);
  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
    auto _in_indptr = in_indptr.Ptr<IdType>();
    auto _in_indices = in_indices.Ptr<IdType>();
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, v_num),
        [&](tbb::blocked_range<int64_t> r) {
          for (int64_t v = r.begin(); v < r.end(); v++) {
            int64_t start = _in_indptr[v];
            int64_t end = _in_indptr[v + 1];
            for (int64_t i = start; i < end; i++) {
              int64_t u = _in_indices[i];
              edge_vec.at(i * 2) = {v, u};
              edge_vec.at(i * 2 + 1) = {u, v};
            }
          }
        });
  });
  LOG(INFO) << "MakeSym start sorting";
  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t cur_e_num = edge_vec.size();
  IdArray indptr = aten::NewIdArray(v_num + 1);
  IdArray indices = aten::NewIdArray(cur_e_num);
  // IdArray retdata = aten::NewIdArray(cur_e_num);

  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
  // memset(retdata.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  int64_t *indices_ptr = indices.Ptr<int64_t>();
  // int64_t * retdata_ptr = retdata.Ptr<int64_t>();
  LOG(INFO) << "MakeSym compute degree";

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const auto &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
          // retdata_ptr[i] = e._data;
        }
      });

  LOG(INFO) << "COO2CSR compute indptr";
  auto out_start = indptr.Ptr<int64_t>();
  auto out_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
  LOG(INFO) << "COO2CSR e_num after convert " << cur_e_num;

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices};
}  // MakeSym

static std::tuple<IdArray, IdArray> COO2CSR(
    int64_t v_num, NDArray src, NDArray dst, bool to_undirected) {
  auto dtype = src->dtype;
  int64_t e_num = src.NumElements();
  LOG(INFO) << "COO2CSR v_num: " << v_num << " | e_num: " << e_num;
  typedef std::vector<Edge> EdgeVec;
  EdgeVec edge_vec;
  edge_vec.resize(e_num + to_undirected * e_num);
  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
    auto src_ptr = src.Ptr<IdType>();
    auto dst_ptr = dst.Ptr<IdType>();
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, e_num),
        [&](tbb::blocked_range<int64_t> r) {
          for (int64_t i = r.begin(); i < r.end(); i++) {
            if (to_undirected) {
              edge_vec.at(i * 2) = {src_ptr[i], dst_ptr[i]};
              edge_vec.at(i * 2 + 1) = {dst_ptr[i], src_ptr[i]};
            } else {
              edge_vec.at(i) = {src_ptr[i], dst_ptr[i]};
            }
          }
        });
  });
  LOG(INFO) << "COO2CSR start sorting";
  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t cur_e_num = edge_vec.size();
  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  IdArray indptr = aten::NewIdArray(v_num + 1);
  IdArray indices = aten::NewIdArray(cur_e_num);
  memset(indptr.Ptr<void>(), 0, sizeof(idx_t) * (v_num + 1));
  memset(indices.Ptr<void>(), 0, sizeof(idx_t) * cur_e_num);
  int64_t *indices_ptr = indices.Ptr<int64_t>();
  int64_t *out_start = indptr.Ptr<int64_t>();
  LOG(INFO) << "COO2CSR compute degree";
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const Edge &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
        }
      });

  LOG(INFO) << "COO2CSR compute indptr";
  auto out_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
  LOG(INFO) << "COO2CSR e_num after convert " << cur_e_num;

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices};

}  // COO2CSR

inline bool nextSNAPline(
    std::ifstream &infile, std::string &line, std::istringstream &iss,
    int64_t &src, int64_t &dest) {
  do {
    if (!getline(infile, line)) return false;
  } while (line.length() == 0 || line[0] == '#');
  iss.clear();
  iss.str(line);
  return !!(iss >> src >> dest);
}

inline void getID(std::vector<int64_t> &idMap, int64_t &id, int64_t &nextID) {
  if (idMap.size() <= id) {
    idMap.resize(id + 4096, -1);
  }

  if (idMap.at(id) == -1) {
    idMap.at(id) = nextID;
    nextID++;
  }
  id = idMap.at(id);
}

static std::pair<NDArray, NDArray> LoadSNAP(
    std::filesystem::path data_file, bool to_sym) {
  LOG(INFO) << "Loading from file: " << data_file;
  LOG(INFO) << "Convert to symmetric: " << to_sym;
  typedef std::vector<Edge> EdgeVec;
  EdgeVec edge_vec;
  std::vector<int64_t> idMap;
  int64_t max_id = 0, src = 0, dst = 0, nextID = 0, line_num = 0;
  std::string line;
  std::istringstream iss;
  std::ifstream infile(data_file.c_str());
  while (nextSNAPline(infile, line, iss, src, dst)) {
    if (src == dst) continue;
    getID(idMap, src, nextID);
    getID(idMap, dst, nextID);
    max_id = std::max(max_id, src);
    max_id = std::max(max_id, dst);
    edge_vec.push_back({src, dst});
    if (to_sym) edge_vec.push_back({dst, src});  // convert to symmetric graph
    line_num++;
    if (line_num % int(2e7) == 0) {
      LOG(INFO) << "LoadSNAP Read: " << line_num / int(1e6) << "M edges";
    }
  }

  tbb::parallel_sort(edge_vec.begin(), edge_vec.end());

  edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
  edge_vec.shrink_to_fit();
  int64_t v_num = nextID;
  int64_t e_num = edge_vec.size();
  LOG(INFO) << "LoadSNAP v_num: " << v_num << " | e_num: " << e_num;

  std::vector<std::atomic<int64_t>> degree(v_num + 1);
  IdArray indptr = aten::NewIdArray(v_num + 1);
  IdArray indices = aten::NewIdArray(e_num);

  int64_t *indices_ptr = indices.Ptr<int64_t>();
  int64_t *out_start = indptr.Ptr<int64_t>();
  LOG(INFO) << "LoadSNAP compute degree";
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, edge_vec.size()),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          const Edge &e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
        }
      });

  LOG(INFO) << "LoadSNAP compute indptr";
  auto out_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);

  CHECK_EQ(out_end - out_start, v_num + 1);
  CHECK_EQ(out_start[v_num], edge_vec.size());
  return {indptr, indices};
}

}  // namespace dgl::dev

#endif  // DGL_DEVCOO2CSR_H
