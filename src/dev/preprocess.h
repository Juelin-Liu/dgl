//
// Created by juelin on 2/22/24.
//

#ifndef DGL_PREPROCESS_H
#define DGL_PREPROCESS_H
#include <dgl/array.h>
#include <oneapi/tbb/parallel_for.h>
#include <algorithm>
#include <numeric>

namespace dgl::dev
{
static NDArray ExpandIndptr(NDArray in_indptr) {
  auto dtype = in_indptr->dtype;
  int64_t v_num = in_indptr.NumElements() - 1;
  ATEN_ID_TYPE_SWITCH(dtype, IdType, {
    auto _in_indptr = in_indptr.Ptr<IdType>();
    IdType e_num = _in_indptr[v_num];

    IdArray indices = NDArray::Empty({e_num}, dtype, in_indptr->ctx);
    auto _indices = indices.Ptr<IdType>();
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, v_num),
        [&](tbb::blocked_range<int64_t> r) {
          for (int64_t v = r.begin(); v < r.end(); v++) {
            IdType start = _in_indptr[v];
            IdType end = _in_indptr[v + 1];
            for (IdType i = start; i < end; i++) {
              _indices[i] = v;
            }
          }
        });
    return indices;
  });
  return aten::NullArray();
};


// Remove vertices with 0 degree
static std::pair<NDArray, NDArray> ReindexCSR(
    NDArray in_indptr, NDArray in_indices) {
  int64_t v_num = in_indptr.NumElements() - 1;
  int64_t e_num = in_indices.NumElements();
  LOG(INFO) << "ReindexCSR v_num before compact = " << v_num;

  IdArray out_indices = in_indices;
  int64_t *_in_indices = in_indices.Ptr<int64_t>();
  int64_t *_out_indices = out_indices.Ptr<int64_t>();
  const int64_t *_in_indptr = in_indptr.Ptr<int64_t>();
  std::vector<int64_t> degree(v_num, 0);
  std::vector<int64_t> org2new(v_num, 0);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          degree.at(i) = _in_indptr[i + 1] - _in_indptr[i];
        }
      });
  int64_t cur_v_num = 0;

  std::vector<int64_t> compacted_degree;

  for (int64_t i = 0; i < v_num; i++) {
    int64_t d = degree.at(i);
    if (d > 0) {
      org2new[i] = cur_v_num++;
      compacted_degree.push_back(d);
    }
  }

  compacted_degree.push_back(0);  // make exclusive scan work

  IdArray out_indptr = aten::NewIdArray(cur_v_num + 1);
  auto out_indptr_start = out_indptr.Ptr<int64_t>();
  auto out_indptr_end = std::exclusive_scan(
      compacted_degree.begin(), compacted_degree.end(), out_indptr_start, 0ll);
  CHECK_EQ(out_indptr_start[cur_v_num], e_num);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, e_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          _out_indices[i] = org2new[_in_indices[i]];
        }
      });

  LOG(INFO) << "ReindexCSR v_num after = " << cur_v_num;
  return {out_indptr, out_indices};
}

// Remove edges with 0 flag
static NDArray CompactCSR(NDArray in_indptr, NDArray flag) {
  int64_t v_num = in_indptr.NumElements() - 1;
  int64_t e_num = flag.NumElements();
  // CHECK_EQ(flag->dtype, DGLDataTypeTraits<bool>::dtype);
  LOG(INFO) << "ReindexCSR e_num before compact = " << e_num;
  bool *_in_indices = flag.Ptr<bool>();
  const int64_t *_in_indptr = in_indptr.Ptr<int64_t>();
  std::vector<int64_t> degree(v_num + 1);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          int64_t start = _in_indptr[i];
          int64_t end = _in_indptr[i + 1];
          for (int64_t j = start; j < end; j++) {
            degree.at(i) += (_in_indices[j]);
          }
        }
      });

  IdArray out_indptr = aten::NewIdArray(v_num + 1);
  auto out_indptr_start = out_indptr.Ptr<int64_t>();
  auto out_indptr_end =
      std::exclusive_scan(degree.begin(), degree.end(), out_indptr_start, 0ll);

  LOG(INFO) << "ReindexCSR e_num after = " << out_indptr_start[v_num];
  return out_indptr;
}

// Remove adjacency lists of vertices with 0 flag
static std::pair<NDArray, NDArray> PartitionCSR(NDArray in_indptr, NDArray in_indices, NDArray flag)
{
  CHECK_EQ(in_indices->dtype.bits, 64);
  CHECK_EQ(in_indptr->dtype.bits, 64);
  CHECK_EQ(flag->dtype.bits, 8);
  CHECK_EQ(flag.NumElements(), in_indptr.NumElements() - 1);

  int64_t v_num = in_indptr.NumElements() - 1;
  int64_t e_num = in_indices.NumElements();

  LOG(INFO) << "PartitionCSR e_num before compact = " << e_num;

  const auto _in_indices = in_indices.Ptr<int64_t >();
  const auto _in_indptr = in_indptr.Ptr<int64_t>();
  const auto _flag = flag.Ptr<bool >();

  std::vector<int64_t> degree(v_num + 1);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          if (_flag[i]) {
            degree.at(i) += _in_indptr[i + 1] - _in_indptr[i];
          } else {
            degree.at(i) = 0;
          }
        }
      });

  IdArray out_indptr = aten::NewIdArray(v_num + 1);
  auto _out_indptr = out_indptr.Ptr<int64_t>();
  auto _out_indptr_end = std::exclusive_scan(degree.begin(), degree.end(), _out_indptr, 0ll);
  const auto new_enum = _out_indptr[v_num];
  IdArray out_indices = aten::NewIdArray(new_enum);
  auto _out_indices = out_indices.Ptr<int64_t >();
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, v_num),
      [&](tbb::blocked_range<int64_t> r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
          if (_flag[i]) {
            auto out_start = _out_indptr[i];
            auto out_end = _out_indptr[i + 1];
            auto in_start = _in_indptr[i];
            auto in_end = _in_indptr[i + 1];
            assert(out_end - out_start == in_end - in_start);
            std::copy_n(_in_indices + in_start, in_end - in_start, _out_indices + out_start);
          }
        }
      });
  LOG(INFO) << "ReindexCSR e_num after = " << new_enum;

  return std::make_pair(out_indptr, out_indices);
}

} // dgl:dev
#endif  // DGL_PREPROCESS_H
