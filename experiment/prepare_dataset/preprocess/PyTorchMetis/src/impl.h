#pragma once

#include <cstdint>
#include <iostream>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>                                                                         
#include <oneapi/tbb/parallel_sort.h>
#include <metis.h>

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


torch::Tensor expand_indptr(torch::Tensor indptr){
    int64_t v_num = indptr.size(0) - 1;
    auto _indptr = indptr.accessor<int64_t, 1>();
    int64_t e_num = _indptr[v_num];
    auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    torch::Tensor ret = torch::empty(e_num, tensor_opts);
    auto _ret = ret.accessor<int64_t, 1>();

    tbb::parallel_for( tbb::blocked_range<int64_t >(0, v_num),
                    [&](tbb::blocked_range<int64_t > r)
    {
        for (int64_t v=r.begin(); v<r.end(); v++)
        {
            int64_t start = _indptr[v];
            int64_t end = _indptr[v+1];
            for (int64_t i = start; i < end; i++) {
                _ret[i] = v;
            }
        }
    });

    return ret;
};

torch::Tensor metis_assignment(int64_t num_partitions, torch::Tensor indptr, torch::Tensor indices, torch::Tensor node_weight, torch::Tensor edge_weight, bool obj_cut){
    int64_t nparts = num_partitions;
    int64_t nvtxs = indptr.size(0) - 1;
    int64_t num_edge = indices.size(0);
    int64_t ncon = 1;
    if (node_weight.size(0)) {
        int64_t nvwgt = node_weight.size(0);
        ncon = nvwgt / nvtxs;
        assert(nvwgt % nvtxs == 0);
    };

    if (edge_weight.size(0)) {
        assert(edge_weight.size(0) == num_edge);
    }

    std::cout << "metis_assignment num_part: " << num_partitions << std::endl; 
    // std::cout << "indptr: " << indptr << std::endl;
    // std::cout << "indices: " << indices << std::endl;
    // std::cout << "node_weight: " << node_weight << std::endl;
    // std::cout << "edge_weight: " << edge_weight << std::endl;
    auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    torch::Tensor ret = torch::empty(nvtxs, tensor_opts);
    int64_t *part = static_cast<int64_t*>(ret.mutable_data_ptr());

    // std::vector<int64_t> ret(nvtxs, 0);
    // int64_t *part = ret.data();
    int64_t *xadj = static_cast<int64_t*>(indptr.data_ptr());
    int64_t *adjncy =static_cast<int64_t*>(indices.data_ptr());

    int64_t *vwgt = nullptr;
    int64_t *ewgt = nullptr;

    if(node_weight.size(0)) vwgt = static_cast<int64_t*>(node_weight.data_ptr());
    if(edge_weight.size(0)) ewgt = static_cast<int64_t*>(edge_weight.data_ptr());

    int64_t objval = 0;

    int64_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_ONDISK] = 1;
    options[METIS_OPTION_NITER] = 1;
    options[METIS_OPTION_NIPARTS] = 1;
    options[METIS_OPTION_DROPEDGES] = edge_weight.size(0) == 0;
    // options[METIS_OPTION_DBGLVL] = METIS_DBG_COARSEN | METIS_DBG_INFO | METIS_DBG_TIME;

    int flag = METIS_PartGraphKway(&nvtxs, 
    &ncon, xadj, adjncy, vwgt,  NULL, ewgt, &nparts,
     NULL, NULL, options, &objval, part);

    if (obj_cut) {
        std::cout << "Partition a graph with " << nvtxs << " nodes and "
                << num_edge << " edges into " << num_partitions << " parts and "
                << "get " << objval << " edge cuts" << std::endl;
    } else {
        std::cout << "Partition a graph with " << nvtxs << " nodes and "
                << num_edge << " edges into " << num_partitions << " parts and "
                << "the communication volume is " << objval << std::endl;
    }

    switch (flag) {
        case METIS_OK:
            return ret;
        case METIS_ERROR_INPUT:
            std::cerr << "Error in Metis partitioning: input error";
        case METIS_ERROR_MEMORY:
            std::cerr << "Error in Metis partitioning: cannot allocate memory";
        default:
            std::cerr << "Error in Metis partitioning: other errors";
    };
    exit(-1);
};


torch::Tensor compact_indptr(torch::Tensor in_indptr, torch::Tensor flag) {
    int64_t v_num = in_indptr.size(0) - 1;
    int64_t e_num = flag.size(0);
    std::cout << "ReindexCSR e_num before compact = " << e_num << std::endl;
    auto _in_indices = flag.accessor<bool, 1>();
    auto _in_indptr = in_indptr.accessor<int64_t, 1>();
    std::vector<int64_t> degree(v_num + 1, 0);

    tbb::parallel_for( tbb::blocked_range<int64_t >(0, v_num),
                      [&](tbb::blocked_range<int64_t > r)
    {
        for (int64_t i=r.begin(); i<r.end(); i++)
        {
          int64_t start = _in_indptr[i];
          int64_t end = _in_indptr[i + 1];
          for (int64_t j = start; j < end; j++){
            degree.at(i) += (_in_indices[j]);
          }
        }
    });
    auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    torch::Tensor ret = torch::empty(v_num + 1, tensor_opts);
    auto out_indptr_start = static_cast<int64_t *>(ret.mutable_data_ptr());
    auto out_indptr_end = std::exclusive_scan(degree.begin(), degree.end(), out_indptr_start, 0ll);
    
    std::cout << "ReindexCSR e_num after = " << out_indptr_start[v_num] << std::endl;
    return ret;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> make_sym(torch::Tensor in_indptr, torch::Tensor in_indices, torch::Tensor data) {
  if (data.size(0) == 0) {
    int64_t e_num = in_indices.size(0);
    int64_t v_num = in_indptr.size(0) - 1;
    std::cout << "MakeSym v_num: " << v_num << " | e_num: " << e_num << std::endl;
    typedef std::vector<Edge> EdgeVec;
    EdgeVec edge_vec;
    edge_vec.resize(e_num * 2);
    auto _in_indptr = in_indptr.accessor<int64_t, 1>();
    auto _in_indices = in_indices.accessor<int64_t, 1>();
    tbb::parallel_for( tbb::blocked_range<int64_t >(0, v_num),
                      [&](tbb::blocked_range<int64_t > r)
    {
        for (int64_t v=r.begin(); v<r.end(); v++)
        {
          int64_t start = _in_indptr[v];
          int64_t end = _in_indptr[v+1];
          for (int64_t i = start; i < end; i++) {
            int64_t u = _in_indices[i];
            edge_vec.at(i * 2) = {v, u};
            edge_vec.at(i * 2 + 1) = {u, v};
          }
        }
    });
    std::cout << "MakeSym start sorting" << std::endl;
    tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
    edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
    edge_vec.shrink_to_fit();
    int64_t cur_e_num = edge_vec.size();

    auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

    torch::Tensor indptr = torch::empty(v_num + 1, tensor_opts);
    torch::Tensor indices = torch::empty(cur_e_num, tensor_opts);
    std::vector<std::atomic<int64_t>> degree(v_num + 1);
    auto indices_ptr = indices.accessor<int64_t, 1>();
    std::cout << "MakeSym compute degree" << std::endl;

    tbb::parallel_for( tbb::blocked_range<int64_t >(0, cur_e_num),
                      [&](tbb::blocked_range<int64_t > r)
    {
        for (int64_t i=r.begin(); i<r.end(); i++)
        {
          const auto& e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
        }
    });

    std::cout << "MakeSym compute indptr" << std::endl;
    auto out_start = static_cast<int64_t *>(indptr.mutable_data_ptr());
    auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
    std::cout << "MakeSym e_num after convert " << cur_e_num << std::endl;

    assert(out_start[v_num] == cur_e_num);
    return {indptr, indices, data};
  } else {
    assert(data.size(0) == in_indices.size(0));
      int64_t e_num = in_indices.size(0);
    int64_t v_num = in_indptr.size(0) - 1;
    std::cout << "MakeSym v_num: " << v_num << " | e_num: " << e_num << std::endl;
    typedef std::vector<EdgeWithData> EdgeVec;
    EdgeVec edge_vec;
    edge_vec.resize(e_num * 2);
    auto _in_indptr = in_indptr.accessor<int64_t, 1>();
    auto _in_indices = in_indices.accessor<int64_t, 1>();
    auto _in_data = data.accessor<int64_t, 1>();
    tbb::parallel_for( tbb::blocked_range<int64_t >(0, v_num),
                      [&](tbb::blocked_range<int64_t > r)
    {
        for (int64_t v=r.begin(); v<r.end(); v++)
        {
          int64_t start = _in_indptr[v];
          int64_t end = _in_indptr[v+1];
          for (int64_t i = start; i < end; i++) {
            int64_t u = _in_indices[i];
            int64_t d = _in_data[i];
            edge_vec.at(i * 2) = {v, u, d};
            edge_vec.at(i * 2 + 1) = {u, v, d};
          }
        }
    });
    std::cout << "MakeSym start sorting" << std::endl;
    tbb::parallel_sort(edge_vec.begin(), edge_vec.end());
    edge_vec.erase(std::unique(edge_vec.begin(), edge_vec.end()), edge_vec.end());
    edge_vec.shrink_to_fit();
    int64_t cur_e_num = edge_vec.size();

    auto tensor_opts = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

    torch::Tensor indptr = torch::empty(v_num + 1, tensor_opts);
    torch::Tensor indices = torch::empty(cur_e_num, tensor_opts);
    torch::Tensor retdata = torch::empty(cur_e_num, tensor_opts);

    std::vector<std::atomic<int64_t>> degree(v_num + 1);
    auto indices_ptr = indices.accessor<int64_t, 1>();
    auto retdata_ptr = retdata.accessor<int64_t, 1>();
    std::cout << "MakeSym compute degree" << std::endl;

    tbb::parallel_for( tbb::blocked_range<int64_t >(0, cur_e_num),
                      [&](tbb::blocked_range<int64_t > r)
    {
        for (int64_t i=r.begin(); i<r.end(); i++)
        {
          const auto& e = edge_vec.at(i);
          degree.at(e._src)++;
          indices_ptr[i] = e._dst;
          retdata_ptr[i] = e._data;
        }
    });

    std::cout << "MakeSym compute indptr" << std::endl;
    auto out_start = static_cast<int64_t *>(indptr.mutable_data_ptr());
    auto out_end = std::exclusive_scan(degree.begin(), degree.end(), out_start, 0ll);
    std::cout << "MakeSym e_num after convert " << cur_e_num << std::endl;

    assert(out_start[v_num] == cur_e_num);
    return {indptr, indices, retdata};

  }
}