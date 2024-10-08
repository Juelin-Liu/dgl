//
// Created by juelin on 2/6/24.
//

#include "batch_sampler.h"

#include "dgl/aten/macro.h"

namespace dgl::dev {
using namespace runtime;
DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_SetGraph")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray indptr = args[0];
      NDArray indices = args[1];
      NDArray data = args[2];
      auto sampler = BatchSampler::Global();
      sampler->SetGraph(indptr, indices, data);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_SetFanout")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      List<Value> fanout_list = args[0];
      std::vector<int64_t> fanouts;
      for (const auto &fanout : fanout_list) {
        fanouts.insert(fanouts.begin(), static_cast<int64_t>(fanout->data));
      }
      auto sampler = BatchSampler::Global();
      sampler->SetFanout(fanouts);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_SetPoolSize")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t pool_size = args[0];
      auto sampler = BatchSampler::Global();
      sampler->SetPoolSize(pool_size);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_SampleBatch")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray seeds = args[0];
      bool replace = args[1];
      auto sampler = BatchSampler::Global();
      *rv = sampler->SampleBatch(seeds, replace);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_SampleBatches")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray seed_arr = args[0];
      const int64_t arr_len = seed_arr.NumElements();
      const int64_t slice_len = args[1];
      const bool replace = args[2];
      const int64_t batch_layer = args[3];
      auto sampler = BatchSampler::Global();
      std::vector<NDArray> seeds;
      int64_t offset = 0;
      while (offset < arr_len) {
        int64_t start = offset;
        int64_t end = std::min(offset + slice_len, arr_len);
        int64_t view_len = end - start;
        seeds.push_back(
            seed_arr.CreateView({view_len}, seed_arr->dtype, offset));
        offset = end;
      }

      int64_t num_elem = 0;
      for (const auto &seed : seeds) {
        num_elem += seed.NumElements();
      }
      CHECK_EQ(num_elem, arr_len);
      *rv = sampler->SampleBatches(seeds, replace, batch_layer);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      bool should_reindex = args[2];
      auto sampler = BatchSampler::Global();
      *rv = sampler->GetBlock(batch_id, layer, should_reindex);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_GetInputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = BatchSampler::Global();
      *rv = sampler->GetInputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_GetOutputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = BatchSampler::Global();
      *rv = sampler->GetOutputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_GetBlockData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      auto sampler = BatchSampler::Global();
      *rv = sampler->GetBlockData(batch_id, layer);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_BatchSampler_Increment")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray count = args[0];
      NDArray row = args[1];
      ATEN_ID_TYPE_SWITCH(count->dtype, CounterType, {
        ATEN_ID_TYPE_SWITCH(row->dtype, IndexType, {
          Increment<kDGLCUDA, CounterType, IndexType>(count, row);
        });
      });
    });
}  // namespace dgl::dev