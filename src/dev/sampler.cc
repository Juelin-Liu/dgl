//
// Created by juelinliu on 12/10/23.
//

#include "sampler.h"

namespace dgl::dev {
using namespace runtime;
DGL_REGISTER_GLOBAL("dev._CAPI_SetGraph")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray indptr = args[0];
      NDArray indices = args[1];
      NDArray data = args[2];
      auto sampler = SamplerObject::Global();
      sampler->SetGraph(indptr, indices, data);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_SetFanout")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      List<Value> fanout_list = args[0];
      std::vector<int64_t> fanouts;
      for (const auto &fanout : fanout_list) {
        fanouts.insert(fanouts.begin(), static_cast<int64_t>(fanout->data));
      }
      auto sampler = SamplerObject::Global();
      sampler->SetFanout(fanouts);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_SetPoolSize")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t pool_size = args[0];
      auto sampler = SamplerObject::Global();
      sampler->SetPoolSize(pool_size);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_SampleBatch")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray seeds = args[0];
      bool replace = args[1];
      auto sampler = SamplerObject::Global();
      *rv = sampler->SampleBatch(seeds, replace);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      bool should_reindex = args[2];
      auto sampler = SamplerObject::Global();
      *rv = sampler->GetBlock(batch_id, layer, should_reindex);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetInputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = SamplerObject::Global();
      *rv = sampler->GetInputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetOutputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = SamplerObject::Global();
      *rv = sampler->GetOutputNode(batch_id);
    });
}  // namespace dgl::dev