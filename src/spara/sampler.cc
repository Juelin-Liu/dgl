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
      auto sampler = Sampler::Global();
      sampler->setGraph(indptr, indices, data);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_SetFanout")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      List<Value> fanout_list = args[0];
      std::vector<int64_t> fanouts;
      for (const auto &fanout : fanout_list) {
        fanouts.insert(fanouts.begin(), static_cast<int64_t>(fanout->data));
      }
      auto sampler = Sampler::Global();
      sampler->setFanouts(fanouts);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_UseBitmap")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      bool use_bitmap = args[0];
      auto sampler = Sampler::Global();
      sampler->useBitmap(use_bitmap);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_SampleBatch")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray seeds = args[0];
      bool replace = args[1];
      auto sampler = Sampler::Global();
      *rv = sampler->sampleOneBatch(seeds, replace);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      bool should_reindex = args[2];
      auto sampler = Sampler::Global();
      *rv = sampler->getBlock(batch_id, layer, should_reindex);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetInputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = Sampler::Global();
      *rv = sampler->getInputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetOutputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = Sampler::Global();
      *rv = sampler->getOutputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetBlockData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      auto sampler = Sampler::Global();
      *rv = sampler->getBlockData(batch_id, layer);
    });
}  // namespace dgl::dev