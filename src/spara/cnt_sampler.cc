//
// Created by juelin on 6/16/24.
//

#include "cnt_sampler.h"
#include "dgl/aten/macro.h"

namespace dgl::dev {
using namespace runtime;
DGL_REGISTER_GLOBAL("dev._CAPI_Cnt_SetGraph").set_body([](DGLArgs args, DGLRetValue *rv) {
  int32_t device_id = args[0];
  NDArray indptr = args[1];
  NDArray indices = args[2];
  auto sampler = CntSampler::Global();
  sampler->setGraph(device_id, indptr, indices);
});

DGL_REGISTER_GLOBAL("dev._CAPI_Cnt_SetFanout")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      List<Value> fanout_list = args[0];
      std::vector<int64_t> fanouts;
      for (const auto &fanout : fanout_list) {
        fanouts.insert(fanouts.begin(), static_cast<int64_t>(fanout->data));
      }
      auto sampler = CntSampler::Global();
      sampler->setFanouts(fanouts);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Cnt_SampleBatch")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray seeds = args[0];
      bool replace = args[1];
      auto sampler = CntSampler::Global();
      *rv = sampler->sampleOneBatch(seeds, replace);
    });


DGL_REGISTER_GLOBAL("dev._CAPI_Cnt_GetNodeFreq")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      auto sampler = CntSampler::Global();
      *rv = sampler->getNodeFreq();
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Cnt_GetEdgeFreq")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      auto sampler = CntSampler::Global();
      *rv = sampler->getEdgeFreq();
    });
}