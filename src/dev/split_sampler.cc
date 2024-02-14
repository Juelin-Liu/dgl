//
// Created by juelin on 2/13/24.
//

#include "split_sampler.h"
#include "array_scatter.h"
#include "cuda/all2all.h"

namespace dgl::dev {
using namespace runtime;

DGL_REGISTER_GLOBAL("dev._CAPI_Split_ScatterForward")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      ScatteredArray scatter_array = args[0];
      NDArray feat = args[1];
      int rank = args[2];
      int world_size = args[3];
      *rv = scatter_array->shuffle_forward(feat, rank, world_size);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_ScatterBackward")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      ScatteredArray scatter_array = args[0];
      NDArray grads = args[1];
      int rank = args[2];
      int world_size = args[3];
      *rv = scatter_array->shuffle_backward(grads, rank, world_size);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_GetScatteredArrayObject")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray frontier = args[0];
      NDArray partition_map = args[1];
      int num_partitions = args[2];
      int rank = args[3];
      int world_size = args[4];
      ScatteredArray scatter_array = ScatteredArray::Create(frontier->shape[0] * 2, 4, frontier->ctx, frontier->dtype);
      Scatter(
          rank, world_size, num_partitions, frontier, partition_map,
          scatter_array);
      *rv = scatter_array;
    });



DGL_REGISTER_GLOBAL("dev._CAPI_InitNccl")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t rank = args[0];
      int64_t nranks = args[1];
      std::string unique_id = args[2];
      ncclUniqueId nccl_id;
      memcpy(nccl_id.internal, unique_id.c_str(), sizeof(char) * unique_id.size());
      auto res = ncclCommInitRank(getNcclPtr().get(), nranks, nccl_id, rank);
      CHECK_EQ(res, ncclSuccess);
      LOG(INFO) << "Initializing NCCL at rank " << rank << " comm: " << getNccl();
    });

DGL_REGISTER_GLOBAL("dev._CAPI_GetUniqueId")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      ncclUniqueId  uniqueId;
      ncclGetUniqueId(&uniqueId);
      *rv = std::string(uniqueId.internal);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_SetGraph")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray indptr = args[0];
      NDArray indices = args[1];
      NDArray data = args[2];
      auto sampler = SplitSampler::Global();
      sampler->setGraph(indptr, indices, data);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_SetPartitionMap")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t num_partitions = args[0];
      NDArray partition_map = args[1];
      auto sampler = SplitSampler::Global();
      sampler->setPartitionMap(partition_map);
      sampler->setNumPartitions(num_partitions);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_SetFanout")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      List<Value> fanout_list = args[0];
      std::vector<int64_t> fanouts;
      for (const auto &fanout : fanout_list) {
        fanouts.insert(fanouts.begin(), static_cast<int64_t>(fanout->data));
      }
      auto sampler = SplitSampler::Global();
      sampler->setFanouts(fanouts);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_UseBitmap")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      bool use_bitmap = args[0];
      auto sampler = SplitSampler::Global();
      sampler->useBitmap(use_bitmap);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_SetNumDP")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t num_dp = args[0];
      auto sampler = SplitSampler::Global();
      sampler->setNumDP(num_dp);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_SetRank")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t rank = args[0];
      int64_t world_size = args[1];
      auto sampler = SplitSampler::Global();
      sampler->setRank(rank, world_size);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_SampleBatch")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      NDArray seeds = args[0];
      bool replace = args[1];
      auto sampler = SplitSampler::Global();
      *rv = sampler->sampleOneBatch(seeds, replace);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_GetBlock")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      bool should_reindex = args[2];
      auto sampler = SplitSampler::Global();
      *rv = sampler->getBlock(batch_id, layer, should_reindex);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_GetInputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = SplitSampler::Global();
      *rv = sampler->getInputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_GetOutputNode")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      auto sampler = SplitSampler::Global();
      *rv = sampler->getOutputNode(batch_id);
    });

DGL_REGISTER_GLOBAL("dev._CAPI_Split_GetBlockData")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      int64_t batch_id = args[0];
      int64_t layer = args[1];
      auto sampler = SplitSampler::Global();
      *rv = sampler->getBlockData(batch_id, layer);
    });
}  // namespace dgl::dev