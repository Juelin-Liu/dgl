//
// Created by juelinliu on 12/11/23.
//

#include "array_ops.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include <cub/cub.cuh>
namespace dgl::dev
{
    namespace impl 
    {
        /**
            * @brief Compute the size of each row in the sampled CSR, without replacement.
            *
            * @tparam IdType The type of node and edge indexes.
            * @param num_rows The number of rows to pick.
            * @param in_rows The set of rows to pick.
            * @param dflag The mask indicating the presense of number in row
            * `in_rows` (output).
            */
        template<typename IdType>
        __global__ void _Mask(const IdType * in_row, const int64_t num_rows, bool* dflag) {
            const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
            if (tIdx < num_rows) {
                dflag[ in_row[tIdx] ] = 1;
            }
        }
    }



    void Reset(NDArray& bitmap, int64_t capacity){
         auto stream = runtime::getCurrentCUDAStream();
        CUDA_CALL(cudaMemsetAsync(bitmap.Ptr<bool>(), 0, capacity, stream));
    };


    int64_t NumItem(const NDArray& bitmap, int64_t capacity){
        auto ctx = bitmap->ctx;
        auto device = runtime::DeviceAPI::Get(ctx);
        int64_t *d_num_item = static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
        int64_t h_num_item = 0;
        size_t temp_storage_bytes = 0;
        auto stream = runtime::getCurrentCUDAStream();
        const bool * dflag = bitmap.Ptr<const bool>();
        CUDA_CALL(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, dflag, d_num_item, capacity, stream));
        void * d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
        CUDA_CALL(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dflag, d_num_item, capacity, stream));
        CUDA_CALL(cudaMemcpyAsync(&h_num_item, d_num_item, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        device->StreamSync(ctx, stream);
        device->FreeWorkspace(ctx, d_temp_storage);
        device->FreeWorkspace(ctx, d_num_item);
        return h_num_item;
    };

    template<DGLDeviceType XPU, typename IdType>
    void Mask(NDArray& bitmap, int64_t capacity, NDArray row){
        CHECK(row.NumElements() <= capacity);
        const dim3 block(256);
        const dim3 grid((row.NumElements() + block.x - 1) / block.x);
        auto stream = runtime::getCurrentCUDAStream();
        bool * dflag = bitmap.Ptr<bool>();
        CUDA_KERNEL_CALL(impl::_Mask<IdType>, grid, block, 0, stream, static_cast<IdType* >(row->data), row.NumElements(), dflag);
    };

    template<DGLDeviceType XPU, typename IdType>
    NDArray Flagged(const NDArray& bitmap, int64_t capacity, DGLContext ctx){
        auto device = runtime::DeviceAPI::Get(ctx);
        auto stream = runtime::getCurrentCUDAStream();
        int64_t *d_num_item = static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
        int64_t num_item = NumItem(bitmap, capacity);
        NDArray flagged = NDArray::Empty({num_item}, DGLDataTypeTraits<IdType>::dtype, ctx);
        IdType *d_out = static_cast<IdType*>(flagged->data);
        auto d_in = cub::CountingInputIterator<IdType>(0);
        size_t   temp_storage_bytes = 0;
        const bool * dflag = bitmap.Ptr<const bool>();

        cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_in, dflag, d_out, d_num_item, capacity, stream);
        void     *d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, dflag, d_out, d_num_item, capacity, stream);

        device->StreamSync(ctx, stream);
        device->FreeWorkspace(ctx, d_temp_storage);
        device->FreeWorkspace(ctx, d_num_item);
        return flagged;
    };

    template void Mask<kDGLCUDA, int16_t>(NDArray&, int64_t, NDArray);
    template void Mask<kDGLCUDA, int32_t>(NDArray&, int64_t, NDArray);
    template void Mask<kDGLCUDA, int64_t>(NDArray&, int64_t, NDArray);

    template NDArray Flagged<kDGLCUDA, int16_t>(const NDArray&, int64_t, DGLContext);
    template NDArray Flagged<kDGLCUDA, int32_t>(const NDArray&, int64_t, DGLContext);
    template NDArray Flagged<kDGLCUDA, int64_t>(const NDArray&, int64_t, DGLContext);

}