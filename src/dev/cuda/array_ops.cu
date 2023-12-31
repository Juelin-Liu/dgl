//
// Created by juelinliu on 12/11/23.
//

#include "array_ops.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include <cstdint>
#include <cub/cub.cuh>
#include "../../array/cuda/atomic.cuh"

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
        __global__ void _Mask(const IdType * in_row, const int64_t num_rows, int8_t* dflag) {
            const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
            if (tIdx < num_rows) {
                dflag[ in_row[tIdx] ] = 1;
            }
        }

        template<typename IdType>
        __global__ void Increment(IdType * array, const int64_t array_len, const IdType * row, const int64_t num_item) {
            const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
            if (tIdx < num_item) {
                const IdType rIdx = row[tIdx];
                aten::cuda::AtomicAdd(array + rIdx, static_cast<IdType>(1));
            }
        }

        template<typename CounterType, typename IndexType>
        __global__ void Increment(CounterType * array, const int64_t array_len, 
                                const IndexType * row, const int64_t num_item) {
            const int64_t tIdx = threadIdx.x + blockIdx.x * blockDim.x;
            if (tIdx < num_item) {
                const IndexType rIdx = row[tIdx];
                aten::cuda::AtomicAdd(array + rIdx, static_cast<CounterType>(1));
            }
        }
    } // impl



    void Reset(NDArray& bitmap){
//            auto stream = runtime::getCurrentCUDAStream();
//        CUDA_CALL(cudaMemsetAsync(bitmap.Ptr<int8_t>(), 0, capacity, stream));
            CUDA_CALL(cudaMemset(bitmap.Ptr<int8_t>(), 0, bitmap.GetSize()));
    };


    int64_t NumItem(const NDArray& bitmap){
        auto ctx = bitmap->ctx;
        auto device = runtime::DeviceAPI::Get(ctx);
        int64_t *d_num_item = static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
        int64_t h_num_item = 0;
        size_t temp_storage_bytes = 0;
        auto stream = runtime::getCurrentCUDAStream();
        const int8_t * dflag = bitmap.Ptr<const int8_t>();
        const int64_t num_items = bitmap.NumElements();
        CUDA_CALL(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, dflag, d_num_item, num_items, stream));
        void * d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
        CUDA_CALL(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dflag, d_num_item, num_items, stream));
        CUDA_CALL(cudaMemcpyAsync(&h_num_item, d_num_item, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        device->StreamSync(ctx, stream);
        device->FreeWorkspace(ctx, d_temp_storage);
        device->FreeWorkspace(ctx, d_num_item);
        return h_num_item;
    };

    template<DGLDeviceType XPU, typename IdType>
    void Mask(NDArray& bitmap, NDArray row){
        const dim3 block(256);
        const dim3 grid((row.NumElements() + block.x - 1) / block.x);
        auto stream = runtime::getCurrentCUDAStream();
        int8_t * dflag = bitmap.Ptr<int8_t>();
        CUDA_KERNEL_CALL(impl::_Mask<IdType>, grid, block, 0, stream, static_cast<IdType* >(row->data), row.NumElements(), dflag);
    };

    template<DGLDeviceType XPU, typename IdType>
    NDArray Flagged(const NDArray& bitmap, DGLContext ctx){
        auto device = runtime::DeviceAPI::Get(ctx);
        auto stream = runtime::getCurrentCUDAStream();
        int64_t *d_num_item = static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
        int64_t num_item = NumItem(bitmap);
        int64_t v_num = bitmap.NumElements();
        auto d_in = cub::CountingInputIterator<IdType>(0);
        NDArray flagged = NDArray::Empty({num_item}, DGLDataTypeTraits<IdType>::dtype, ctx);
        IdType *d_out = static_cast<IdType*>(flagged->data);
        size_t   temp_storage_bytes = 0;
        const int8_t * dflag = bitmap.Ptr<const int8_t>();
        cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, d_in, dflag, d_out, d_num_item, v_num, stream);
        void     *d_temp_storage = device->AllocWorkspace(ctx, temp_storage_bytes);
        cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, dflag, d_out, d_num_item, v_num, stream);

        device->StreamSync(ctx, stream);
        device->FreeWorkspace(ctx, d_temp_storage);
        device->FreeWorkspace(ctx, d_num_item);
        return flagged;
    };
    
    template<DGLDeviceType XPU, typename IdType>
    void Increment(NDArray& count, const NDArray& row){
        CHECK_EQ(count->ctx, row->ctx);
        const dim3 block(256);
        const dim3 grid((row.NumElements() + block.x - 1) / block.x);
        auto stream = runtime::getCurrentCUDAStream();
        CUDA_KERNEL_CALL(impl::Increment, grid, block, 0, stream, count.Ptr<IdType>(), count.NumElements(), row.Ptr<IdType>(), row.NumElements());
        auto device = runtime::DeviceAPI::Get(count->ctx);
        device->StreamSync(count->ctx, stream);
    }; 

    template<DGLDeviceType XPU, typename CounterType, typename IndexType>
    void Increment(NDArray& count, const NDArray& row){
        CHECK_EQ(count->ctx, row->ctx);
        const dim3 block(256);
        const dim3 grid((row.NumElements() + block.x - 1) / block.x);
        auto stream = runtime::getCurrentCUDAStream();
        CUDA_KERNEL_CALL(impl::Increment, grid, block, 0, stream, count.Ptr<CounterType>(), count.NumElements(), row.Ptr<IndexType>(), row.NumElements());
        auto device = runtime::DeviceAPI::Get(count->ctx);
        device->StreamSync(count->ctx, stream);
    }; 

    template void Mask<kDGLCUDA, int32_t>(NDArray&, NDArray);
    template void Mask<kDGLCUDA, int64_t>(NDArray&, NDArray);
    template NDArray Flagged<kDGLCUDA, int32_t>(const NDArray&, DGLContext);
    template NDArray Flagged<kDGLCUDA, int64_t>(const NDArray&, DGLContext);
    template void Increment<kDGLCUDA, int32_t>(NDArray&, const NDArray&);
    template void Increment<kDGLCUDA, int64_t>(NDArray&, const NDArray&);
    template void Increment<kDGLCUDA, int32_t, int32_t>(NDArray&, const NDArray&);
    template void Increment<kDGLCUDA, int64_t, int32_t>(NDArray&, const NDArray&);
    template void Increment<kDGLCUDA, int32_t, int64_t>(NDArray&, const NDArray&);
    template void Increment<kDGLCUDA, int64_t, int64_t>(NDArray&, const NDArray&);
}