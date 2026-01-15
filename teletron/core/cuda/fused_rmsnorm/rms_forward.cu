#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include<cuda_runtime.h>
#include<c10/cuda/CUDAStream.h>
#include<cub/block/block_reduce.cuh>
#include <rmsnorm_kernel.h>
#include <rmsnorm_kernel.cuh>
//要求cuda 12.0及以上



__global__ void RMSNormImpl(__nv_bfloat16 *output,
                            const __nv_bfloat16 *input,
                            const __nv_bfloat16 *weight,
                            const int64_t rows,
                            const int64_t cols,
                            const double epsilon,
                            float *inv_variance)
{
    extern __shared__ float sum_sm[];
    constexpr int pack_size = 8;
    const int tidx = threadIdx.x;
    const int num_packs_per_row = cols / pack_size;
    const int pack_off = num_packs_per_row * pack_size;

    int64_t row=(blockIdx.y * gridDim.x +blockIdx.x) *blockDim.y + threadIdx.y;
    if(row>=rows) return;

    for (; row < rows; row += gridDim.y * gridDim.x*blockDim.y)
    {
        const __nv_bfloat16 *current_input_row = input + row * cols;
        __nv_bfloat16 *current_output_row = output + row * cols;

        float x2_pack_sum = 0.0f;

        float val=0.0f;
        int pack_idx=0;
        __nv_bfloat162 pack_data[pack_size / 2];
        __nv_bfloat162 pack_data1[pack_size / 2];
        for (pack_idx = tidx; pack_idx < num_packs_per_row; pack_idx += blockDim.x)
        {
            const __nv_bfloat16* input_pack_ptr = current_input_row + pack_idx * pack_size;
            const __nv_bfloat162* input_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(input_pack_ptr);


           *reinterpret_cast<float4*>(pack_data1) = *reinterpret_cast<float4*>((const_cast<__nv_bfloat162*>(input_bf162_ptr)));
           #pragma unroll
           for (int i = 0; i < pack_size / 2; i++){
                
                auto tmp_f2 = __bfloat1622float2(pack_data1[i]);
                x2_pack_sum += tmp_f2.x * tmp_f2.x + tmp_f2.y * tmp_f2.y;
           }
        }

        for (pack_idx = pack_off + tidx; pack_idx < cols; pack_idx += blockDim.x)
        {
            val = __bfloat162float(current_input_row[pack_idx]);
            x2_pack_sum += val * val;
        }


        if(blockDim.y==1) x2_pack_sum = blockReduceSum<float>(x2_pack_sum);
        else if(blockDim.y ==8 ){
            blockReduceSumTemp<float,32>(x2_pack_sum,sum_sm);
            x2_pack_sum=sum_sm[threadIdx.y];
        }else{
            blockReduceSumTemp<float,16>(x2_pack_sum,sum_sm);
            x2_pack_sum=sum_sm[threadIdx.y];
        }


        float inv_mean = rsqrtf(max(x2_pack_sum / static_cast<float>(cols),0.0f) + epsilon);
        if (tidx == 0)
        {
            inv_variance[row] = inv_mean;
        }

        for (pack_idx = tidx; pack_idx < num_packs_per_row; pack_idx += blockDim.x)
        {
            const int col_offset = pack_idx* pack_size;
            __nv_bfloat16* output_pack_ptr = current_output_row + col_offset;
            const __nv_bfloat16* input_pack_ptr = current_input_row + col_offset;
            const __nv_bfloat162* input_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(input_pack_ptr);
           *reinterpret_cast<float4*>(pack_data1) = *reinterpret_cast<float4*>((const_cast<__nv_bfloat162*>(input_bf162_ptr)));

            const __nv_bfloat16* x_weight_pack_ptr = weight + col_offset;
            *reinterpret_cast<float4*>(pack_data) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(x_weight_pack_ptr));
            #pragma unroll
            for (int i = 0; i < pack_size>>1; ++i) {
                
                auto val_bf2 =  pack_data1[i];
                auto val_f2=__bfloat1622float2(val_bf2);
                val_f2.x  = val_f2.x * inv_mean;
                val_f2.y  = val_f2.y * inv_mean;
                pack_data[i] = __hmul2(__float22bfloat162_rn(val_f2), pack_data[i]);
            }
            *reinterpret_cast<float4*>(output_pack_ptr) = *reinterpret_cast<float4*>(pack_data);
        }
        for (pack_idx = pack_off + tidx; pack_idx < cols; pack_idx += blockDim.x)
        {
            val = __bfloat162float(current_input_row[pack_idx]);
            float weight_val= __bfloat162float(weight[pack_idx]);
            current_output_row[pack_idx] = __float2bfloat16(val * weight_val * inv_mean);
        }
    }
}

void launch_rms_forward(
    at::BFloat16 *output,
    const at::BFloat16 *input,
    const at::BFloat16 *weight,
    const int64_t rows,
    const int64_t cols,
    const double epsilon,
    float *inv_variance, cudaStream_t stream)
{

    // auto row_grid = (rows>>4 <12000 )? rows>>4:rows>>6;
    const int64_t MAX_GRID_X = 65535;
    const int blockdim_y = (cols > 512) ? 1 : (cols > 128) ? 8 : 16;
    const int blockdim_x = 
    (cols > 8000) ? 1024 :
    (cols > 2048) ? 512 :
    (cols > 1024) ? 256 :
    (cols > 512)  ? 128 :
    (cols > 128)  ? 32 :
                    16 ;
    int64_t new_rows = rows / blockdim_y;
    auto row_grid = (new_rows <12000 )? new_rows:new_rows>>2;
    int64_t grid_x = std::min(row_grid  , MAX_GRID_X);
    int64_t grid_y = (row_grid + MAX_GRID_X - 1) / MAX_GRID_X;
    grid_y = std::min(grid_y, MAX_GRID_X);

    dim3 grid(grid_x, grid_y);                           // 一个block转为处理一行数据
    dim3 block(blockdim_x,blockdim_y);
 
    auto sh_mem_size = (blockdim_y > 1)? blockdim_y:0;
    size_t rows_shared_mem_size = sh_mem_size * sizeof(float);
    cudaStream_t s = stream ? stream : c10::cuda::getCurrentCUDAStream();
    RMSNormImpl<<<grid, block, rows_shared_mem_size, s>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        rows, cols, static_cast<float>(epsilon),
        inv_variance);
    CUDA_CHECK(cudaGetLastError());
}