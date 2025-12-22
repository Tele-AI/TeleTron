#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <rmsnorm_kernel.h>
#include <rmsnorm_kernel.cuh>



__global__ void RMSNormImpl_bp(
    __nv_bfloat16* dinp, 
    const __nv_bfloat16* dout, 
    const __nv_bfloat16* x_norm, 
    const __nv_bfloat16* weight, 
    const float* inv_variance,
    const int64_t rows, const int64_t cols
) {
    extern __shared__ float sum_sm[];
    constexpr int pack_size = 8;
    const int tidx = threadIdx.x;
    const int num_packs_per_row = cols / pack_size;
    const int pack_off = num_packs_per_row * pack_size;
    int col_offset=0;

    int64_t row=(blockIdx.y * gridDim.x +blockIdx.x)*blockDim.y+threadIdx.y;

    for (; row < rows; row += gridDim.y * gridDim.x * blockDim.y)
    {
        const __nv_bfloat16 *current_dout_row = dout + row * cols;
        const __nv_bfloat16 *current_x_norm_row = x_norm + row * cols;
        __nv_bfloat16 *current_dinp_row = dinp + row * cols;
        const float inv_mean = inv_variance[row];

        float x2_pack_sum = 0.0f;
        __nv_bfloat162 pack_weight[pack_size/2];
        __nv_bfloat162 pack_dout[pack_size / 2];
        __nv_bfloat162 pack_x_norm[pack_size / 2];

        for (int pack_idx = tidx; pack_idx < num_packs_per_row; pack_idx += blockDim.x)
        {
          col_offset = pack_idx * pack_size;
          *reinterpret_cast<float4*>(pack_dout) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(current_dout_row +col_offset));
          *reinterpret_cast<float4*>(pack_x_norm) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(current_x_norm_row + col_offset));
        

          #pragma unroll
          for(int i = 0; i < pack_size/2; ++i) {
              auto doutWX_i = __bfloat1622float2(__hmul2(pack_dout[i], pack_x_norm[i]));
            
              x2_pack_sum += doutWX_i.x   + doutWX_i.y;
        }
        }

        for (int i = pack_off + tidx; i < cols; i += blockDim.x)
        {
            float dout_val = __bfloat162float(current_dout_row[i]);
            float x_norm_val = __bfloat162float(current_x_norm_row[i]);
            float val = dout_val * x_norm_val;
            x2_pack_sum += val;
        }

        if(blockDim.y==1) x2_pack_sum = blockReduceSum<float>(x2_pack_sum);
        else if(blockDim.y ==8 ){
            blockReduceSumTemp<float,32>(x2_pack_sum,sum_sm);
            x2_pack_sum=sum_sm[threadIdx.y];
        }else{
            blockReduceSumTemp<float,16>(x2_pack_sum,sum_sm);
            x2_pack_sum=sum_sm[threadIdx.y];
        }

        x2_pack_sum =  x2_pack_sum / static_cast<float>(cols);


        for (int pack_idx = tidx; pack_idx < num_packs_per_row; pack_idx += blockDim.x)
        {
          col_offset = pack_idx * pack_size;  
          *reinterpret_cast<float4*>(pack_dout) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(current_dout_row +col_offset));
          *reinterpret_cast<float4*>(pack_x_norm) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(current_x_norm_row + col_offset));
          *reinterpret_cast<float4*>(pack_weight) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(weight + col_offset));  
          #pragma unroll
          for (int i = 0; i < pack_size / 2; ++i) {

              auto d2 = __bfloat1622float2(pack_dout[i]);
              auto w2 = __bfloat1622float2(pack_weight[i]);
              auto x2 = __bfloat1622float2(pack_x_norm[i]);

              auto dw1 = d2.x * w2.x; auto dw2 = d2.y * w2.y;
              auto rsw1 = x2.x / w2.x; auto rsw2 = x2.y / w2.y; 

              pack_dout[i] = __floats2bfloat162_rn(inv_mean *( dw1 - x2_pack_sum * rsw1 ), inv_mean * (dw2 - x2_pack_sum * rsw2));
          }
          *reinterpret_cast<float4*>(current_dinp_row + col_offset) = *reinterpret_cast<float4*>(pack_dout);
        } 
    
        for (int i = pack_off + tidx; i < cols; i += blockDim.x)
        {
            float dout_val = __bfloat162float(current_dout_row[i]);
            float x_norm_val = __bfloat162float(current_x_norm_row[i]);
            float weight_val= __bfloat162float(weight[i]);
            float dw = dout_val * weight_val;
            float rsw= x_norm_val / weight_val;
            current_dinp_row[i] = __float2bfloat16(
                inv_mean * (dw - rsw * x2_pack_sum)
            );
        }
    }
}

void launch_rms_backward(
    at::BFloat16* dinp, 
    at::BFloat16* dweight,
    const at::BFloat16* dout, 
    const at::BFloat16* x_rmsnorm, const at::BFloat16* weight, const float* inv_variance,
    const int64_t rows, const int64_t cols,cudaStream_t stream
) {
    launch_rms_weight_backward(
        (__nv_bfloat16*)dweight,  
        ( const __nv_bfloat16*)dout, ( const __nv_bfloat16*)x_rmsnorm, ( const __nv_bfloat16*)weight,
        rows, cols,stream
    );
    const int64_t MAX_GRID_X = 65535;
    const int blockdim_y = (cols > 512) ? 1 : (cols > 128) ? 8 : 16;
    const int blockdim_x = 
    (cols > 8000) ? 1024 :
    (cols > 2048) ? 512 :
    (cols > 1024) ? 256 :
    (cols > 512)  ? 128 :
    (cols >128)   ? 32 :
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
	RMSNormImpl_bp<<<grid,block,rows_shared_mem_size,s>>>(
		(__nv_bfloat16*)dinp, 
        ( const __nv_bfloat16*)dout, 
        ( const __nv_bfloat16*)x_rmsnorm,  ( const __nv_bfloat16*)weight,  inv_variance,
        rows, cols
	);
    CUDA_CHECK(cudaGetLastError());
}
