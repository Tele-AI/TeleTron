#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <rmsnorm_kernel.h>
#include <rmsnorm_kernel.cuh>

template<const int num_rows_per_block=32>
__global__ void RMSNorm_weight_bp_opt(
    __nv_bfloat16* dweight,
    const __nv_bfloat16* dout, const __nv_bfloat16* x_rmsnorm, const __nv_bfloat16* weight,
    const int64_t rows, const int64_t cols, float* dweight_float
) {
    constexpr int pack_size = 4;
    const int tidx = threadIdx.x;
    const int num_packs_per_row = cols / pack_size;
    const int pack_off = num_packs_per_row * pack_size;

    int64_t row_start= (blockIdx.y * gridDim.x +blockIdx.x)*num_rows_per_block;
    int64_t row_end=min(row_start+num_rows_per_block,rows);

    extern __shared__ float shared_mem[]; 
    float* shared_dweight = shared_mem; // size: cols
    

    for (int i = tidx; i < cols; i += blockDim.x) {
        shared_dweight[i] = 0.0f;
    }
    __syncthreads();

    for (int64_t row = row_start; row < row_end; ++row)
    {
        const __nv_bfloat16 *current_dout_row = dout + row * cols;
        const __nv_bfloat16 *current_x_norm_row = x_rmsnorm + row * cols;
        const __nv_bfloat16 *current_weight_row = weight;

        const __nv_bfloat164 *dout_pack4_ptr = reinterpret_cast<const __nv_bfloat164 *>(current_dout_row);
        const __nv_bfloat164 *x_norm_pack4_ptr = reinterpret_cast<const __nv_bfloat164 *>(current_x_norm_row);
        const __nv_bfloat164 *weight_pack4_ptr = reinterpret_cast<const __nv_bfloat164 *>(current_weight_row);
        for (int pack_idx = tidx; pack_idx < num_packs_per_row; pack_idx += blockDim.x)
        {
            __nv_bfloat164 dout_bfloat164 = *(dout_pack4_ptr + pack_idx);
            __nv_bfloat164 x_norm_bfloat164 = *(x_norm_pack4_ptr + pack_idx);
            __nv_bfloat164 weight_bfloat164 = *(weight_pack4_ptr + pack_idx);
            float4 dout_float4=__bfloat1642float4(dout_bfloat164);
            float4 x_norm_float4=__bfloat1642float4(x_norm_bfloat164);
            float4 weight_float4=__bfloat1642float4(weight_bfloat164);
            shared_dweight[pack_idx * pack_size + 0] += dout_float4.x * x_norm_float4.x/weight_float4.x;
            shared_dweight[pack_idx * pack_size + 1] += dout_float4.y * x_norm_float4.y/weight_float4.y;
            shared_dweight[pack_idx * pack_size + 2] += dout_float4.z * x_norm_float4.z/weight_float4.z;
            shared_dweight[pack_idx * pack_size + 3] += dout_float4.w * x_norm_float4.w/weight_float4.w;
        }

        for (int i = pack_off + tidx; i < cols; i += blockDim.x)
        {
            float dout_val = __bfloat162float(current_dout_row[i]);
            float x_norm_val = __bfloat162float(current_x_norm_row[i]);
            float weight_val = __bfloat162float(current_weight_row[i]);
            float val = dout_val * x_norm_val / weight_val;
            shared_dweight[i] += val;
        }
    }
    __syncthreads();
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    float* dweight_float_ptr = dweight_float + block_id * cols;
    for (int i = tidx; i < cols; i += blockDim.x) {
        dweight_float_ptr[i] = shared_dweight[i];
    }
}

__global__ void RMSNorm_weight_reduce(
    __nv_bfloat16* dweight,
    const float* dweight_float,
    const int64_t cols,const int64_t num_rows
) {
    int col_idx = blockIdx.x;
    int tidx = threadIdx.x;
    if (col_idx >= cols) return;
    float sum = 0.0f;
    for (int row_id = tidx; row_id < num_rows; row_id += blockDim.x) {
        sum += dweight_float[row_id * cols + col_idx];
    }
    sum=blockReduceSum<float>(sum);
    if (tidx == 0) {
        dweight[col_idx] = __float2bfloat16(sum);  
    }
}

//待添加： grid_sync方法 减少kernel调用次数
void launch_rms_weight_backward(
    __nv_bfloat16* dweight,
    const __nv_bfloat16* dout, const __nv_bfloat16* x_rmsnorm, const __nv_bfloat16* weight,
    const int64_t rows, const int64_t cols,cudaStream_t stream)
{
    cudaStream_t s = stream ? stream : c10::cuda::getCurrentCUDAStream();

    int num_rows_per_block = (rows > 384000)? 256:(rows > 192000)? 128:(rows >96000)? 64 : (rows > 48000) ? 32 : (rows > 24000)? 16: 8; 
    int new_rows=(rows+num_rows_per_block-1)/num_rows_per_block;
     auto dweight_tmp = at::empty({new_rows,cols}, 
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    const int MAX_GRID_X = 65535;
    dim3 block = 
    (cols > 8000) ? dim3(1024,1) :
    (cols > 2048) ? dim3(512,1) :
    (cols > 1024) ? dim3(256,1) :
    (cols > 512)  ? dim3(64,1) :
                    dim3(32,1);
    int grid_x = std::min(new_rows,MAX_GRID_X);
    int grid_y = (new_rows + MAX_GRID_X - 1) / MAX_GRID_X;
    grid_y = std::min(grid_y, MAX_GRID_X);
    dim3 grid(grid_x,grid_y);
    size_t cols_shared_mem_size = cols * sizeof(float);
    if(rows>384000) {
    RMSNorm_weight_bp_opt<256><<<grid, block,cols_shared_mem_size,s>>>(
        dweight, dout, x_rmsnorm, weight, rows, cols,dweight_tmp.data_ptr<float>()
    );
    }else if(rows > 192000){
    RMSNorm_weight_bp_opt<128><<<grid, block,cols_shared_mem_size,s>>>(
        dweight, dout, x_rmsnorm, weight, rows, cols,dweight_tmp.data_ptr<float>()
    );
    }else if(rows > 96000){
    RMSNorm_weight_bp_opt<64><<<grid, block,cols_shared_mem_size,s>>>(
        dweight, dout, x_rmsnorm, weight, rows, cols,dweight_tmp.data_ptr<float>()
    );
    }else if(rows > 48000){
    RMSNorm_weight_bp_opt<32><<<grid, block,cols_shared_mem_size,s>>>(
        dweight, dout, x_rmsnorm, weight, rows, cols,dweight_tmp.data_ptr<float>()
    );
    }else if(rows > 24000){
    RMSNorm_weight_bp_opt<16><<<grid, block,cols_shared_mem_size,s>>>(
        dweight, dout, x_rmsnorm, weight, rows, cols,dweight_tmp.data_ptr<float>()
    );        
    }else{
    RMSNorm_weight_bp_opt<8><<<grid, block,cols_shared_mem_size,s>>>(
        dweight, dout, x_rmsnorm, weight, rows, cols,dweight_tmp.data_ptr<float>()
    );          
    }
    CUDA_CHECK(cudaGetLastError());

    RMSNorm_weight_reduce<<<cols,block,0,s>>>(
        dweight, dweight_tmp.data_ptr<float>(), cols,new_rows
    );
    CUDA_CHECK(cudaGetLastError());
}
