#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include<c10/cuda/CUDAStream.h>

void launch_rms_forward(
	at::BFloat16 *  output,     
    const at::BFloat16 *  input, 
    const at::BFloat16 *  weight, 
    const int64_t rows,         
    const int64_t cols,         
    const double epsilon,       
    float* inv_variance,cudaStream_t stream=nullptr);

void launch_rms_weight_backward(
    __nv_bfloat16* dweight,
    const __nv_bfloat16* dout, const __nv_bfloat16* x_rmsnorm, const __nv_bfloat16* weight,
    const int64_t rows, const int64_t cols,cudaStream_t stream=nullptr);

void launch_rms_backward(
    at::BFloat16* dinp, 
    at::BFloat16* dweight,
    const at::BFloat16* dout, 
    const at::BFloat16* x_rmsnorm, const at::BFloat16* weight, const float* inv_variance,
    const int64_t rows, const int64_t cols,cudaStream_t stream=nullptr
);