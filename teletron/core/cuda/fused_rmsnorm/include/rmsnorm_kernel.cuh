#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include<c10/cuda/CUDAStream.h>

#define CUDA_CHECK(expr)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (expr);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)


typedef struct __align__(8) {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
    __nv_bfloat16 z;
    __nv_bfloat16 w;
} __nv_bfloat164;

// float4 -> __nv_bfloat164
__device__ __forceinline__ __nv_bfloat164 __float42bfloat164(const float4 &f) {
    __nv_bfloat164 bf;
    bf.x = __float2bfloat16(f.x);
    bf.y = __float2bfloat16(f.y);
    bf.z = __float2bfloat16(f.z);
    bf.w = __float2bfloat16(f.w);
    return bf;
}

// __nv_bfloat164 -> float4
__device__ __forceinline__ float4 __bfloat1642float4(const __nv_bfloat164 &bf) {
    float4 f;
    f.x = __bfloat162float(bf.x);
    f.y = __bfloat162float(bf.y);
    f.z = __bfloat162float(bf.z);
    f.w = __bfloat162float(bf.w);
    return f;
}

constexpr int WARP_SIZE = 32;

template <typename T>
__forceinline__ __device__  T warpReduceSum(T val)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T,int X>
__forceinline__ __device__  T warpReduceSumTemp(T val)
{
    for (int offset = X /2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__forceinline__ __device__ T blockReduceSum(T val)
{
    const int tidx = threadIdx.x;
    const int lane_id = tidx & (warpSize-1);
    const int warp_id = tidx / warpSize;
    T warp_sum = warpReduceSum<T>(val);

    __shared__ T shared_warp_sum[WARP_SIZE];

    if (lane_id == 0)
        shared_warp_sum[warp_id] = warp_sum;

    __syncthreads();
    
    T block_sum=0;

    if (warp_id == 0)
    {
        int num_warps=(blockDim.x+warpSize-1)/warpSize;
        block_sum=(lane_id<num_warps)? shared_warp_sum[lane_id]: static_cast<T>(0);
        block_sum=warpReduceSum<T>(block_sum);
        if(lane_id==0) shared_warp_sum[0]=block_sum;  
    }
    __syncthreads();
    return shared_warp_sum[0];
}

template<typename T,int X>
__forceinline__ __device__ void blockReduceSumTemp(T val, T* dout){
    int tidx = threadIdx.x;
    val = warpReduceSumTemp<T,X>(val);

    if(tidx == 0) dout[threadIdx.y] = val;
    __syncthreads();
}

template <typename T>
__forceinline__ __device__ T blockReduceSumThready(T val)
{
    const int tidx = threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id = tidx & (warpSize-1);
    const int warp_id = tid / warpSize;
    const int warp_num_per_row = blockDim.x / warpSize;
    const int warp_id_start = warp_id / warp_num_per_row * warp_num_per_row;

    T warp_sum = warpReduceSum<T>(val);

    __shared__ T shared_warp_sum[WARP_SIZE];

    if (lane_id == 0)
        shared_warp_sum[warp_id] = warp_sum;

    __syncthreads();
    
    T block_sum=0;

    if (warp_id % warp_num_per_row ==0)
    {
        int num_warps=(blockDim.x+warpSize-1)/warpSize;
        block_sum=(lane_id<num_warps)? shared_warp_sum[warp_id + lane_id]: static_cast<T>(0);
        block_sum=warpReduceSum<T>(block_sum);
        if(lane_id==0) shared_warp_sum[warp_id]=block_sum;  
    }
    __syncthreads();
    return shared_warp_sum[warp_id_start];
}
