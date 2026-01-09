#include <torch/extension.h>
#include "rmsnorm_kernel.h"
#include <stdio.h>


void torch_launch_rms_forward(
                                        torch::Tensor &output, 
                                       const torch::Tensor &input,
                                       const torch::Tensor &weight,
                                       int64_t rows,
                                       int64_t cols,
                                       double epsilon,
                                       torch::Tensor &inv_variance) {
    launch_rms_forward(
        (at::BFloat16 *)(output.data_ptr()),
        (const at::BFloat16 *)(input.data_ptr()),
        (const at::BFloat16 *)(weight.data_ptr()),
        rows,
        cols,
        epsilon,
        (float *)(inv_variance.data_ptr())
    );
}

void torch_launch_rms_backward(
                                        torch::Tensor &dinp,
                                        torch::Tensor &dweight,
                                        const torch::Tensor &dout,
                                        const torch::Tensor &x_norm,
                                        const torch::Tensor &weight,
                                        const torch::Tensor &inv_variance,
                                        int64_t rows,
                                        int64_t cols) {
    launch_rms_backward( 
        (at::BFloat16 *)(dinp.data_ptr()),
        (at::BFloat16 *)(dweight.data_ptr()),
        (const at::BFloat16 *)(dout.data_ptr()),
        (const at::BFloat16 *)(x_norm.data_ptr()),
        (const at::BFloat16 *)(weight.data_ptr()),
        static_cast<const float*>(inv_variance.data_ptr()),
        rows,
        cols 
    );
}


PYBIND11_MODULE(custom_rmsnorm, m) {
    m.def("torch_launch_rms_forward",
          &torch_launch_rms_forward,
          "RMSNorm forward kernel");
    
    m.def("torch_launch_rms_backward",
          &torch_launch_rms_backward,
          "RMSNorm backward kernel");
}


TORCH_LIBRARY(Custom_RMSNorm, m) {
    m.def("torch_launch_rms_forward", torch_launch_rms_forward);
    m.def("torch_launch_rms_backward", torch_launch_rms_backward);
}