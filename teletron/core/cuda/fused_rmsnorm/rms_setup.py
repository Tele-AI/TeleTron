from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_rmsnorm",  
    include_dirs=["include"],     
    ext_modules=[
        CUDAExtension(
            "custom_rmsnorm",  
            sources=[
                "rms_ops.cpp",  # 
                "rms_forward.cu",  
                "rms_backward.cu",
                "weight_backward.cu"
            ],
            extra_compile_args={"cxx": ["-O3"], 
                                'nvcc': [
                    '-O3', 
                    '-arch=sm_90',       
                    '-DENABLE_BF16',     
                    '--use_fast_math',    
                    '-gencode=arch=compute_90,code=sm_90'
                ]
            }  
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)