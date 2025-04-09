# cubin_caller

This is a simple example of how to call a cubin file from python and julia.

In python, we use Pytorch to allocate memory on the device and cuda.core for kernel launch.

In julia, we use CUDA.jl instead.

## compile the cubin file

use nvcc to compile the cubin file:

```
nvcc -arch=sm_90a -cubin -o kernel.cubin saxpy.cu
```

or use triton to dump the tuned cubin file:

```
python triton_kernel.py
```

## prepare the launch parameters

In python, we use numpy to present float32, float64, uint64, int64, etc.

In julia, we need to specify the type when launch the kernel.

For triton kernel, we need to note that:
1. the number of parameters may differ between generated cubin and triton kernel. 
2. Triton would use too many shared memory, we may need to adjust the shared memory limitation via driver api.