# cubin_caller

This is a simple example of how to call a cubin file from python and julia.

In python, we use Pytorch to allocate memory on the device and cuda.core for kernel launch.

In julia, we use CUDA.jl instead.

## compile the cubin file

```
nvcc -arch=sm_90a -o kernel.cubin saxpy.cu
```

## prepare the launch parameters

In python, we use numpy to present float32, float64, uint64, int64, etc.

In julia, we need to specify the type when launch the kernel.

