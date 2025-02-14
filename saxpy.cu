extern "C" __global__ void saxpy_kernel(float a, float *x, float *y, float *out,
                                        unsigned int n) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}