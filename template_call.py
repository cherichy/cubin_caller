# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import sys

import numpy as np
import torch
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# compute out = a * x + y
code = """
template<typename T>
__global__ void saxpy(const T a,
                      const T* x,
                      const T* y,
                      T* out,
                      size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        out[tid] = a * x[tid] + y[tid];
    }
}
"""

dev = Device()
dev.set_current()
s = dev.create_stream()

# prepare program
arch = "".join(f"{i}" for i in dev.compute_capability)
program_options = ProgramOptions(std="c++11", arch=f"sm_{arch}")
prog = Program(code, code_type="c++", options=program_options)


# run in single precision
dtypes = [np.float32, np.float64]
torch_dtypes = [torch.float32, torch.float64]
kernels = ["saxpy<float>", "saxpy<double>"]

mod = prog.compile(
    "cubin",
    logs=sys.stdout,
    name_expressions=kernels,
)

for dtype, torch_dtype, kernel in zip(dtypes, torch_dtypes, kernels):
    ker = mod.get_kernel(kernel)
    # prepare input/output
    size = np.uint64(1024)
    a = dtype(10)
    x = torch.randn(size, dtype=torch_dtype, device="cuda")
    y = torch.randn(size, dtype=torch_dtype, device="cuda")
    out = torch.empty_like(x, dtype=torch_dtype, device="cuda")
    dev.sync()  # cupy runs on a different stream from s, so sync before accessing

    # prepare launch
    block = 128
    grid = int((size + block - 1) // block)
    config = LaunchConfig(grid=grid, block=block)
    ker_args = (a, x.data_ptr(), y.data_ptr(), out.data_ptr(), size)

    # launch kernel on stream s
    launch(s, config, ker, *ker_args)
    s.sync()

    # check result
    torch.testing.assert_close(out, a * x + y)
    print(f"kernel {kernel} passed.")

s.close()
