# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import numpy as np
import torch

from cuda.core.experimental import Device, LaunchConfig, launch
from cuda.core.experimental._module import ObjectCode


def module_cubin(cubin_path):
    data = open(cubin_path, "rb").read()
    return ObjectCode(data, "cubin")

def main():
    device = Device()
    device.set_current()
    s = device.create_stream()

    module = module_cubin("triton_kernel.cubin")

    kernel = module.get_kernel("saxpy_kernel")

    # prepare input/output
    N = np.uint32(1024)

    a = np.float32(2.0)
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    y = torch.randn(N, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x, dtype=torch.float32, device="cuda")

    # prepare launch
    block = 128
    grid = int((N + block - 1) // block)
    config = LaunchConfig(grid=grid, block=block, stream=s)
    ker_args = (a, x.data_ptr(), y.data_ptr(), out.data_ptr(), N)

    # launch kernel on stream s
    launch(kernel, config, *ker_args)
    s.sync()

    # check result
    torch.testing.assert_close(out, a * x + y)

    # clean up 
    s.close()

    print("done!")

if __name__ == "__main__":
    main()