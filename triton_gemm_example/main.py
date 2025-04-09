# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import numpy as np
import torch
import json
import os

from cuda.core.experimental import Device, LaunchConfig, launch
from cuda.core.experimental._module import ObjectCode
from cuda.core.experimental._utils import driver, handle_return

def module_cubin(cubin_path):
    data = open(cubin_path, "rb").read()
    return ObjectCode(data, "cubin")


def matmul(a, b):
    device = Device()
    device.set_current()
    s = device.create_stream()

    module = module_cubin("triton_kernel.cubin")

    kernel = module.get_kernel("matmul_kernel")

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # prepare launch
    kernel_config = json.load(open("kernel_config.json"))
    block = kernel_config["num_warps"]*32 if os.path.exists("kernel_config.json") else 256
    block_m = kernel_config["BLOCK_SIZE_M"]
    block_n = kernel_config["BLOCK_SIZE_N"]
    grid = int((N + block_n - 1) // block_n) * int((M + block_m - 1) // block_m)
    print(f"[INFO] {grid=}, {block=}")

    smem_sz = kernel_config["shmem_size"]

    # Triton kernel would use too much shmem, need to raise the shmem limit use driver API.
    config = LaunchConfig(grid=grid, block=block, stream=s, shmem_size=smem_sz)
    handle_return(driver.cuKernelSetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_sz, int(kernel._handle), device._id ))
    cures = handle_return(driver.cuKernelGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(kernel._handle), device._id))
    print(f"[INFO] Shmem limit set to {cures/1024} KB")

    # need to set the correct stride align with the PTX signature.
    ker_args = (a.data_ptr(), b.data_ptr(), c.data_ptr(), np.uint32(M), np.uint32(N), np.uint32(K), np.uint32(a.stride(0)), np.uint32(b.stride(1)), np.uint32(c.stride(0)))

    print(f"[INFO] {ker_args=}")
    # print(f"{M=}, {N=}, {K=}, {a.stride(0)=}, {a.stride(1)=}, {b.stride(0)=}, {b.stride(1)=}, {c.stride(0)=}, {c.stride(1)=}")

    # launch kernel on stream s
    launch(kernel, config, *ker_args)
    s.sync()

    # clean up 
    s.close()

    return c

def main(M, N, K, dtype):
    # prepare input/output
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()
    c = b.T

    out = matmul(a, c)

    # check result
    torch.testing.assert_close(out, torch.matmul(a, b.T))

    print("done!")

if __name__ == "__main__":
    main(1024, 4096, 4096, torch.float16)
