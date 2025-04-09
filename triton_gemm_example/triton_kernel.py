import triton
import triton.language as tl
import torch
import json


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def matmul_get_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K" : BK, "GROUP_SIZE_M" : 8}, num_stages=s, num_warps=w) \
        for BM in [128] \
        for BN in [128] \
        for BK in [128] \
        for s in [3] \
        for w in [4] \
    ]


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel(a_ptr, b_ptr, c_ptr,  #
                  M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  BLOCK_SIZE_M: tl.constexpr,  #
                  BLOCK_SIZE_N: tl.constexpr,  #
                  BLOCK_SIZE_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if (c_ptr.dtype.element_ty == tl.float8e4nv):
        c = accumulator.to(tl.float8e4nv)
    else:
        c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    kernel = matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    print(f"{M=}, {N=}, {K=}, {a.stride(0)=}, {a.stride(1)=}, {b.stride(0)=}, {b.stride(1)=}, {c.stride(0)=}, {c.stride(1)=}")
    return c, kernel


def main(M, N, K, dtype):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b = b.T.contiguous()

    out, kernel = matmul(a, b.T)
    # Get the PTX and CUBIN from the kernel
    ptx = kernel.asm["ptx"]
    cubin = kernel.asm["cubin"]
    
    # Save the CUBIN file
    with open("triton_kernel.cubin", "wb") as f:
        f.write(cubin)
    
    save_target = ["ptx"] #,"llir","ttir","ttgir"]
    for target in save_target:
        with open(f"triton_kernel.{target}", "w") as f:
            f.write(kernel.asm[target])
       
    kernel_config = matmul_kernel.best_config.all_kwargs()
    kernel_config["shmem_size"] = kernel.metadata.shared
    json.dump(kernel_config, open("kernel_config.json", "w"))
    print(f"[INFO] {kernel_config=}")

    torch.testing.assert_close(out, torch.matmul(a, b.T))
    print("done!")


if __name__ == "__main__":
    main(1024, 4096, 4096, torch.float16)
