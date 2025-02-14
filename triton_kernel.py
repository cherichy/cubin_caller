import triton
import triton.language as tl
import torch

@triton.jit
def saxpy_kernel(
    a,
    x_ptr,
    y_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr = 128,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n
    # Load x and y vectors
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Compute saxpy
    out = a * x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def saxpy(a, x, y, out, n):
    # Launch kernel with appropriate grid
    grid = (triton.cdiv(n, 128),)
    kernel = saxpy_kernel[grid](a, x, y, out, n)
    return kernel


def main():
    a = 2.0
    n = 1024
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    y = torch.randn(n, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x, dtype=torch.float32, device="cuda")

    kernel = saxpy(a, x, y, out, n)
    # Get the PTX and CUBIN from the kernel
    ptx = kernel.asm["ptx"]
    cubin = kernel.asm["cubin"]
    
    # Save the CUBIN file
    with open("triton_kernel.cubin", "wb") as f:
        f.write(cubin)
        
    # Save the PTX file
    with open("triton_kernel.ptx", "w") as f:
        f.write(ptx)

    torch.testing.assert_close(out, a * x + y)
    print("done!")


if __name__ == "__main__":
    main()
