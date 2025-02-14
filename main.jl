using CUDA
using Test

# Load the cubin file
cubin_data = read("kernel.cubin")
mod = CuModule(cubin_data)
kernel = CuFunction(mod, "saxpy_kernel")

# Prepare input data
N = UInt32(1024)
a = Float32(2.0)
x = CUDA.rand(Float32, N)
y = CUDA.rand(Float32, N)
out = CUDA.similar(x)

# Configure launch parameters
threads = 128
blocks = ceil(Int, N/threads)

# Launch kernel with 
CUDA.@sync CUDA.cudacall(
    kernel,
    (Float32, CuPtr{Float32}, CuPtr{Float32}, CuPtr{Float32}, UInt32),
    a, x, y, out, N;
    blocks = blocks,
    threads = threads
)

# Verify results
@test out ≈ a .* x .+ y

