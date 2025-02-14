kernel.cubin: saxpy.cu
	nvcc -arch=sm_90a -o kernel.cubin saxpy.cu

clean:
	rm -f kernel.cubin

