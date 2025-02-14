kernel.cubin: saxpy.cu
	nvcc -arch=sm_90a -cubin -o kernel.cubin saxpy.cu

clean:
	rm -f kernel.cubin

