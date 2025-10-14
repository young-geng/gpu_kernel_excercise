# Makefile for matrix reduce

.PHONY: clean test_matmul_v1 profile_matmul_v1 test_matmul_v2 profile_matmul_v2

matmul_v1: matmul_v1.cu
	nvcc -lopenblas -Xcompiler -fopenmp -O3 -o matmul_v1 matmul_v1.cu

test_matmul_v1: matmul_v1
	./matmul_v1

profile_matmul_v1: matmul_v1
	ncu ./matmul_v1 | tee matmul_v1_profile.txt


matmul_v2: matmul_v2.cu
	nvcc -lopenblas -Xcompiler -fopenmp -O3 -o matmul_v2 matmul_v2.cu

test_matmul_v2: matmul_v2
	./matmul_v2

profile_matmul_v2: matmul_v2
	ncu ./matmul_v2 | tee matmul_v2_profile.txt


clean:
	@rm -f matmul_v1 matmul_v1_profile.txt matmul_v2 matmul_v2_profile.txt