# Makefile for matrix transpose

.PHONY: clean test_matrix_transpose_v1 profile_matrix_transpose_v1 test_matrix_transpose_v2 profile_matrix_transpose_v2 test_matrix_transpose_v3 profile_matrix_transpose_v3

matrix_transpose_v1: matrix_transpose_v1.cu
	nvcc -Xcompiler -fopenmp -O3 -o matrix_transpose_v1 matrix_transpose_v1.cu

test_matrix_transpose_v1: matrix_transpose_v1
	./matrix_transpose_v1

profile_matrix_transpose_v1: matrix_transpose_v1
	ncu ./matrix_transpose_v1 | tee matrix_transpose_v1_profile.txt


matrix_transpose_v2: matrix_transpose_v2.cu
	nvcc -Xcompiler -fopenmp -O3 -o matrix_transpose_v2 matrix_transpose_v2.cu

test_matrix_transpose_v2: matrix_transpose_v2
	./matrix_transpose_v2

profile_matrix_transpose_v2: matrix_transpose_v2
	ncu ./matrix_transpose_v2 | tee matrix_transpose_v2_profile.txt


matrix_transpose_v3: matrix_transpose_v3.cu
	nvcc -Xcompiler -fopenmp -O3 -o matrix_transpose_v3 matrix_transpose_v3.cu

test_matrix_transpose_v3: matrix_transpose_v3
	./matrix_transpose_v3

profile_matrix_transpose_v3: matrix_transpose_v3
	ncu ./matrix_transpose_v3 | tee matrix_transpose_v3_profile.txt


clean:
	@rm -f matrix_transpose_v1 matrix_transpose_v1_profile.txt matrix_transpose_v2 matrix_transpose_v2_profile.txt matrix_transpose_v3 matrix_transpose_v3_profile.txt