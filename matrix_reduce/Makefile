# Makefile for matrix reduce

.PHONY: clean test_matrix_reduce profile_matrix_reduce

matrix_reduce: matrix_reduce.cu
	nvcc -O3 -o matrix_reduce matrix_reduce.cu

test_matrix_reduce: matrix_reduce
	./matrix_reduce

profile_matrix_reduce: matrix_reduce
	ncu ./matrix_reduce | tee matrix_reduce_profile.txt


clean:
	@rm -f matrix_reduce matrix_reduce_profile.txt