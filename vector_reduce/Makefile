# Makefile for vector reduce

.PHONY: clean test_vector_reduce profile_vector_reduce

vector_reduce: vector_reduce.cu
	nvcc -O3 -o vector_reduce vector_reduce.cu

test_vector_reduce: vector_reduce
	./vector_reduce

profile_vector_reduce: vector_reduce
	ncu ./vector_reduce | tee vector_reduce_profile.txt


clean:
	@rm -f vector_reduce vector_reduce_profile.txt