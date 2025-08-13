# Makefile for vector add

.PHONY: clean test_vector_add profile_vector_add

vector_add: vector_add.cu
	nvcc -O3 -o vector_add vector_add.cu

test_vector_add: vector_add
	./vector_add

profile_vector_add: vector_add
	ncu ./vector_add | tee vector_add_profile.txt


clean:
	@rm -f vector_add vector_add_profile.txt