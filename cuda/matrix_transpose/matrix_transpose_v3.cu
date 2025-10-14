#include <stdio.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <random>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define ceil_div(a, b) ((a + b - 1) / b)


const int DSIZE = 32 * 1024; // 1GB of data
const int BLOCK_SIZE = 32;
const int GRID_SIZE = 32;


__global__ void matrix_transpose(const float *input, float *output, int ds) {
  __shared__ float block[BLOCK_SIZE * (BLOCK_SIZE + 1)];
  for (int bx = blockIdx.x * blockDim.x; bx + threadIdx.x < ds; bx += gridDim.x * blockDim.x) {
    for (int by = blockIdx.y * blockDim.y; by + threadIdx.y < ds; by += gridDim.y * blockDim.y) {
      block[threadIdx.y * (BLOCK_SIZE + 1) + threadIdx.x] = input[(by + threadIdx.y) * ds + bx + threadIdx.x];
      __syncthreads();
      output[(bx + threadIdx.y) * ds + by + threadIdx.x] = block[threadIdx.x * (BLOCK_SIZE + 1) + threadIdx.y];
    }
  }

}


int main(){
  float *host_matrix = new float[DSIZE * DSIZE];
  float *host_output = new float[DSIZE * DSIZE];
  float *answer = new float[DSIZE * DSIZE];
  float *device_matrix, *device_output;
  printf("Initializing data.\n");
  #pragma omp parallel
  {
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num()); // Seed with thread ID
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    #pragma omp for collapse(2)
    for (int i = 0; i < DSIZE; i++) {
      for (int j = 0; j < DSIZE; j++) {
        host_matrix[i * DSIZE + j] = dis(gen);
        answer[j * DSIZE + i] = host_matrix[i * DSIZE + j];
      }
    }
  }
  printf("Allocating and transferring memory.\n");
  cudaMalloc(&device_matrix, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&device_output, DSIZE*DSIZE*sizeof(float));

  cudaCheckErrors("cudaMalloc failure"); // error checking
  cudaMemcpy(device_matrix, host_matrix, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  printf("Launching kernel.\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 grid(GRID_SIZE, GRID_SIZE, 1);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

  cudaEventRecord(start);
  matrix_transpose<<<grid, block>>>(device_matrix, device_output, DSIZE);

  cudaEventRecord(stop);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(host_output, device_output, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("Checking answers......");
  for (int i = 0; i < DSIZE; i++){
    for(int j = 0; j < DSIZE; j++) {
      assert(answer[i * DSIZE + j] == host_output[i * DSIZE + j]);
    }
  }
  printf(" passed!\n");

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float memory_gbps = 2 * float(DSIZE) * float(DSIZE) * 4 / 1024 / 1024 / 1024 / (milliseconds / 1000);
  printf("Time: %f ms, Memory Bandwidth: %f GB/s\n", milliseconds, memory_gbps);
  return 0;
}

