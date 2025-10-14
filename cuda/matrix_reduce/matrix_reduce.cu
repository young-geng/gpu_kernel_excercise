#include <stdio.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cooperative_groups.h>

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


const int DSIZE = 32 * 1024;
const int BLOCK_SIZE = 1024;
const int GRID_SIZE = 340;


__device__ void block_reduce(float val, float *sdata){
  sdata[threadIdx.x] = val;
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  for (int remaining_size = blockDim.x; remaining_size > 1; remaining_size = ceil_div(remaining_size, warpSize)) {
     __syncthreads();
    if (threadIdx.x < remaining_size) {
      val = sdata[threadIdx.x];  // Load results from last round
      for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        // Perform the reduction in a warp.
        val += __shfl_down_sync(__activemask(), val, offset);
      }
    }
    __syncthreads();
    if (threadIdx.x < remaining_size && lane_id == 0) {
      sdata[warp_id] = val;
    }
  }
  __syncthreads();
}

__global__ void matrix_reduce(const float *matrix, float *output, int ds){
  float val;
  __shared__ float sdata[BLOCK_SIZE];

  for (int row_id = blockIdx.x; row_id < ds; row_id += gridDim.x) {
    val = 0.0;
    for(int i = threadIdx.x; i < ds; i += blockDim.x) {
      val += matrix[row_id * ds + i];
    }
    block_reduce(val, sdata);
    if(threadIdx.x == 0) {
      output[row_id] = sdata[0];
    }
  }
}

int main(){
  float *host_matrix = new float[DSIZE * DSIZE];
  float *host_output = new float[DSIZE];
  float *answer = new float[DSIZE];
  double temp;
  float *device_matrix, *device_output;
  for (int i = 0; i < DSIZE; i++){
    temp = 0.0;
    for(int j = 0; j < DSIZE; j++) {
      host_matrix[i * DSIZE + j] = rand()/(float)RAND_MAX / sqrt(DSIZE);
      temp += (double)host_matrix[i * DSIZE + j];
    }
    answer[i] = (float)temp;
  }
  cudaMalloc(&device_matrix, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&device_output, DSIZE*sizeof(float));

  cudaCheckErrors("cudaMalloc failure"); // error checking
  cudaMemcpy(device_matrix, host_matrix, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matrix_reduce<<<GRID_SIZE, BLOCK_SIZE>>>(device_matrix, device_output, DSIZE);

  cudaEventRecord(stop);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(host_output, device_output, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("Checking answers......");
  for(int i = 0; i < DSIZE; i++) {
    if(abs(host_output[i] - answer[i]) > 1e-3) {
      printf("Host: %f, device: %f\n", answer[i], host_output[i]);
    }
    assert(abs(host_output[i] - answer[i]) < 1e-1);
  }
  printf(" passed!\n");

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time: %f ms\n", milliseconds);
  return 0;
}

