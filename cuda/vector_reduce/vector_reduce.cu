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


const int DSIZE = 256 * 1024 * 1024;
const int BLOCK_SIZE = 1024;
const int GRID_SIZE = 170;


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
}

__global__ void vreduce(const float *vector, float *output, int ds){
  float val = 0.0;
  __shared__ float sdata[BLOCK_SIZE];

  auto global_thread_group = cooperative_groups::this_grid();
  int global_thread_id = global_thread_group.thread_rank();
  int global_n_threads = global_thread_group.size();

  // Load all data;
  for(int i = global_thread_id; i < ds; i += global_n_threads) {
    val += vector[i];
  }

  int num_remaining_elements = global_n_threads;

  while(num_remaining_elements > 1){
    block_reduce(val, sdata);
    if(threadIdx.x == 0 && global_thread_id < num_remaining_elements) {
      output[blockIdx.x] = sdata[0];
    }
    num_remaining_elements = ceil_div(num_remaining_elements, blockDim.x);
    global_thread_group.sync();
    if (global_thread_id < num_remaining_elements) {
      val = output[global_thread_id];
    } else {
      val = 0.0;
    }
  }
}

int main(){
  float *host_vector = new float[DSIZE];  // allocate space for vectors in host memory
  float *host_output = new float[1];
  double answer = 0.0;
  float *device_vector, *device_output;
  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    host_vector[i] = rand()/(float)RAND_MAX / sqrt(DSIZE);
    // We need to perform the reduction sequenctially in double, otherwise we get
    // wrong answer for large amount of numbers to reduce.
    answer += (double)host_vector[i];
  }
  cudaMalloc(&device_vector, DSIZE*sizeof(float));
  cudaMalloc(&device_output, GRID_SIZE*sizeof(float));

  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy vector A to device:
  cudaMemcpy(device_vector, host_vector, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // vreduce<<<GRID_SIZE, BLOCK_SIZE>>>(device_vector, device_output, DSIZE);
  void *args[] = {(void *)&device_vector, (void *)&device_output, (void *)&DSIZE};
  cudaLaunchCooperativeKernel((void*)vreduce, GRID_SIZE, BLOCK_SIZE, args);

  cudaEventRecord(stop);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  printf("Host answer: %f, device output: %f\n", (float)answer, host_output[0]);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time: %f ms\n", milliseconds);
  return 0;
}

