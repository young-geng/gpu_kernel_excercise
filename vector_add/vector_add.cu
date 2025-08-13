#include <stdio.h>
#include <cassert>

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


const int DSIZE = 256 * 1024 * 1024; // 1GB of data
const int block_size = 1024;  // CUDA maximum is 1024
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  int idx = block_size * blockIdx.x + threadIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds)
    C[idx] = A[idx] + B[idx]; // do the vector (element) add here
}

int main(){

  float *h_A, *h_B, *h_C, *h_S, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  h_S = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
    h_S[i] = h_A[i] + h_B[i];
  }
  cudaMalloc(&d_A, DSIZE*sizeof(float));  // allocate device space for vector A
  cudaMalloc(&d_B, DSIZE*sizeof(float));  // allocate device space for vector B
  cudaMalloc(&d_C, DSIZE*sizeof(float));  // allocate device space for vector C
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  // copy vector B to device:
  cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vadd<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_B, d_C, DSIZE);
  cudaEventRecord(stop);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float memory_gbps = 2 * float(DSIZE) * 4 / 1024 / 1024 / 1024 / (milliseconds / 1000);
  printf("Time: %f ms, Memory Bandwidth: %f GB/s\n", milliseconds, memory_gbps);
  printf("Checking correctness...... ");
  for (int i = 0; i < DSIZE; i++){
    assert(h_S[i] == h_C[i] && "Incorrect result!");
  }
  printf("passed!\n");
  return 0;
}

