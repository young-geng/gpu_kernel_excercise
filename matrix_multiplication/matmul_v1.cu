#include <stdio.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cblas.h>
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


const int DSIZE = 8 * 1024;
const int BLOCK_SIZE = 1024;
const int GRID_SIZE = 1024;

// Most naive matmul kernel, using blocks for rows of A and threads for cols of B
__global__ void matmul(const float *A, const float *B, float *output, int ds){
  float accu;
  for (int row_id = blockIdx.x; row_id < ds; row_id += gridDim.x) {
    for (int col_id = threadIdx.x; col_id < ds; col_id += blockDim.x) {
      accu = 0.0;
      for (int i = 0; i < ds; i++) {
        accu += A[row_id * ds + i] * B[i * ds + col_id];
      }
      output[row_id * ds + col_id] = accu;
    }
  }
}

int main(){
  float *host_matrix_A = new float[DSIZE * DSIZE];
  float *host_matrix_B = new float[DSIZE * DSIZE];
  float *host_output = new float[DSIZE * DSIZE];
  float *answer = new float[DSIZE * DSIZE];
  float *device_matrix_A, *device_matrix_B, *device_output;
  printf("Filling in random data.\n");
  #pragma omp parallel
  {
    std::random_device rd;
    std::mt19937 gen(rd() + omp_get_thread_num()); // Seed with thread ID
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    #pragma omp for collapse(2)
    for (int i = 0; i < DSIZE; i++) {
      for (int j = 0; j < DSIZE; j++) {
        host_matrix_A[i * DSIZE + j] = dis(gen);
        host_matrix_B[i * DSIZE + j] = dis(gen);
      }
    }
  }

  printf("Computing answer on CPU.\n");
  cblas_sgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    DSIZE,
    DSIZE,
    DSIZE,
    1.0f,
    host_matrix_A,
    DSIZE,
    host_matrix_B,
    DSIZE,
    1.0,
    answer,
    DSIZE
  );



  cudaMalloc(&device_matrix_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&device_matrix_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&device_output, DSIZE*DSIZE*sizeof(float));

  cudaCheckErrors("cudaMalloc failure"); // error checking
  cudaMemcpy(device_matrix_A, host_matrix_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_matrix_B, host_matrix_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  matmul<<<GRID_SIZE, BLOCK_SIZE>>>(device_matrix_A, device_matrix_B, device_output, DSIZE);

  cudaEventRecord(stop);
  cudaCheckErrors("kernel launch failure");
  cudaMemcpy(host_output, device_output, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time: %f ms\n", milliseconds);

  printf("Checking answers......");
  for(int i = 0; i < DSIZE; i++) {
    for(int j = 0; j < DSIZE; j++) {
      if (abs(host_output[i * DSIZE + j] - answer[i * DSIZE + j]) >= 1e-1) {
        printf("Incorrect result! GPU: %f, CPU: %f\n", host_output[i * DSIZE + j], answer[i * DSIZE + j]);
        return -1;
      }
    }
  }
  printf(" passed!\n");

  return 0;
}

