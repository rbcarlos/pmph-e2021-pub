#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "mmm-kernels.cu.h"

#define WIDTH_A  1024//1024 //1024//2048
#define HEIGHT_A 1024//2048//2048//2048
#define WIDTH_B  4096//2048
#define TILE     16

#define GPU_RUNS 100

/////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


void randomInit(float* data, int size) {
   for (int i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}


template<class T>
void matMult(T* A, T* B, T* C, int colsA, int rowsA, int colsB) {
  for(int i = 0; i < rowsA; i++) {
    for(int j = 0; j < colsB; j++) {
      float sum = 0.0;
      for(int k = 0; k < colsA; k++) {
        sum += A[i*colsA + k] * B[k * colsB + j];
      }
      C[i * colsB + j] = sum;
    }
  } 
}

template<class T>
bool validate(float* A,float* B, unsigned int sizeAB){
    for(int i = 0; i < sizeAB; i++)
      if (fabs(A[i] - B[i]) > 0.0005){
        printf("INVALID RESULT %d %f %f\n", i, A[i], B[i]);
        return false;
      }
    printf("VALID RESULT!\n");
    return true;
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////
 
int main() {
   // set seed for rand()
   srand(2006);
 
   // 1. allocate host memory for the two matrices
   unsigned int size_A = WIDTH_A * HEIGHT_A;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WIDTH_B * WIDTH_A;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);
 
   // 2. initialize host memory
   randomInit(h_A, size_A);
   randomInit(h_B, size_B);
    
   // 3. allocate device memory
   float* d_A;
   float* d_B;
   cudaMalloc((void**) &d_A, mem_size_A);
   cudaMalloc((void**) &d_B, mem_size_B);
 
   // 4. copy host memory to device
   cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
 
   // 5. allocate host memory for the result C
   unsigned int size_C = HEIGHT_A * WIDTH_B;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C   = (float*) malloc(mem_size_C);
   float* seq_C = (float*) malloc(mem_size_C);
 
   // 6. allocate device memory for the result
   float *d_C;
   cudaMalloc((void**) &d_C, mem_size_C);
 
   // 7. compute sequential matrix multiplication
   {
      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      matMult<float>(h_A, h_B, seq_C, WIDTH_A, HEIGHT_A, WIDTH_B);

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
      printf("Sequential Naive version runs in: %lu microsecs\n", elapsed);
   }

   

   // execute the naive kernel
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/TILE ); 
      int  dimx = ceil( ((float) WIDTH_B)/TILE );
      dim3 block(TILE, TILE, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int k=0; k<GPU_RUNS; k++) {
          matMultKer<float> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Naive MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Naive MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU Naive MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }

 
   // execute the block-tiled kernel
   {
      // setup execution parameters
      int  dimy = ceil( ((float)HEIGHT_A)/TILE ); 
      int  dimx = ceil( ((float) WIDTH_B)/TILE );
      dim3 block(TILE, TILE, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int k=0; k<GPU_RUNS; k++) {
        matMultTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A); 
      } 
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Block-Tiled MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Block-Tiled MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU Block-Tiled MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
      cudaMemset(d_C, 0, mem_size_C);
   }

   // execute the block+register tiled kernel
   // ToDo: please fill in the implementation below
   //       (for TILE = 16)
   {
      // 1. you would probably want to compute some valid grid and block here
      int  dimy = ceil( ((float) HEIGHT_A)/TILE ); 
      int  dimx = ceil( ((float) WIDTH_B)/TILE );
      dim3 block(TILE, TILE, 1);
      dim3 grid (dimx, dimy, 1);

      unsigned long int elapsed;
      struct timeval t_start, t_end, t_diff;
      gettimeofday(&t_start, NULL); 
      
      for(int k=0; k<GPU_RUNS; k++) {
          // 2. you would probably want to call here the kernel: 
          matMultRegTiledKer<float,TILE> <<< grid, block >>>(d_A, d_B, d_C, HEIGHT_A, WIDTH_B, WIDTH_A);
      }
      cudaDeviceSynchronize();

      gettimeofday(&t_end, NULL);
      timeval_subtract(&t_diff, &t_end, &t_start);
      elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

      // copy result from device to host
      cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
      // validate
      printf("GPU Block+Register Tiled MMM version ... ");
      validate<float>(seq_C, h_C, size_C);

      printf("GPU Block+Register Tiled MMM version runs in: %lu microsecs\n", elapsed);
      float microsecPerMatrixMul = elapsed; 
      double flopsPerMatrixMul = 2.0 * HEIGHT_A * WIDTH_B * WIDTH_A; 
      double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (microsecPerMatrixMul / (1000.0f * 1000.0f)); 
      printf( "GPU Block+Register Tiled MMM Performance= %.2f GFlop/s, Time= %.3f microsec %d %d\n", gigaFlops, microsecPerMatrixMul, grid.x, grid.y); 
   }


   // 7. clean up memory
   free(h_A);
   free(h_B);
   free(h_C);
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);
}

