#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

unsigned int N = 1e6;
unsigned int mem_size = N * sizeof(float);
unsigned int block_size = 256;
unsigned int num_blocks = ((N + ( block_size + 1)) / block_size);

int interval_subtract( struct timeval* result, struct timeval* t2, struct timeval* t1) {
    unsigned int resolution = 1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1-> tv_sec);
    result->tv_sec = diff/resolution;
    result->tv_usec = diff%resolution;
    return (diff<0); 
}

int validate(float* h_out, float* cpu_out, float eps, int N) {
    for(int i=0; i<N; i++) {
        if(fabs(h_out[i] - cpu_out[i]) > eps) {
            printf("%.2f from gpu and %.2f from cpu", h_out[i], cpu_out[i]);
            return 1;
        }
    }
    return 0;
}

void simpleCPU(float* h_in, float *h_out, int N) {
    for(int i =0; i<N; i++){
        h_out[i] = (h_in[i] / (h_in[i] - 2.3)) * (h_in[i] / (h_in[i] - 2.3)) * (h_in[i] / (h_in[i] - 2.3));
    }
} 

__global__ void simpleKernel(float* d_in, float *d_out, int N) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    if (gid < N) {
        d_out[gid] = (d_in[gid] / (d_in[gid] - 2.3)) * (d_in[gid] / (d_in[gid] - 2.3)) * (d_in[gid] / (d_in[gid] - 2.3)); //(x/(x-2.3))^3
    }
}

int main(int argc, char** argv) {
    unsigned long int elapsed_gpu, elapsed_cpu; struct timeval t_start, t_end, t_diff;
    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);

    float* cpu_in = (float*) malloc(mem_size);
    float* cpu_out = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (float)(i+1);
        cpu_in[i] = (float)(i+1);
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    gettimeofday(&t_start, NULL);

    for(int i=0; i<10; i++){
        // execute the kernel
        simpleKernel<<< num_blocks, block_size>>>(d_in, d_out, N);
    } cudaThreadSynchronize();
    
    gettimeofday(&t_end, NULL);
    interval_subtract(&t_diff, &t_end, &t_start);

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    elapsed_gpu = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / 10;
    printf("Average GPU run took %d microseconds (%.2fms)\n", elapsed_gpu, elapsed_gpu / 1000.0);

    gettimeofday(&t_start, NULL);
    for(int i=0; i < 10; i++){
        simpleCPU(cpu_in, cpu_out, N);
    }

    gettimeofday(&t_end, NULL);
    interval_subtract(&t_diff, &t_end, &t_start);
    elapsed_cpu = (t_diff.tv_sec * 1e6 + t_diff.tv_usec) / 10;
    printf("Average CPU run took %d microseconds (%.2fms)\n", elapsed_cpu, (elapsed_cpu / 1000.0));
    //printf("Achieved speedup of %.2f\n", elapsed_cpu / elapsed_gpu);

    float eps = 0.0001;
    if (validate(h_out, cpu_out, eps, N) == 1) {
        printf("INVALID\n");
    }else {
        printf("VALID\n");
    }

    // clean-up memory
    free(h_in);       free(h_out);
    free(cpu_in);     free(cpu_out);
    cudaFree(d_in);   cudaFree(d_out);
}
