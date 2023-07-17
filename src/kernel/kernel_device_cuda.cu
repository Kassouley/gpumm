#include <stdio.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "device_cuda.h"

__global__ void kernel_cuda (unsigned int n, double* a, const double* b, const double* c)
{ 
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) 
    {
        double value = 0.0;
        for (unsigned int i = 0; i < n; i++) 
        {
            value += b[row * n + i] * c[i * n + col];
        }
        a[row * n + col] = value;
    }
}

#ifdef CUDA_WO_DT
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    dim3 nbThreads (n, n);
    dim3 nbBlocks (1,1);
    if ( n > 32 )
    {
        nbThreads.x = 32;
        nbThreads.y = 32;
        nbBlocks.x = ceil(double(n)/double(nbThreads.x));
        nbBlocks.y = ceil(double(n)/double(nbThreads.y));
    }
    kernel_cuda<<<nbBlocks, nbThreads>>>(n, a, b, c);
}
#endif

#ifdef CUDA
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    int size = n * n * sizeof(double);

    double* d_a;
    double* d_b;
    double* d_c;
    
	CHECK(cudaMalloc(&d_a, size));
    CHECK(cudaMalloc(&d_b, size));
    CHECK(cudaMalloc(&d_c, size));

    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice));

    dim3 nbThreads (n, n);
    dim3 nbBlocks (1,1);
    if ( n > 32 )
    {
        nbThreads.x = 32;
        nbThreads.y = 32;
        nbBlocks.x = ceil(double(n)/double(nbThreads.x));
        nbBlocks.y = ceil(double(n)/double(nbThreads.y));
    }

    kernel_cuda<<<nbBlocks, nbThreads>>>(n, d_a, d_b, d_c);
        
	CHECK(cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}
#endif