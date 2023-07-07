#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"
#include "device_cuda.h"

extern cublasHandle_t handle;

#ifdef CUBLAS
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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_c, n, &beta, d_a, n);
        
	CHECK(cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}
#endif


#ifdef CUBLAS_WO_DT
void kernel (unsigned int n, double* a, const double* b, const double* c)
{ 
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b, n, c, n, &beta, a, n);
}
#endif