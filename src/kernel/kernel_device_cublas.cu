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

    double alpha = 1.0f;
    double beta = 0.0f;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_c, n, d_b, n, &beta, d_a, n);
        
	CHECK(cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}
#endif


#ifdef CUBLAS_WO_DT
void kernel (unsigned int n, double* a, const double* b, const double* c)
{ 
    double alpha = 1.0f;
    double beta = 0.0f;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, c, n, b, n, &beta, a, n);
}
#endif