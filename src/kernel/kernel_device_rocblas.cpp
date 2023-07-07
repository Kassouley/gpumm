#include <stdio.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "device_hip.h"

extern rocblas_handle handle;

#ifdef ROCBLAS
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    int size = n * n * sizeof(double);
    
    double* d_a;
    double* d_b;
    double* d_c;

    hipMalloc((void**)&d_a, size);
    hipMalloc((void**)&d_b, size);
    hipMalloc((void**)&d_c, size);

    hipMemcpy(d_b, b, size, hipMemcpyHostToDevice);
    hipMemcpy(d_c, c, size, hipMemcpyHostToDevice);

    const double alpha = 1.0f; 
    const double beta = 0.0f;

    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                n, n, n, &alpha, d_c, n, d_b, n, &beta, d_a, n);

    hipMemcpy(a, d_a, size, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
#endif

#ifdef ROCBLAS_WO_DT
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    const double alpha = 1.0f; 
    const double beta = 0.0f; 
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                n, n, n, &alpha, c, n, b, n, &beta, a, n);
}
#endif

