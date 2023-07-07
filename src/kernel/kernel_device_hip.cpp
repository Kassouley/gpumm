#include <stdio.h>
#include <hip/hip_runtime.h>
#include "kernel.h"
#include "device_hip.h"

__global__ void kernel_hip (unsigned int n, double* a, const double* b, const double* c)
{
    int col = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

#ifdef HIP_WO_DT
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
    hipLaunchKernelGGL(kernel_hip, nbBlocks, nbThreads, 0, 0, n, a, b, c);
}
#endif

#ifdef HIP
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

    dim3 nbThreads (n, n);
    dim3 nbBlocks (1,1);
    if ( n > 32 )
    {
        nbThreads.x = 32;
        nbThreads.y = 32;
        nbBlocks.x = ceil(double(n)/double(nbThreads.x));
        nbBlocks.y = ceil(double(n)/double(nbThreads.y));
    }

    hipLaunchKernelGGL(kernel_hip, nbBlocks, nbThreads, 0, 0, n, d_a, d_b, d_c);
        
	hipMemcpy(a, d_a, size, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
#endif