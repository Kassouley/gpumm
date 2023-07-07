#ifndef KERNEL_H
#define KERNEL_H

#if defined(HIP_WO_DT) || defined(HIP)
#include <hip/hip_runtime.h>
__global__ void kernel_hip (unsigned int n, double* a, const double* b, const double* c);
#include "device_hip.h"
#endif

#if defined(CUDA_WO_DT) || defined(CUDA)
#include <cuda_runtime.h>
__global__ void kernel_cuda (unsigned int n, double* a, const double* b, const double* c);
#include "device_cuda.h"
#endif

#if defined(ROCBLAS_WO_DT) || defined(ROCBLAS)
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "device_hip.h"
#endif

#if defined(CUBLAS_WO_DT) || defined(CUBLAS)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_cuda.h"
#endif

#if defined(GPU_OMP_WO_DT) || defined(GPU_OMP)
#include <omp.h>
#include "device_omp.h"
#endif

#if defined(OPENACC_WO_DT) || defined(OPENACC)
#include <openacc.h>
#include "device_acc.h"
#endif

void kernel (unsigned int n, double* a, const double* b, const double* c);

#endif