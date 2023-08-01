#ifndef DEVICE_CUDA_H
#define DEVICE_CUDA_H

#include <cuda_runtime.h>

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

#define GPUMM_ALLOC(ptr, size) \
{\
    CHECK(cudaMalloc(&ptr, size)); \
}

#define GPUMM_MEMCPY_HtD(dst,src,size) \
{\
    CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));\
}

#define GPUMM_MEMCPY_DtH(dst,src,size) \
{\
    CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));\
}

#define GPUMM_FREE(ptr) \
{\
    CHECK(cudaFree(ptr));\
}

#define GPUMM_DEVICE_SYNC() \
{\
    cudaDeviceSynchronize();\
}\

#if defined(CUBLAS_WO_DT) || defined(CUBLAS)
#define GPUMM_HANDLE_ENABLE
#define GPUMM_BLAS_HANDLE cublasHandle_t
#define GPUMM_HANDLE_CREATE(handle) \
{\
    cublasCreate(&handle);\
}
#define GPUMM_HANDLE_DESTROY(handle) \
{\
    cublasDestroy(handle);\
}
#endif

#endif