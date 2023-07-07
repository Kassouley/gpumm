#ifndef DEVICE_HIP_H
#define DEVICE_HIP_H

#include <hip/hip_runtime.h>
#include "kernel.h"

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

#define GPUMM_ALLOC(ptr, size) \
{\
    CHECK(hipMalloc((void**)&ptr, size));\
}

#define GPUMM_MEMCPY_HtD(dst,src,size) \
{\
    CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));\
}

#define GPUMM_MEMCPY_DtH(dst,src,size) \
{\
    CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));\
}

#define GPUMM_FREE(ptr) \
{\
    CHECK(hipFree(ptr));\
}

#if defined(ROCBLAS_WO_DT) || defined(ROCBLAS)
#define GPUMM_HANDLE_ENABLE
#define GPUMM_BLAS_HANDLE rocblas_handle
#define GPUMM_HANDLE_CREATE(handle) \
{\
    rocblas_create_handle(&handle);\
}
#define GPUMM_HANDLE_DESTROY(handle) \
{\
    rocblas_destroy_handle(handle);\
}
#endif

#endif
