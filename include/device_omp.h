#ifndef DEVICE_OMP_H
#define DEVICE_OMP_H

#include <omp.h>

#define GPUMM_ALLOC(ptr, size) \
{\
    ptr = omp_target_alloc(size, 0);\
    if ( ptr == NULL ) \
    { \
        fprintf(stderr, "error: 'malloc ptr is null' at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
}

#define GPUMM_MEMCPY_HtD(dst,src,size) \
{\
    omp_target_memcpy(dst, src, size, 0, 0, 0, omp_get_initial_device());\
}

#define GPUMM_MEMCPY_DtH(dst,src,size) \
{\
    omp_target_memcpy(dst, src, size, 0, 0, omp_get_initial_device(), 0);\
}

#define GPUMM_FREE(ptr) \
{\
    omp_target_free(ptr, 0);\
}

#endif