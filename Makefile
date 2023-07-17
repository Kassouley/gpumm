# -------------------- CC -------------------- #

ifneq ($(filter $(KERNEL), HIP HIP_WO_DT ROCBLAS ROCBLAS_WO_DT),)
	CC=hipcc
else ifneq ($(filter $(KERNEL), CUDA CUDA_WO_DT CUBLAS CUBLAS_WO_DT),)
	CC=nvcc
else
	ifeq ($(GPU), AMD)
		CC=/opt/rocm/llvm/bin/clang
	else ifeq ($(GPU), NVIDIA)
		CC=nvc
	endif
endif

# ------------------ CFLAGS ------------------ #

CFLAGS=-g -O3 -lm -I./include -D $(KERNEL)
CMEASURE=-D $(CLOCK)

# ------------------ LFLAGS ------------------ #

ifeq ($(KERNEL),CBLAS)
	LFLAGS=-lblas
else ifneq ($(filter $(KERNEL), ROCBLAS ROCBLAS_WO_DT),)
	LFLAGS=-lrocblas  -L/opt/rocm-5.4.3/rocblas/lib/librocblas.so  -I/opt/rocm-5.4.3/include/
else ifneq ($(filter $(KERNEL), CUBLAS CUBLAS_WO_DT),)
	LFLAGS=-lcublas
endif

# ----------------- OPT_FLAGS ----------------- #

ifeq ($(KERNEL),CPU_OMP)
	OPT_FLAGS=-fopenmp
else ifneq ($(filter $(KERNEL), GPU_OMP GPU_OMP_WO_DT),)
	ifeq ($(GPU), AMD)
		OPT_FLAGS=-fopenmp=libomp -target x86_64-pc-linux-gnu \
		-fopenmp-targets=amdgcn-amd-amdhsa \
		-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030
	else ifeq ($(GPU), NVIDIA)
		OPT_FLAGS=-fopenmp -mp=gpu -Minfo=mp
	endif
else ifneq ($(filter $(KERNEL), OPENACC OPENACC_WO_DT),)
	OPT_FLAGS=-acc -Minfo=accel 
else ifneq ($(filter $(KERNEL), CUDA CUDA_WO_DT CUBLAS CUBLAS_WO_DT),)
	OPT_FLAGS=-gencode=arch=compute_52,code=sm_52 \
  			  -gencode=arch=compute_60,code=sm_60 \
  			  -gencode=arch=compute_61,code=sm_61 \
  			  -gencode=arch=compute_70,code=sm_70 \
  			  -gencode=arch=compute_75,code=sm_75 \
  			  -gencode=arch=compute_80,code=sm_80 \
  			  -gencode=arch=compute_80,code=compute_80
endif

# ------------------- SRC ------------------- #

SRC_COMMON=src/tab.c 

KERNEL_DIR=./src/kernel
BENCH_DIR=./src/bench
CHECK_DIR=./src/check
CALIB_DIR=./src/calibrate

IS_KERNEL_IN_C := $(filter $(KERNEL), BASIS CPU_OMP CBLAS GPU_OMP OPENACC)
IS_KERNEL_IN_C_WO_DT := $(filter $(KERNEL), GPU_OMP_WO_DT OPENACC_WO_DT)
IS_KERNEL_IN_CPP := $(filter $(KERNEL), HIP ROCBLAS CUDA CUBLAS)
IS_KERNEL_IN_CPP_WO_DT := $(filter $(KERNEL), HIP_WO_DT ROCBLAS_WO_DT CUDA_WO_DT CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_IN_C),)
	SRC_CHECKER=$(CHECK_DIR)/driver_check.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_CHECKER=$(CHECK_DIR)/driver_check_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_CHECKER=$(CHECK_DIR)/driver_check.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_CHECKER=$(CHECK_DIR)/driver_check_wo_dt.cpp
endif

ifneq ($(IS_KERNEL_IN_C),)
	SRC_DRIVER=$(BENCH_DIR)/driver_measure.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_DRIVER=$(BENCH_DIR)/driver_measure_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_DRIVER=$(BENCH_DIR)/driver_measure.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_DRIVER=$(BENCH_DIR)/driver_measure_wo_dt.cpp
endif

ifneq ($(IS_KERNEL_IN_C),)
	SRC_CALIB=$(CALIB_DIR)/driver_calib.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_CALIB=$(CALIB_DIR)/driver_calib_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_CALIB=$(CALIB_DIR)/driver_calib.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_CALIB=$(CALIB_DIR)/driver_calib_wo_dt.cpp
endif

IS_KERNEL_CPU := $(filter $(KERNEL), BASIS CPU_OMP CBLAS)
IS_KERNEL_OMP := $(filter $(KERNEL), GPU_OMP GPU_OMP_WO_DT)
IS_KERNEL_ACC := $(filter $(KERNEL), OPENACC OPENACC_WO_DT)
IS_KERNEL_HIP := $(filter $(KERNEL), HIP HIP_WO_DT)
IS_KERNEL_ROCBLAS := $(filter $(KERNEL), ROCBLAS ROCBLAS_WO_DT)
IS_KERNEL_CUDA := $(filter $(KERNEL), CUDA CUDA_WO_DT)
IS_KERNEL_CUBLAS := $(filter $(KERNEL), CUBLAS CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_CPU),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_cpu.c 
else ifneq ($(IS_KERNEL_OMP),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_omp.c
else ifneq ($(IS_KERNEL_ACC),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_acc.c
else ifneq ($(IS_KERNEL_HIP),)
	SRC_KERNEL=$(KERNEL_DIR)/test.cpp
else ifneq ($(IS_KERNEL_ROCBLAS),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_rocblas.cpp
else ifneq ($(IS_KERNEL_CUDA),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_cuda.cu
else ifneq ($(IS_KERNEL_CUBLAS),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_cublas.cu
endif

all: check calibrate measure

check: src/tab.c
	$(CC) -o $@ $^ $(SRC_KERNEL) $(SRC_CHECKER) $(CFLAGS) $(LFLAGS) $(OPT_FLAGS)

measure: src/tab.c src/rdtsc.c src/print_measure.c src/time_measure.c 
	$(CC) -o $@ $^ $(SRC_KERNEL) $(SRC_DRIVER) $(CFLAGS) $(CMEASURE) $(LFLAGS) $(OPT_FLAGS)

calibrate: src/tab.c src/rdtsc.c src/print_calib.c
	$(CC) -o $@ $^ $(SRC_KERNEL) $(SRC_CALIB) $(CFLAGS) $(CMEASURE) $(LFLAGS) $(OPT_FLAGS)
	
# measure_src:
	
# driver_check.o: driver_check.c
# 	$(CC) $(CFLAGS) -c $< -o $@
# driver_calib.o: driver_calib.c
# 	$(CC) $(CFLAGS) -c $< -o $@
# driver.o: driver.c
# 	$(CC) $(CFLAGS) -c $< -o $@ 


# kernel.o: kernel.c
# 	$(CC) $(CFLAGS) $(OMPFLAGS) -c $< -o $@

clean:
	rm -rf *.o check calibrate measure

