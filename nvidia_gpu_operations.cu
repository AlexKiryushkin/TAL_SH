
#include <cstdio>

#include "device_algebra.h"
#include "kernel_auxiliary_data.h"
#include "talsh_complex.h"

#ifndef NO_GPU
//GPU DEBUG FUNCTIONS:
__host__ int gpu_get_error_count()
/** Returns the total number of CUDA errors occured on current GPU.
    A negative return status means an error occurred. **/
{
  int i;
  cudaError_t err = cudaMemcpyFromSymbol((void*)&i, gpu_error_count, sizeof(gpu_error_count), 0, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) { return i; }
  else { return -1; }
}

__host__ int gpu_get_debug_dump(int* dump)
/** Returns the debug dump (int array) from current GPU.
    A positive return status is the length of the debug dump.
    A negative return status means an error occurred. **/
{
  cudaError_t err = cudaMemcpyFromSymbol((void*)dump, gpu_debug_dump, sizeof(int) * GPU_DEBUG_DUMP_SIZE, 0, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) { return GPU_DEBUG_DUMP_SIZE; }
  else { return -1; }
}
#endif /*NO_GPU*/

#ifndef NO_GPU

//NV-TAL INITIALIZATION/SHUTDOWN (internal use only):
__host__ int init_gpus(int gpu_beg, int gpu_end)
/** Initializes all GPU contexts for the current MPI process. Returned positive value is
the number of initialized GPUs. A negative return status means an error occured.
Each enabled GPU from the range [gpu_beg:gpu_end] will obtain its own cublasHandle as well.
The first GPU from the given range will be left active at the end. If <gpu_beg> > <gpu_end>,
no GPU will be initialized. **/
{
  int i, j, n, errc;
  void* base_ptr;
  cudaError_t err;
#ifndef NO_BLAS
  cublasStatus_t err_cublas;
#endif
#ifdef USE_CUTENSOR
  cutensorStatus_t err_cutensor;
#endif

  n = 0; for (i = 0; i < MAX_GPUS_PER_NODE; i++) gpu_up[i] = GPU_OFF; //initial GPU status
  if (gpu_beg >= 0 && gpu_end >= gpu_beg) {
    err = cudaGetDeviceCount(&i); if (err != cudaSuccess) return -1;
    if (gpu_end >= MAX_GPUS_PER_NODE || gpu_end >= i) return -2;
    //Initialize a mapped bank for tensor operation prefactors for GPU usage:
    errc = slab_clean(&prefactors); if (errc != 0) return -3;
    errc = slab_construct(&prefactors, sizeof(talshComplex8), (size_t)(MAX_GPUS_PER_NODE * MAX_CUDA_TASKS), sizeof(talshComplex8), 1U); if (errc != 0) return -4;
    errc = slab_get_base_ptr(&prefactors, &base_ptr); if (errc != 0) return -5;
    err = cudaHostGetDevicePointer(&gpu_prefs_base_ptr, base_ptr, 0); if (err != cudaSuccess) return -6;
    //Initialize each GPU device:
    for (i = gpu_end; i >= gpu_beg; i--) {
      err = cudaSetDevice(i);
      if (err == cudaSuccess) {
        gpu_up[i] = GPU_MINE; err = cudaGetDeviceProperties(&(gpu_prop[i]), i); if (err != cudaSuccess) gpu_up[i] = GPU_OFF;
        if (gpu_up[i] > GPU_OFF) {
          //SHMEM width:
          errc = gpu_set_shmem_width(GPU_SHMEM_WIDTH);
          if (errc != 0 && VERBOSE) printf("#WARNING(tensor_algebra_gpu_nvidia:init_gpus): Unable to set GPU SHMEM width %d: Error %d \n", GPU_SHMEM_WIDTH, errc);
#ifndef NO_BLAS
          //cuBLAS.v2 context:
          err_cublas = cublasCreate(&(cublas_handle[i]));
          if (err_cublas == CUBLAS_STATUS_SUCCESS) {
            gpu_up[i] = GPU_MINE_CUBLAS;
            err_cublas = cublasSetPointerMode(cublas_handle[i], CUBLAS_POINTER_MODE_DEVICE);
            if (err_cublas != CUBLAS_STATUS_SUCCESS) gpu_up[i] = GPU_MINE;
          }
#endif
#ifdef USE_CUTENSOR
          //cuTensor context:
          err_cutensor = cutensorInit(&(cutensor_handle[i]));
          if (err_cutensor != CUTENSOR_STATUS_SUCCESS) return -7;
          err = cudaMalloc(&(cutensor_workspace[i]), CUTENSOR_WORKSPACE_SIZE);
          if (err == cudaSuccess) {
            cutensor_worksize[i] = CUTENSOR_WORKSPACE_SIZE;
          }
          else {
            cutensor_workspace[i] = NULL;
            cutensor_worksize[i] = 0;
          }
#endif
        }
        //CUDA stream bank:
        if (gpu_up[i] > GPU_OFF) {
          for (j = 0; j < MAX_CUDA_TASKS; j++) CUDAStreamFreeHandle[i][j] = j; CUDAStreamFFE[i] = MAX_CUDA_TASKS;
          for (j = 0; j < MAX_CUDA_TASKS; j++) {
            err = cudaStreamCreate(&(CUDAStreamBank[i][j])); if (err != cudaSuccess) { gpu_up[i] = GPU_OFF; break; };
          }
        }
        //CUDA event bank:
        if (gpu_up[i] > GPU_OFF) {
          for (j = 0; j < MAX_CUDA_EVENTS; j++) CUDAEventFreeHandle[i][j] = j; CUDAEventFFE[i] = MAX_CUDA_EVENTS;
          for (j = 0; j < MAX_CUDA_EVENTS; j++) {
            err = cudaEventCreate(&(CUDAEventBank[i][j])); if (err != cudaSuccess) { gpu_up[i] = GPU_OFF; break; };
          }
        }
        //Last task:
        LastTask[i] = NULL;
        //Clear GPU statistics:
        gpu_stats[i].tasks_submitted = 0;
        gpu_stats[i].tasks_completed = 0;
        gpu_stats[i].tasks_deferred = 0;
        gpu_stats[i].tasks_failed = 0;
        gpu_stats[i].flops = 0.0;
        gpu_stats[i].traffic_in = 0.0;
        gpu_stats[i].traffic_out = 0.0;
        gpu_stats[i].time_active = 0.0;
        gpu_stats[i].time_start = clock();
        //Accept GPU as ready (active):
        if (gpu_up[i] > GPU_OFF) n++;
      }
    }
    //Peer memory access (UVA based):
#ifdef UNIFIED_ADDRESSING
    for (i = gpu_end; i >= gpu_beg; i--) {
      if (gpu_up[i] > GPU_OFF) {
        if (gpu_prop[i].unifiedAddressing != 0) {
          err = cudaSetDevice(i);
          if (err == cudaSuccess) {
            for (j = gpu_end; j >= gpu_beg; j--) {
              if (j != i && gpu_up[j] > GPU_OFF) {
                if (gpu_prop[j].unifiedAddressing != 0) {
                  err = cudaDeviceEnablePeerAccess(j, 0); //device i can access memory of device j
                  if ((err != cudaSuccess) && VERBOSE) printf("\n#MSG(tensor_algebra_gpu_nvidia): GPU peer no access: %d->%d\n", i, j);
                }
                else {
                  if (VERBOSE) printf("\n#MSG(tensor_algebra_gpu_nvidia): GPU peer no access: %d->%d\n", i, j);
                }
              }
            }
          }
          else {
            gpu_up[i] = GPU_OFF; n--;
          }
        }
        err = cudaGetLastError(); //clear the GPU#i error status
      }
    }
#endif
  }
  return n; //number of initialized GPU's
}

__host__ int free_gpus(int gpu_beg, int gpu_end)
/** Destroys all GPU/CUBLAS contexts on all GPU devices belonging to the MPI process.
A positive value returned is the number of failed GPUs; a negative one is an error.
If <gpu_beg> > <gpu_end>, nothing wil be done. **/
{
  int i, j, n, failure;
  cudaError_t err;
#ifndef NO_BLAS
  cublasStatus_t err_cublas;
#endif

  failure = 0; n = 0;
  if (gpu_beg >= 0 && gpu_end >= gpu_beg) {
    err = cudaGetDeviceCount(&i); if (err != cudaSuccess) return -1;
    if (gpu_end >= MAX_GPUS_PER_NODE || gpu_end >= i) return -2;
    //Free the mapped bank of tensor operation prefactors:
    i = slab_destruct(&prefactors); if (i != 0) failure++;
    gpu_prefs_base_ptr = NULL;
    //Free GPU devices:
    for (i = gpu_beg; i <= gpu_end; i++) {
      if (gpu_up[i] > GPU_OFF) {
        n++; err = cudaSetDevice(i);
        if (err == cudaSuccess) {
#ifdef USE_CUTENSOR
          if (cutensor_workspace[i] != NULL) cudaFree(cutensor_workspace[i]);
          cutensor_workspace[i] = NULL;
          cutensor_worksize[i] = 0;
#endif
#ifndef NO_BLAS
          if (gpu_up[i] >= GPU_MINE_CUBLAS) { err_cublas = cublasDestroy(cublas_handle[i]); if (err_cublas == CUBLAS_STATUS_SUCCESS) gpu_up[i] = GPU_MINE; }
#endif
          //CUDA stream bank:
          if (gpu_up[i] > GPU_OFF) {
            for (j = 0; j < MAX_CUDA_TASKS; j++) CUDAStreamFreeHandle[i][j] = j; CUDAStreamFFE[i] = MAX_CUDA_TASKS;
            for (j = 0; j < MAX_CUDA_TASKS; j++) { err = cudaStreamDestroy(CUDAStreamBank[i][j]); if (err != cudaSuccess) failure++; }
          }
          //CUDA event bank:
          if (gpu_up[i] > GPU_OFF) {
            for (j = 0; j < MAX_CUDA_EVENTS; j++) CUDAEventFreeHandle[i][j] = j; CUDAEventFFE[i] = MAX_CUDA_EVENTS;
            for (j = 0; j < MAX_CUDA_EVENTS; j++) { err = cudaEventDestroy(CUDAEventBank[i][j]); if (err != cudaSuccess) failure++; }
          }
          //Last task:
          LastTask[i] = NULL;
          n--; err = cudaDeviceReset();
        }
        gpu_up[i] = GPU_OFF; //GPU is taken out of use regardless of its status!
      }
    }
  }
  if (failure && VERBOSE) printf("#WARNING(tensor_algebra_gpu_nvidia:free_gpus): Resource deallocation was not fully successful!");
  return n;
}

__host__ int gpu_get_device_count(int* dev_count)
/** Returns the total number of NVIDIA GPUs found on the node. **/
{
  const char* err_msg;
  cudaError_t cuda_err = cudaGetDeviceCount(dev_count);
  if (cuda_err != cudaSuccess) {
    err_msg = cudaGetErrorString(cuda_err);
    if (VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_get_device_count): %s\n", err_msg);
    *dev_count = -1; return 1;
  }
  return 0;
}

__host__ int gpu_is_mine(int gpu_num)
/** Positive return: GPU is mine; 0: GPU is not mine; -1: invalid <gpu_num>. **/
{
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) { return gpu_up[gpu_num]; }
  else { return -1; }
}

__host__ int gpu_busy_least()
/** Returns the ID of the least busy GPU (non-negative) or -1 (no GPU found). **/
{
  int i, j, m, n;
  m = -1; n = -1;
  for (i = 0; i < MAX_GPUS_PER_NODE; i++) {
    if (gpu_up[i] > GPU_OFF) {
      j = gpu_stats[i].tasks_submitted - (gpu_stats[i].tasks_completed + gpu_stats[i].tasks_deferred + gpu_stats[i].tasks_failed);
      if (m >= 0) {
        if (j < m) { m = j; n = i; };
      }
      else {
        m = j; n = i;
      }
    }
  }
  return n;
}

__host__ int gpu_in_focus(int gpu_num)
/** If <gpu_num> is not passed here, returns the id of the current GPU in focus.
    If <gpu_num> is passed here, returns YEP if it is currently in focus, NOPE otherwise.
    In case of error, returns NVTAL_FAILURE (negative integer). **/
{
  int n;
  cudaError_t err;
  err = cudaGetDevice(&n); if (err != cudaSuccess) return NVTAL_FAILURE;
  if (gpu_num >= 0) { if (n == gpu_num) { return YEP; } else { return NOPE; } }
  if (n < 0 || n >= MAX_GPUS_PER_NODE) return NVTAL_FAILURE; //GPU id must not exceed the TALSH limit per node
  return n;
}

__host__ int gpu_activate(int gpu_num)
/** If GPU is enabled (mine), does cudaSetDevice; returns non-zero otherwise (error). **/
{
  int cur_gpu;
  cudaError_t err;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_up[gpu_num] > GPU_OFF) {
      cur_gpu = gpu_in_focus();
      if (cur_gpu != gpu_num) {
        err = cudaSetDevice(gpu_num);
        if (err != cudaSuccess) { if (cur_gpu >= 0) err = cudaSetDevice(cur_gpu); return 3; }
      }
    }
    else {
      return 2; //GPU is not mine
    }
  }
  else {
    return 1; //invalid <gpu_num>
  }
  return 0;
}

__host__ size_t gpu_device_memory_size(int gpu_num)
/** Returns the total memory (bytes) for a given GPU device. **/
{
  size_t bytes;

  bytes = 0;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_up[gpu_num] > GPU_OFF) bytes = gpu_prop[gpu_num].totalGlobalMem;
  }
  return bytes;
}

__host__ double gpu_get_flops(int gpu_num)
/** Returns the current flop count executed by GPU #gpu_num,
    or by all avaialble GPU devices if gpu_num = -1. **/
{
  int i, b, f;
  double total_flops;

  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    b = gpu_num; f = gpu_num; //select a specific GPU
  }
  else if (gpu_num == -1) {
    b = 0; f = MAX_GPUS_PER_NODE - 1; //select all GPUs
  }
  else {
    return -1.0; //invalid GPU number
  }
  total_flops = 0.0;
  for (i = b; i <= f; i++) {
    if (gpu_is_mine(i) != GPU_OFF) total_flops += gpu_stats[i].flops;
  }
  return total_flops;
}

//NV-TAL INTERNAL CONTROL:
__host__ int gpu_set_shmem_width(int width) {
  /** Sets the GPU shared memory bank width:
      <width> = R4: 4 bytes;
      <width> = R8: 8 bytes. **/
  cudaError_t cerr;
  if (width == R8) {
    cerr = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }
  else if (width == R4) {
    cerr = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
  }
  else {
    return 1; //invalid <width> passed
  }
  if (cerr != cudaSuccess) return 2;
  return 0;
}

__host__ int gpu_enable_fast_math(int gpu_num) {
  /** Enables fast math on GPU. **/
  int gs, gf, i;

  if (gpu_num >= 0) {
    gs = gpu_num; gf = gpu_num;
  }
  else {
    gs = 0; gf = MAX_GPUS_PER_NODE - 1;
  }
#ifndef NO_BLAS
  for (i = gs; i <= gf; ++i) {
    if (gpu_is_mine(i) >= GPU_MINE_CUBLAS) {
      if (cublasSetMathMode(cublas_handle[i], CUBLAS_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) return 1;
    }
    else {
      if (gpu_num >= 0) return 2;
    }
  }
#else
  return 3;
#endif
  return 0;
}

__host__ int gpu_disable_fast_math(int gpu_num) {
  /** Disables fast math on GPU. **/
  int gs, gf, i;

  if (gpu_num >= 0) {
    gs = gpu_num; gf = gpu_num;
  }
  else {
    gs = 0; gf = MAX_GPUS_PER_NODE - 1;
  }
#ifndef NO_BLAS
  for (i = gs; i <= gf; ++i) {
    if (gpu_is_mine(i) >= GPU_MINE_CUBLAS) {
      if (cublasSetMathMode(cublas_handle[i], CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS) return 1;
    }
    else {
      if (gpu_num >= 0) return 2;
    }
  }
#else
  return 3;
#endif
  return 0;
}

__host__ int gpu_query_fast_math(int gpu_num) {
  /** Queries the status of fast math on given GPU. **/
#ifndef NO_BLAS
  cublasMath_t math_mode;
  if (gpu_is_mine(gpu_num) >= GPU_MINE_CUBLAS) {
    if (cublasGetMathMode(cublas_handle[gpu_num], &math_mode) == CUBLAS_STATUS_SUCCESS) {
      if (math_mode == CUBLAS_TENSOR_OP_MATH) return YEP;
    }
  }
#endif
  return NOPE;
}

__host__ void gpu_set_transpose_algorithm(int alg) {
  /** Activates either the scatter or the shared-memory based tensor transpose algorithm.
      Invalid <alg> values will activate the basic shared-memory algorithm (default). **/
  if (alg == EFF_TRN_OFF) { TRANS_SHMEM = EFF_TRN_OFF; }
#ifdef USE_CUTT
  else if (alg == EFF_TRN_ON_CUTT) { TRANS_SHMEM = EFF_TRN_ON_CUTT; }
#endif
  else { TRANS_SHMEM = EFF_TRN_ON; } //any other value will result in the default setting
  return;
}

__host__ void gpu_set_matmult_algorithm(int alg) {
  /** Activates either cuBLAS (fast) or my own (slow) BLAS CUDA kernels. **/
#ifndef NO_BLAS
  if (alg == BLAS_ON) { DISABLE_BLAS = BLAS_ON; }
  else { DISABLE_BLAS = BLAS_OFF; };
#endif
  return;
}

__host__ int gpu_print_stats(int gpu_num)
/** Prints GPU statistics for GPU#<gpu_num>. If <gpu_num>=-1,
    prints GPU statistics for all active GPUs.
    A negative return status means invalid <gpu_num>. **/
{
  int i, b, f;
  double total_flops, total_traffic_in, total_traffic_out;
  clock_t ctm;

  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    b = gpu_num; f = gpu_num; //select a specific GPU
  }
  else if (gpu_num == -1) {
    b = 0; f = MAX_GPUS_PER_NODE - 1; //select all GPUs
  }
  else {
    return -1; //invalid GPU number
  }
  total_flops = 0.0; total_traffic_in = 0.0; total_traffic_out = 0.0;
  for (i = b; i <= f; i++) {
    if (gpu_is_mine(i) != GPU_OFF) {
      ctm = clock();
      gpu_stats[i].time_active = ((double)(ctm - gpu_stats[i].time_start)) / CLOCKS_PER_SEC;
      total_flops += gpu_stats[i].flops;
      total_traffic_in += gpu_stats[i].traffic_in;
      total_traffic_out += gpu_stats[i].traffic_out;
      printf("\n#MSG(TAL-SH::NV-TAL): Statistics on GPU #%d:\n", i);
      printf(" Number of tasks submitted: %llu\n", gpu_stats[i].tasks_submitted);
      printf(" Number of tasks completed: %llu\n", gpu_stats[i].tasks_completed);
      printf(" Number of tasks deferred : %llu\n", gpu_stats[i].tasks_deferred);
      printf(" Number of tasks failed   : %llu\n", gpu_stats[i].tasks_failed);
      printf(" Number of Flops processed: %G\n", gpu_stats[i].flops);
      printf(" Number of Bytes to GPU   : %G\n", gpu_stats[i].traffic_in);
      printf(" Number of Bytes from GPU : %G\n", gpu_stats[i].traffic_out);
      printf(" Time active (sec)        : %f\n", gpu_stats[i].time_active);
      printf("#END_MSG\n");
      //  }else{
      //   printf("\n#MSG(TAL-SH::NV-TAL): Statistics on GPU #%d: GPU is OFF\n",i);
    }
  }
  if (gpu_num == -1) {
    printf("\n#MSG(TAL-SH::NV-TAL): Statistics across all GPU devices:\n");
    printf(" Number of Flops processed   : %G\n", total_flops);
    printf(" Number of Bytes to GPUs     : %G\n", total_traffic_in);
    printf(" Number of Bytes from GPUs   : %G\n", total_traffic_out);
    if (total_traffic_in + total_traffic_out > 0.0) {
      printf(" Average arithmetic intensity: %G\n", total_flops / (total_traffic_in + total_traffic_out));
    }
    else {
      printf(" Average arithmetic intensity: %G\n", 0.0);
    }
    printf("#END_MSG\n");
  }
  return 0;
}
#endif /*NO_GPU*/

