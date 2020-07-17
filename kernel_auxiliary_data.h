#pragma once

#ifndef NO_GPU

#include <cuda.h>
#include <cuda_runtime.h>

#endif

#ifndef NO_GPU
//PARAMETERS:
#define GPU_DEBUG_DUMP_SIZE 128 //size of the GPU debug dump (int array)
#endif /*NO_GPU*/

//----------------------------------------------------------------------------------------------------
//PARAMETERS:
static int VERBOSE = 1; //verbosity for error messages
static int DEBUG = 0; //debugging mode
#ifndef NO_GPU
//GLOBAL DATA:
// GPU control on the current MPI process:
static int gpu_up[MAX_GPUS_PER_NODE] = { GPU_OFF }; //GPU_OFF(0): GPU is disabled; GPU_MINE(1): GPU is enabled; GPU_MINE_CUBLAS(2): GPU is BLAS enabled
static cudaDeviceProp gpu_prop[MAX_GPUS_PER_NODE]; //properties of all GPUs present on the node
static talsh_stats_t gpu_stats[MAX_GPUS_PER_NODE]; //runtime statistics for all GPUs present on the node
#ifndef NO_BLAS
// Infrastructure for CUBLAS:
static cublasHandle_t cublas_handle[MAX_GPUS_PER_NODE]; //each GPU present on a node obtains its own cuBLAS context handle
#endif /*NO_BLAS*/
#ifdef USE_CUTENSOR
// Infrastructure for cuTensor:
static cutensorHandle_t cutensor_handle[MAX_GPUS_PER_NODE];   //each GPU present on a node obtains its own cuTensor context handle
static void* cutensor_workspace[MAX_GPUS_PER_NODE] = { NULL }; //cuTensor workspace (in GPU memory)
static size_t cutensor_worksize[MAX_GPUS_PER_NODE] = { 0 };     //cuTensor workspace size
static const size_t CUTENSOR_WORKSPACE_SIZE = 128 * 1048576;  //default cuTensor workspace size
#endif /*USE_CUTENSOR*/
// Slabs for the GPU asynchronous resources:
//  CUDA stream handles:
static cudaStream_t CUDAStreamBank[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //pre-allocated CUDA stream handles (for each CUDA device)
static int CUDAStreamFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //free CUDA stream handles
static int CUDAStreamFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAStreamFreeHandle
//  CUDA event handles:
static cudaEvent_t CUDAEventBank[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //pre-allocated CUDA event handles (for each CUDA device)
static int CUDAEventFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //free CUDA event handles
static int CUDAEventFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAEventFreeHandle
// Mapped slab of tensor operation prefactors for GPU usage:
static slab_t prefactors;         //mapped slab of prefactors
static void* gpu_prefs_base_ptr; //mapped device pointer of the slab base
// Slab of GPU constant memory arguments for each GPU (managed by "mem_manager.cpp"):
__device__ __constant__ int const_args_dims[MAX_GPU_ARGS][MAX_TENSOR_RANK]; //storage for device constant memory arguments: dimension extents
__device__ __constant__ int const_args_prmn[MAX_GPU_ARGS][MAX_TENSOR_RANK]; //storage for device constant memory arguments: permutation
// GPU error control and debugging for each GPU:
__device__ static int gpu_error_count = 0; //total number of CUDA errors registered on device till the current moment
__device__ static int gpu_debug_dump[GPU_DEBUG_DUMP_SIZE]; //debug dump
// Global CUDA event recording policy:
static int PRINT_TIMING = 1; //non-zero value enables time printing statements
// Infrastructure for function <gpu_tensor_block_copy_dlf> (blocking and non-blocking):
#ifdef USE_CUTT
static int TRANS_SHMEM = EFF_TRN_ON_CUTT; //switch between shared-memory tensor transpose and scatter tensor transpose
#else
static int TRANS_SHMEM = EFF_TRN_ON; //switch between shared-memory tensor transpose and scatter tensor transpose
#endif /*USE_CUTT*/
// Infrastructure for <gpu_tensor_block_contract_dlf> (non-blocking):
#ifndef NO_BLAS
static int DISABLE_BLAS = 0; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#else
static int DISABLE_BLAS = 1; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#endif /*NO_BLAS*/
static cudaTask_t* LastTask[MAX_GPUS_PER_NODE]; //last CUDA task successfully scheduled on each GPU
static float h_sgemm_beta_one = 1.0f;
static float h_sgemm_beta_zero = 0.0f;
static double h_dgemm_beta_one = 1.0;
static double h_dgemm_beta_zero = 0.0;
static cuComplex h_cgemm_beta_one = { 1.0f,0.0f };
static cuComplex h_cgemm_beta_zero = { 0.0f,0.0f };
static cuDoubleComplex h_zgemm_beta_one = { 1.0,0.0 };
static cuDoubleComplex h_zgemm_beta_zero = { 0.0,0.0 };
__device__ __constant__ static float sgemm_alpha_plus = 1.0f;                  //default alpha constant for SGEMM
__device__ __constant__ static float sgemm_alpha_minus = -1.0f;                //default alpha constant for SGEMM
__device__ __constant__ static float sgemm_beta_one = 1.0f;                    //default beta constant SGEMM
__device__ __constant__ static float sgemm_beta_zero = 0.0f;                   //zero beta constant SGEMM
__device__ __constant__ static double dgemm_alpha_plus = 1.0;                  //default alpha constant for DGEMM
__device__ __constant__ static double dgemm_alpha_minus = -1.0;                //default alpha constant for DGEMM
__device__ __constant__ static double dgemm_beta_one = 1.0;                    //default beta constant DGEMM
__device__ __constant__ static double dgemm_beta_zero = 0.0;                   //zero beta constant DGEMM
__device__ __constant__ static cuComplex cgemm_alpha_plus = { 1.0f,0.0f };       //default alpha constant CGEMM
__device__ __constant__ static cuComplex cgemm_alpha_minus = { -1.0f,0.0f };     //default alpha constant CGEMM
__device__ __constant__ static cuComplex cgemm_beta_one = { 1.0f,0.0f };         //default beta constant CGEMM
__device__ __constant__ static cuComplex cgemm_beta_zero = { 0.0f,0.0f };        //zero beta constant CGEMM
__device__ __constant__ static cuDoubleComplex zgemm_alpha_plus = { 1.0,0.0 };   //default alpha constant ZGEMM
__device__ __constant__ static cuDoubleComplex zgemm_alpha_minus = { -1.0,0.0 }; //default alpha constant ZGEMM
__device__ __constant__ static cuDoubleComplex zgemm_beta_one = { 1.0,0.0 };     //default beta constant ZGEMM
__device__ __constant__ static cuDoubleComplex zgemm_beta_zero = { 0.0,0.0 };    //zero beta constant ZGEMM
// Infrastructure for kernels <gpu_array_norm2__>:
__device__ static int norm2_wr_lock = 0; //write lock shared by all <gpu_array_norm2__> running on GPU
// Infrastructure for kernels <gpu_array_dot_product__>:
__device__ static int dot_product_wr_lock = 0; //write lock shared by all <gpu_array_dot_product__> running on GPU
#endif /*NO_GPU*/
