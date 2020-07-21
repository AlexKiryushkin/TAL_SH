#pragma once

#ifndef NO_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#endif

#ifndef NO_GPU
//PARAMETERS:
#define GPU_DEBUG_DUMP_SIZE 128 //size of the GPU debug dump (int array)
#endif /*NO_GPU*/

#include "device_algebra.h"
#include "mem_manager.h"

//----------------------------------------------------------------------------------------------------
//PARAMETERS:
extern int VERBOSE; //verbosity for error messages
extern int DEBUG; //debugging mode
#ifndef NO_GPU
//GLOBAL DATA:
// GPU control on the current MPI process:
extern int gpu_up[MAX_GPUS_PER_NODE]; //GPU_OFF(0): GPU is disabled; GPU_MINE(1): GPU is enabled; GPU_MINE_CUBLAS(2): GPU is BLAS enabled
extern cudaDeviceProp gpu_prop[MAX_GPUS_PER_NODE]; //properties of all GPUs present on the node
extern talsh_stats_t gpu_stats[MAX_GPUS_PER_NODE]; //runtime statistics for all GPUs present on the node
#ifndef NO_BLAS
// Infrastructure for CUBLAS:
extern cublasHandle_t cublas_handle[MAX_GPUS_PER_NODE]; //each GPU present on a node obtains its own cuBLAS context handle
#endif /*NO_BLAS*/
#ifdef USE_CUTENSOR
// Infrastructure for cuTensor:
extern cutensorHandle_t cutensor_handle[MAX_GPUS_PER_NODE];   //each GPU present on a node obtains its own cuTensor context handle
extern void* cutensor_workspace[MAX_GPUS_PER_NODE]; //cuTensor workspace (in GPU memory)
extern size_t cutensor_worksize[MAX_GPUS_PER_NODE];     //cuTensor workspace size
extern const size_t CUTENSOR_WORKSPACE_SIZE;  //default cuTensor workspace size
#endif /*USE_CUTENSOR*/
// Slabs for the GPU asynchronous resources:
//  CUDA stream handles:
extern cudaStream_t CUDAStreamBank[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //pre-allocated CUDA stream handles (for each CUDA device)
extern int CUDAStreamFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //free CUDA stream handles
extern int CUDAStreamFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAStreamFreeHandle
//  CUDA event handles:
extern cudaEvent_t CUDAEventBank[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //pre-allocated CUDA event handles (for each CUDA device)
extern int CUDAEventFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //free CUDA event handles
extern int CUDAEventFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAEventFreeHandle
// Mapped slab of tensor operation prefactors for GPU usage:
extern slab_t prefactors;         //mapped slab of prefactors
extern void* gpu_prefs_base_ptr; //mapped device pointer of the slab base
// Slab of GPU constant memory arguments for each GPU (managed by "mem_manager.cpp"):
__device__ __constant__ int const_args_dims[MAX_GPU_ARGS][MAX_TENSOR_RANK]; //storage for device constant memory arguments: dimension extents
__device__ __constant__ int const_args_prmn[MAX_GPU_ARGS][MAX_TENSOR_RANK]; //storage for device constant memory arguments: permutation
// GPU error control and debugging for each GPU:
__device__ int gpu_error_count = 0; //total number of CUDA errors registered on device till the current moment
__device__ int gpu_debug_dump[GPU_DEBUG_DUMP_SIZE]; //debug dump
// Global CUDA event recording policy:
extern int PRINT_TIMING ; //non-zero value enables time printing statements
// Infrastructure for function <gpu_tensor_block_copy_dlf> (blocking and non-blocking):
#ifdef USE_CUTT
extern int TRANS_SHMEM; //switch between shared-memory tensor transpose and scatter tensor transpose
#else
extern int TRANS_SHMEM; //switch between shared-memory tensor transpose and scatter tensor transpose
#endif /*USE_CUTT*/
// Infrastructure for <gpu_tensor_block_contract_dlf> (non-blocking):
#ifndef NO_BLAS
extern int DISABLE_BLAS; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#else
extern int DISABLE_BLAS; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#endif /*NO_BLAS*/
extern cudaTask_t* LastTask[MAX_GPUS_PER_NODE]; //last CUDA task successfully scheduled on each GPU
extern float h_sgemm_beta_one;
extern float h_sgemm_beta_zero;
extern double h_dgemm_beta_one;
extern double h_dgemm_beta_zero;
extern cuComplex h_cgemm_beta_one;
extern cuComplex h_cgemm_beta_zero;
extern cuDoubleComplex h_zgemm_beta_one;
extern cuDoubleComplex h_zgemm_beta_zero;
__device__ __constant__ float sgemm_alpha_plus = 1.0f;                  //default alpha constant for SGEMM
__device__ __constant__ float sgemm_alpha_minus = -1.0f;                //default alpha constant for SGEMM
__device__ __constant__ float sgemm_beta_one = 1.0f;                    //default beta constant SGEMM
__device__ __constant__ float sgemm_beta_zero = 0.0f;                   //zero beta constant SGEMM
__device__ __constant__ double dgemm_alpha_plus = 1.0;                  //default alpha constant for DGEMM
__device__ __constant__ double dgemm_alpha_minus = -1.0;                //default alpha constant for DGEMM
__device__ __constant__ double dgemm_beta_one = 1.0;                    //default beta constant DGEMM
__device__ __constant__ double dgemm_beta_zero = 0.0;                   //zero beta constant DGEMM
__device__ __constant__ cuComplex cgemm_alpha_plus = { 1.0f,0.0f };       //default alpha constant CGEMM
__device__ __constant__ cuComplex cgemm_alpha_minus = { -1.0f,0.0f };     //default alpha constant CGEMM
__device__ __constant__ cuComplex cgemm_beta_one = { 1.0f,0.0f };         //default beta constant CGEMM
__device__ __constant__ cuComplex cgemm_beta_zero = { 0.0f,0.0f };        //zero beta constant CGEMM
__device__ __constant__ cuDoubleComplex zgemm_alpha_plus = { 1.0,0.0 };   //default alpha constant ZGEMM
__device__ __constant__ cuDoubleComplex zgemm_alpha_minus = { -1.0,0.0 }; //default alpha constant ZGEMM
__device__ __constant__ cuDoubleComplex zgemm_beta_one = { 1.0,0.0 };     //default beta constant ZGEMM
__device__ __constant__ cuDoubleComplex zgemm_beta_zero = { 0.0,0.0 };    //zero beta constant ZGEMM
#endif /*NO_GPU*/
