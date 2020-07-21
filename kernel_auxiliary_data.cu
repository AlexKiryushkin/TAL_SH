
#include "kernel_auxiliary_data.h"

//----------------------------------------------------------------------------------------------------
//PARAMETERS:
int VERBOSE = 1; //verbosity for error messages
int DEBUG = 0; //debugging mode
#ifndef NO_GPU
//GLOBAL DATA:
// GPU control on the current MPI process:
int gpu_up[MAX_GPUS_PER_NODE] = { GPU_OFF }; //GPU_OFF(0): GPU is disabled; GPU_MINE(1): GPU is enabled; GPU_MINE_CUBLAS(2): GPU is BLAS enabled
cudaDeviceProp gpu_prop[MAX_GPUS_PER_NODE]; //properties of all GPUs present on the node
talsh_stats_t gpu_stats[MAX_GPUS_PER_NODE]; //runtime statistics for all GPUs present on the node
#ifndef NO_BLAS
// Infrastructure for CUBLAS:
cublasHandle_t cublas_handle[MAX_GPUS_PER_NODE]; //each GPU present on a node obtains its own cuBLAS context handle
#endif /*NO_BLAS*/
#ifdef USE_CUTENSOR
// Infrastructure for cuTensor:
cutensorHandle_t cutensor_handle[MAX_GPUS_PER_NODE];   //each GPU present on a node obtains its own cuTensor context handle
void* cutensor_workspace[MAX_GPUS_PER_NODE] = { NULL }; //cuTensor workspace (in GPU memory)
size_t cutensor_worksize[MAX_GPUS_PER_NODE] = { 0 };     //cuTensor workspace size
const size_t CUTENSOR_WORKSPACE_SIZE = 128 * 1048576;  //default cuTensor workspace size
#endif /*USE_CUTENSOR*/
// Slabs for the GPU asynchronous resources:
//  CUDA stream handles:
cudaStream_t CUDAStreamBank[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //pre-allocated CUDA stream handles (for each CUDA device)
int CUDAStreamFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_TASKS]; //free CUDA stream handles
int CUDAStreamFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAStreamFreeHandle
//  CUDA event handles:
cudaEvent_t CUDAEventBank[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //pre-allocated CUDA event handles (for each CUDA device)
int CUDAEventFreeHandle[MAX_GPUS_PER_NODE][MAX_CUDA_EVENTS]; //free CUDA event handles
int CUDAEventFFE[MAX_GPUS_PER_NODE]; //number of free handles left in CUDAEventFreeHandle
// Mapped slab of tensor operation prefactors for GPU usage:
slab_t prefactors;         //mapped slab of prefactors
void* gpu_prefs_base_ptr; //mapped device pointer of the slab base
// Global CUDA event recording policy:
int PRINT_TIMING = 1; //non-zero value enables time printing statements
// Infrastructure for function <gpu_tensor_block_copy_dlf> (blocking and non-blocking):
#ifdef USE_CUTT
int TRANS_SHMEM = EFF_TRN_ON_CUTT; //switch between shared-memory tensor transpose and scatter tensor transpose
#else
int TRANS_SHMEM = EFF_TRN_ON; //switch between shared-memory tensor transpose and scatter tensor transpose
#endif /*USE_CUTT*/
// Infrastructure for <gpu_tensor_block_contract_dlf> (non-blocking):
#ifndef NO_BLAS
int DISABLE_BLAS = 0; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#else
int DISABLE_BLAS = 1; //non-zero value will disable cuBLAS usage (if it had been cuBLAS compiled/linked)
#endif /*NO_BLAS*/
cudaTask_t* LastTask[MAX_GPUS_PER_NODE]; //last CUDA task successfully scheduled on each GPU
float h_sgemm_beta_one = 1.0f;
float h_sgemm_beta_zero = 0.0f;
double h_dgemm_beta_one = 1.0;
double h_dgemm_beta_zero = 0.0;
cuComplex h_cgemm_beta_one = { 1.0f,0.0f };
cuComplex h_cgemm_beta_zero = { 0.0f,0.0f };
cuDoubleComplex h_zgemm_beta_one = { 1.0,0.0 };
cuDoubleComplex h_zgemm_beta_zero = { 0.0,0.0 };
#endif /*NO_GPU*/

