#pragma once

#ifndef NO_GPU

#include <cuda.h>
#include <cuda_runtime.h>

#endif

#include "tensor_algebra.h"
#include "device_algebra.h"
#include "mem_manager.h"
#include "talsh_complex.h"
#include "kernel_auxiliary_data.h"
#include "talsh_complex_wrapper.h"

#ifndef NO_GPU
// CUDA KERNELS:
template <typename T>
__global__ void gpu_array_init__(size_t tsize, T* arr, T val);
template <typename T>
__global__ void gpu_scalar_multiply__(const T* left_arg, const T* right_arg, T* dest_arg, T alpha,
  int left_conj = 0, int right_conj = 0);
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T* __restrict__ arr0, const T* __restrict__ arr1,
  T alpha, int left_conj = 0);
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T* __restrict__ arr0, const T* __restrict__ arr1, const T* __restrict__ scalar,
  T alpha, int left_conj = 0);
template <typename T>
__global__ void gpu_array_dot_product__(size_t tsize, const T* arr1, const T* arr2, volatile T* dprod,
  T alpha, int left_conj = 0, int right_conj = 0);
template <typename T>
__global__ void gpu_array_product__(size_t tsize1, const T* arr1, size_t tsize2, const T* arr2, T* arr0,
  T alpha, int left_conj = 0, int right_conj = 0);
template <typename T>
__global__ void gpu_tensor_block_add_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out);
template <typename T>
__global__ void gpu_tensor_block_copy_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out);
template <typename T>
__global__ void gpu_tensor_block_copy_cmplx_split_in_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out);
template <typename T>
__global__ void gpu_tensor_block_copy_cmplx_split_out_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out);
template <typename T>
__global__ void gpu_tensor_block_copy_scatter_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out);
template <typename T>
__global__ void gpu_matrix_multiply_tn__(size_t ll, size_t lr, size_t lc, const T* arg1, const T* arg2, T* arg0, T alpha);

#include "gpu_array_add_def.h"
#include "gpu_array_dot_product_def.h"
#include "gpu_array_init_def.h"
#include "gpu_array_product_def.h"
#include "gpu_matrix_multiply_tn_def.h"
#include "gpu_scalar_multiply_def.h"
#include "gpu_tensor_block_add_dlf_def.h"
#include "gpu_tensor_block_copy_cmplx_split_in_dlf_def.h"
#include "gpu_tensor_block_copy_cmplx_split_out_dlf_def.h"
#include "gpu_tensor_block_copy_dlf_def.h"
#include "gpu_tensor_block_copy_scatter_dlf_def.h"

#endif
