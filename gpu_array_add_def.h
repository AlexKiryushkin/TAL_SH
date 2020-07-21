#pragma once

#ifndef NO_GPU

// ARRAY ADDITION:
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T* __restrict__ arr0, const T* __restrict__ arr1, T alpha, int left_conj)
/** arr0(0:tsize-1)+=arr1(0:tsize-1)*alpha **/
{
  size_t _ti = blockIdx.x * blockDim.x + threadIdx.x;
  size_t _gd = gridDim.x * blockDim.x;
  for (size_t l = _ti; l < tsize; l += _gd)
  {
    arr0[l] = talshAdd(arr0[l], talshMul(talshConjugate(arr1[l], left_conj), alpha));
  }
}

// ARRAY ADDITION AND SCALING:
template <typename T>
__global__ void gpu_array_add__(size_t tsize, T* __restrict__ arr0, const T* __restrict__ arr1, const T* __restrict__ scalar,
  T alpha, int left_conj)
  /** arr0(0:tsize-1)+=arr1(0:tsize-1)*scalar*alpha **/
{
  size_t _ti = blockIdx.x * blockDim.x + threadIdx.x;
  size_t _gd = gridDim.x * blockDim.x;
  T pref = talshMul(*scalar, alpha);
  for (size_t l = _ti; l < tsize; l += _gd)
  {
    arr0[l] = talshAdd(arr0[l], talshMul(talshConjugate(arr1[l], left_conj), pref));
  }
}

#endif
