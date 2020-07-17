#pragma once

#ifndef NO_GPU

// ARRAY INITIALIZATION:
template <typename T>
__global__ void gpu_array_init__(size_t tsize, T* arr, T val)
/** arr(0:tsize-1)=val **/
{
  size_t _ti = blockIdx.x * blockDim.x + threadIdx.x;
  size_t _gd = gridDim.x * blockDim.x;
  for (size_t l = _ti; l < tsize; l += _gd) arr[l] = val;
  return;
}

#endif
