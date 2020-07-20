#pragma once

#ifndef NO_GPU

// ARRAY RESCALING:
// REAL:
template <typename T>
__global__ void gpu_array_scale__(size_t tsize, T* arr, T alpha)
/** arr(0:tsize-1)*=alpha **/
{
  size_t _ti = blockIdx.x * blockDim.x + threadIdx.x;
  size_t _gd = gridDim.x * blockDim.x;
  for (size_t l = _ti; l < tsize; l += _gd)
  {
    arr[l] = arr[l] * alpha;
  }
}

#endif

