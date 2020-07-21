#pragma once

#ifndef NO_GPU

template <class T, std::enable_if_t<RealType<T>::valid, int8_t> = 0>
__device__ void atomicIncrement(T * address, T increment)
{
  atomicAdd(address, increment);
}

template <class T, std::enable_if_t<ComplexType<T>::valid, int8_t> = 0>
__device__ void atomicIncrement(T * address, T increment)
{
  atomicAdd(&address->x, increment.x);
  atomicAdd(&address->y, increment.y);
}

template <typename T>
__global__ void gpu_array_dot_product__(size_t tsize, const T* arr1, const T* arr2, volatile T* dprod,
  T alpha, int left_conj, int right_conj)
  /** Scalar (GPU) += arr1(0:tsize-1) * arr2(0:tsize-1) * alpha **/
{
  extern __shared__ char sh_buf[];
  T * dprs = (T*)(&sh_buf[0]);
  T dpr{};

  for (size_t l = blockIdx.x * blockDim.x + threadIdx.x; l < tsize; l += gridDim.x * blockDim.x)
  {
    dpr = talshAdd(dpr, talshMul(talshConjugate(arr1[l], left_conj), talshConjugate(arr2[l], right_conj)));
  }
  dprs[threadIdx.x] = talshMul(dpr, alpha);
  __syncthreads();

  unsigned int s = blockDim.x;
  while (s > 1) 
  {
    unsigned int j = (s + 1U) >> 1;
    if (threadIdx.x + j < s)
    {
      dprs[threadIdx.x] = talshAdd(dprs[threadIdx.x], dprs[threadIdx.x + j]);
    }
    __syncthreads();
    s = j;
  }

  if (threadIdx.x == 0) 
  {
    int i = 1;
    while (i != 0)
    {
      i = atomicMax(&dot_product_wr_lock, 1);
      if (i == 0)
      {
        atomicIncrement(const_cast<T *>(dprod), dprs[0]);
      }
    }

    i = atomicExch(&dot_product_wr_lock, 0);
  }
  __syncthreads();
}

#endif
