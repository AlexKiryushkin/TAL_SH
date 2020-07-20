#pragma once

#ifndef NO_GPU

// SCALAR MULTIPLICATION:
template <typename T>
__global__ void gpu_scalar_multiply__(const T* left_arg, const T* right_arg, T* dest_arg, T alpha,
  int left_conj, int right_conj)
  /** Scalar += Scalar * Scalar * Alpha **/
{
  if (blockIdx.x == 0 && threadIdx.x == 0) 
  {
    *dest_arg = (*dest_arg) +
      alpha * talshConjugate(*left_arg, left_conj) * talshConjugate(*right_arg, right_conj);
  }
}

#endif
