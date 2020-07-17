#pragma once

#ifndef NO_GPU

// SCALAR MULTIPLICATION:
// REAL:
template <typename T>
__global__ void gpu_scalar_multiply__(const T* left_arg, const T* right_arg, T* dest_arg, T alpha,
  int left_conj, int right_conj)
  /** Scalar += Scalar * Scalar * Alpha **/
{
  if (blockIdx.x == 0 && threadIdx.x == 0) 
  {
    *dest_arg += (*left_arg) * (*right_arg) * alpha;
  }
}
// COMPLEX4:
template <>
__global__ void gpu_scalar_multiply__<talshComplex4>(const talshComplex4* left_arg, const talshComplex4* right_arg,
  talshComplex4* dest_arg, talshComplex4 alpha,
  int left_conj, int right_conj)
  /** Scalar += Scalar * Scalar * Alpha **/
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (left_conj != 0) {
      if (right_conj != 0) {
        *dest_arg = talshComplex4Add(*dest_arg, talshComplex4Mul(talshComplex4Mul(talshComplex4Conjg(*left_arg), talshComplex4Conjg(*right_arg)), alpha));
      }
      else {
        *dest_arg = talshComplex4Add(*dest_arg, talshComplex4Mul(talshComplex4Mul(talshComplex4Conjg(*left_arg), *right_arg), alpha));
      }
    }
    else {
      if (right_conj != 0) {
        *dest_arg = talshComplex4Add(*dest_arg, talshComplex4Mul(talshComplex4Mul(*left_arg, talshComplex4Conjg(*right_arg)), alpha));
      }
      else {
        *dest_arg = talshComplex4Add(*dest_arg, talshComplex4Mul(talshComplex4Mul(*left_arg, *right_arg), alpha));
      }
    }
  }
}
// COMPLEX8:
template <>
__global__ void gpu_scalar_multiply__<talshComplex8>(const talshComplex8* left_arg, const talshComplex8* right_arg,
  talshComplex8* dest_arg, talshComplex8 alpha,
  int left_conj, int right_conj)
  /** Scalar += Scalar * Scalar * Alpha **/
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (left_conj != 0) {
      if (right_conj != 0) {
        *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(talshComplex8Conjg(*left_arg), talshComplex8Conjg(*right_arg)), alpha));
      }
      else {
        *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(talshComplex8Conjg(*left_arg), *right_arg), alpha));
      }
    }
    else {
      if (right_conj != 0) {
        *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(*left_arg, talshComplex8Conjg(*right_arg)), alpha));
      }
      else {
        *dest_arg = talshComplex8Add(*dest_arg, talshComplex8Mul(talshComplex8Mul(*left_arg, *right_arg), alpha));
      }
    }
  }
}

#endif
