#pragma once

#include "talsh_complex.h"

#ifndef NO_GPU
#define TALSH_HOST_DEVICE __host__ __device__
#else
#define TALSH_HOST_DEVICE
#endif

#ifndef NO_GPU
  #define TALSH_INLINE __forceinline__
#else
  #ifdef __cplusplus
    #define TALSH_INLINE inline
  #else
    #define TALSH_INLINE
  #endif
#endif

/**
 * Set complex function
 */
TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 talshComplexSet(float real, float imag)
{
  return talshComplex4Set(real, imag);
}
TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 talshComplexSet(double real, double imag)
{
  return talshComplex8Set(real, imag);
}

/*
 * Conjugate complex function
 */
TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 talshComplexConjg(talshComplex4 cmplx)
{
  return talshComplex4Conjg(cmplx);
}
TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 talshComplexConjg(talshComplex8 cmplx)
{
  return talshComplex8Conjg(cmplx);
}

/*
 * Absolute complex value function
 */
TALSH_HOST_DEVICE TALSH_INLINE float talshComplexAbs(talshComplex4 cmplx)
{
  return talshComplex4Abs(cmplx);
}
TALSH_HOST_DEVICE TALSH_INLINE double talshComplexAbs(talshComplex8 cmplx)
{
  return talshComplex8Abs(cmplx);
}

/*
 * Absolute squared complex value function
 */
TALSH_HOST_DEVICE TALSH_INLINE float talshComplexAsq(talshComplex4 cmplx)
{
  return talshComplex4Asq(cmplx);
}
TALSH_HOST_DEVICE TALSH_INLINE double talshComplexAsq(talshComplex8 cmplx)
{
  return talshComplex8Asq(cmplx);
}

/*
 * Summation complex values function
 */
TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 talshComplexAdd(talshComplex4 x, talshComplex4 y)
{
  return talshComplex4Add(x, y);
}
TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 talshComplexAdd(talshComplex8 x, talshComplex8 y)
{
  return talshComplex8Add(x, y);
}

/*
 * Subtraction complex values function
 */
TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 talshComplexSub(talshComplex4 x, talshComplex4 y)
{
  return talshComplex4Sub(x, y);
}
TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 talshComplexSub(talshComplex8 x, talshComplex8 y)
{
  return talshComplex8Sub(x, y);
}

/*
 * Multiplication complex values function
 */
TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 talshComplexMul(talshComplex4 x, talshComplex4 y)
{
  return talshComplex4Mul(x, y);
}
TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 talshComplexMul(talshComplex8 x, talshComplex8 y)
{
  return talshComplex8Mul(x, y);
}

/*
 * Division complex values function
 */
TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 talshComplexDiv(talshComplex4 x, talshComplex4 y)
{
  return talshComplex4Div(x, y);
}
TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 talshComplexDiv(talshComplex8 x, talshComplex8 y)
{
  return talshComplex8Div(x, y);
}

template <class T, std::enable_if_t<RealType<T>::valid, int8_t> = 0>
TALSH_HOST_DEVICE TALSH_INLINE T talshConjugate(T value, int)
{
  return value;
}

template <class T, std::enable_if_t<ComplexType<T>::valid, int8_t> = 0>
TALSH_HOST_DEVICE TALSH_INLINE T talshConjugate(T value, int conj)
{
  return (conj == 0) ? value : talshComplexConjg(value);
}

TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 operator*(talshComplex4 lhs, talshComplex4 rhs)
{
  return talshComplexMul(lhs, rhs);
}

TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 operator*(talshComplex8 lhs, talshComplex8 rhs)
{
  return talshComplexMul(lhs, rhs);
}

TALSH_HOST_DEVICE TALSH_INLINE talshComplex4 operator+(talshComplex4 lhs, talshComplex4 rhs)
{
  return talshComplexAdd(lhs, rhs);
}

TALSH_HOST_DEVICE TALSH_INLINE talshComplex8 operator+(talshComplex8 lhs, talshComplex8 rhs)
{
  return talshComplexAdd(lhs, rhs);
}
