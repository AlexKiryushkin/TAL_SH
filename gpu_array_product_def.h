#pragma once

#ifndef NO_GPU

// ARRAY DIRECT PRODUCT:
template <typename T>
__global__ void gpu_array_product__(size_t tsize1, const T* arr1, size_t tsize2, const T* arr2, T* arr0,
  T alpha, int left_conj, int right_conj)
  /** arr0[0:tsize2-1][0:tsize1-1]+=arr1[0:tsize1-1]*arr2[0:tsize2-1]*alpha **/
{
  __shared__ T lbuf[THRDS_ARRAY_PRODUCT + 1];
  __shared__ T rbuf[THRDS_ARRAY_PRODUCT];

  size_t _tx = (size_t)threadIdx.x;
  for (size_t _jb = blockIdx.y * THRDS_ARRAY_PRODUCT; _jb < tsize2; _jb += gridDim.y * THRDS_ARRAY_PRODUCT)
  {
    size_t _jn = ((_jb + THRDS_ARRAY_PRODUCT > tsize2) ? (tsize2 - _jb) : THRDS_ARRAY_PRODUCT);
    if (_tx < _jn)
    {
      rbuf[_tx] = talshConjugate(arr2[_jb + _tx], right_conj) * alpha;
    }

    for (size_t _ib = blockIdx.x * THRDS_ARRAY_PRODUCT; _ib < tsize1; _ib += gridDim.x * THRDS_ARRAY_PRODUCT) 
    {
      size_t _in = ((_ib + THRDS_ARRAY_PRODUCT > tsize1) ? _in = tsize1 - _ib : THRDS_ARRAY_PRODUCT);
      if (_tx < _in)
      {
        lbuf[_tx] = talshConjugate(arr1[_ib + _tx], left_conj);
      }
      __syncthreads();

      for (size_t _jc = 0; _jc < _jn; _jc++)
      {
        if (_tx < _in)
        {
          arr0[(_jb + _jc) * tsize1 + _ib + _tx] = arr0[(_jb + _jc) * tsize1 + _ib + _tx] + lbuf[_tx] * rbuf[_jc];
        }
      }
      __syncthreads();
    }
  }
}

#endif
