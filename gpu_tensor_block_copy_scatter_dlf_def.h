#pragma once

#ifndef NO_GPU

// TENSOR TRANSPOSE (naive scatter version):
template <typename T>
__global__ void gpu_tensor_block_copy_scatter_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out)
  /**
  Scattering version of tensor transpose: tens_out=TRN(tens_in):
  INPUT:
   # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0: permuted dimension order will be imposed);
   # drc - index permutation direction (0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
   # dim_num - tensor block rank;
   # const_args_pos - entry in the __constant__ memory bank where tensor block dimension extents (const_args_dims)
                      and index permutation (const_args_prmn) are stored;
   # tens_in[0:] - input tensor;
  OUTPUT:
   # tens_out[0:] - output (transposed) tensor;
  **/
{
  __shared__ int n2o[MAX_TENSOR_RANK];
  __shared__ size_t vol, base_in[MAX_TENSOR_RANK], base_out[MAX_TENSOR_RANK];
  int i, j, k;
  size_t _vol, _addr_in, _addr_out, _si;

  if (dim_num == 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0) tens_out[0] = tens_in[0];
  }
  else if (dim_num == 1) {
    _vol = const_args_dims[const_args_pos][0];
    j = blockIdx.x * blockDim.x + threadIdx.x;
    for (_addr_in = j; _addr_in < _vol; _addr_in += gridDim.x * blockDim.x) { tens_out[_addr_in] = tens_in[_addr_in]; }
  }
  else if (dim_num > 1) {
    if (threadIdx.x == 0) {
      k = 0; for (i = 0; i < dim_num; i++) { j = const_args_prmn[const_args_pos][i] - 1; n2o[j] = i; if (j != i) k = 1; }
      if (k == 0) { //trivial permutation
        n2o[0] = dim_num; //trivial permutation flag
        _vol = 1; for (i = 0; i < dim_num; i++) { _vol *= const_args_dims[const_args_pos][i]; }; vol = _vol;
      }
      else { //non-trivial permutation
        if (dmo == 0) { //normal dimension order
          _vol = 1; for (i = 0; i < dim_num; i++) { base_in[i] = _vol; _vol *= const_args_dims[const_args_pos][i]; }; vol = _vol;
          if (drc == 0) { //normal index permutation
            _vol = 1; for (i = 0; i < dim_num; i++) { k = n2o[i]; base_out[k] = _vol; _vol *= const_args_dims[const_args_pos][k]; }
          }
          else { //inversed index permutation
            _vol = 1; for (i = 0; i < dim_num; i++) {
              k = const_args_prmn[const_args_pos][i] - 1; base_out[k] = _vol; _vol *= const_args_dims[const_args_pos][k];
            }
          }
        }
        else { //inversed dimension order
          if (drc == 0) { //normal index permutation
            _vol = 1; for (i = 0; i < dim_num; i++) {
              k = const_args_prmn[const_args_pos][i] - 1; base_in[i] = _vol; _vol *= const_args_dims[const_args_pos][k];
            }; vol = _vol;
            _vol = 1; for (i = 0; i < dim_num; i++) { k = n2o[i]; base_out[k] = _vol; _vol *= const_args_dims[const_args_pos][i]; }
          }
          else { //inversed index permutation
            _vol = 1; for (i = 0; i < dim_num; i++) {
              k = n2o[i]; base_in[i] = _vol; _vol *= const_args_dims[const_args_pos][k];
            }; vol = _vol;
            _vol = 1; for (i = 0; i < dim_num; i++) {
              k = const_args_prmn[const_args_pos][i] - 1; base_out[k] = _vol; _vol *= const_args_dims[const_args_pos][i];
            }
          }
        }
      }
    }
#ifdef DEBUG_GPU
    //DEBUG RECORD begin:
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      j = 0; gpu_debug_dump[j++] = dim_num;
      for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = const_args_dims[const_args_pos][i];
      for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = const_args_prmn[const_args_pos][i];
      for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = base_in[i];
      for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = base_out[i];
      gpu_debug_dump[j++] = vol; gpu_debug_dump[j++] = -1;
    }
    //DEBUG RECORD end.
#endif /*DEBUG_GPU*/
    __syncthreads();
    _vol = vol;
    if (n2o[0] >= dim_num) { //trivial permutation
      k = gridDim.x * blockDim.x; j = blockIdx.x * blockDim.x + threadIdx.x;
      for (_addr_in = j; _addr_in < _vol; _addr_in += k) { tens_out[_addr_in] = tens_in[_addr_in]; }
    }
    else { //non-trivial permutation
      j = blockIdx.x * blockDim.x + threadIdx.x;
      for (_addr_in = j; _addr_in < _vol; _addr_in += gridDim.x * blockDim.x) {
        _addr_out = 0; _si = _addr_in; for (i = dim_num - 1; i >= 0; i--) { _addr_out += (_si / base_in[i]) * base_out[i]; _si %= base_in[i]; }
        tens_out[_addr_out] = tens_in[_addr_in];
      }
    }
  }
  else { //dim_num < 0
    if (threadIdx.x == 0) i = atomicAdd(&gpu_error_count, 1); //record an error (for each thread block)
  }
  return;
}

#endif
