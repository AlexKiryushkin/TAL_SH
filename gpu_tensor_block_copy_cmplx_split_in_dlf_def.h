#pragma once

#ifndef NO_GPU

// TENSOR TRANSPOSE (shared-memory version):
template <typename T>
__global__ void gpu_tensor_block_copy_cmplx_split_in_dlf__(int dmo, int drc, int dim_num, int const_args_pos,
  const T* __restrict__ tens_in, T* __restrict__ tens_out)
  /**
  Shared-memory version of tensor transpose: tens_out=TRN(tens_in): Complex arguments only:
  INPUT:
   # dmo - dimension extents order (0: normal, as it is in <const_args>; not 0: permuted dimension order will be imposed);
   # drc - index permutation direction (0: normal, as it is in <const_args>; not 0: inversed permutation will be used);
   # dim_num - tensor block rank;
   # const_args_pos - entry in the __constant__ memory bank where tensor block dimension extents (const_args_dims)
                      and index permutation (const_args_prmn) are stored;
   # tens_in[0:] - complex input tensor in split representation;
  OUTPUT:
   # tens_out[0:] - complex output (transposed) tensor in normal representation;
  NOTES:
   # Minimal CUDA execution configuration is <<<1,warpSize>>>
   # Number of threads per block must be multiple of the warpSize!
  **/
{
  __shared__ T buf0[TENS_TRANSP_BUF_SIZE];
  __shared__ float val;
  __shared__ size_t base_in[MAX_TENSOR_RANK], base_out[MAX_TENSOR_RANK];
  __shared__ size_t ftb[TENS_TRANSP_TAB_SIZE], gtb[TENS_TRANSP_TAB_SIZE];
  __shared__ int htb[TENS_TRANSP_TAB_SIZE], stb[TENS_TRANSP_TAB_SIZE];
  __shared__ int dim_in[MAX_TENSOR_RANK], dim_out[MAX_TENSOR_RANK], o2n[MAX_TENSOR_RANK], n2o[MAX_TENSOR_RANK];
  __shared__ int pri[MAX_TENSOR_RANK], tmp0[MAX_TENSOR_RANK];
  __shared__ int err_code, minor, minor_in, minor_out, s1_ind, s1_ond, s1_step, s1_dim, s2_ind, s2_ond, s2_step, s2_dim, ns1, ns2;
  __shared__ size_t vol, vol_ext;
  size_t _vol, _addr_in, _addr_out, _addr, _work_piece;
  int i, j, k, l, m, n, _vol_minor, _vol_in, _vol_out, _s1, _s2;
  /*
  SHARED MEMORY USE (bytes) =
   + TENS_TRANSP_BUF_SIZE*sizeof(T)
   + MAX_TENSOR_RANK*(8+8+4+4+4+4+4+4)
   + TENS_TRANSP_TAB_SIZE*(8+8+4+4)
   + 4*15 + 8*2
  MIN REGISTER USE (bytes) per thread =
   + 4*4 + 4*11 + 8*5 = 100
  */

  static_assert(ComplexType<T>::valid, "Non-complex types are not allowed!");
  typename ComplexType<T>::RealType* tens_in_real = (typename ComplexType<T>::RealType*)tens_in;
  //Determine the minor index set (only the master thread in each thread block):
  if (threadIdx.x == 0) {
    err_code = 0;
    if (dim_num >= 0 && dim_num <= MAX_TENSOR_RANK && blockDim.x >= warpSize && blockDim.x % warpSize == 0) {
      s1_ind = dim_num + 1; s2_ind = dim_num - 1;
      _vol = 1; for (i = 0; i < dim_num; i++) {
        _vol *= const_args_dims[const_args_pos][i]; if (const_args_prmn[const_args_pos][i] != i + 1) s1_ind = 0;
      }; vol = _vol; //total volume (number of tensor elements)
      if (s1_ind == 0) { //non-trivial permutation
   // Set input/output permutations and dimension extents:
        if (drc == 0) { //normal index permutation
          for (i = 0; i < dim_num; i++) o2n[i] = const_args_prmn[const_args_pos][i] - 1; for (i = 0; i < dim_num; i++) n2o[o2n[i]] = i;
        }
        else { //inversed index permutation
          for (i = 0; i < dim_num; i++) n2o[i] = const_args_prmn[const_args_pos][i] - 1; for (i = 0; i < dim_num; i++) o2n[n2o[i]] = i;
        }
        if (dmo == 0) { //normal dimension order
          for (i = 0; i < dim_num; i++) dim_in[i] = const_args_dims[const_args_pos][i];
          for (i = 0; i < dim_num; i++) dim_out[o2n[i]] = dim_in[i];
        }
        else { //inversed dimension order
          for (i = 0; i < dim_num; i++) dim_out[i] = const_args_dims[const_args_pos][i];
          for (i = 0; i < dim_num; i++) dim_in[n2o[i]] = dim_out[i];
        }
        s1_step = dim_in[s1_ind]; s2_step = dim_in[s2_ind];
        if (_vol > TENS_TRANSP_BUF_SIZE) { //tensor block does not fit into the shared memory buffer
    // Determine the input/output minor index sets and the combined minor index set:
          l = (int)(sqrt((float)TENS_TRANSP_BUF_SIZE));
          minor_in = 0; _vol_in = 1; for (i = 0; i < dim_num; i++) { j = _vol_in * dim_in[i]; if (j > l) break; minor_in++; _vol_in = j; }
          minor_out = 0; _vol_out = 1; for (i = 0; i < dim_num; i++) { j = _vol_out * dim_out[i]; if (j > l) break; minor_out++; _vol_out = j; }
          minor = minor_in; _vol_minor = _vol_in; for (i = 0; i < minor_out; i++) { if (n2o[i] >= minor_in) { minor++; _vol_minor *= dim_out[i]; } }
          m = 1; _s1 = 0; _s2 = 0;
          while (_vol_minor < TENS_TRANSP_BUF_SIZE && m != 0) {
            m = 0;
            if (_s1 == 0) { for (i = minor_in; i < dim_num; i++) { if (o2n[i] < minor_out) { minor_in++; _vol_in *= dim_in[i]; } else { break; } } }
            if (_s2 == 0) { for (i = minor_out; i < dim_num; i++) { if (n2o[i] < minor_in) { minor_out++; _vol_out *= dim_out[i]; } else { break; } } }
            j = dim_in[minor_in]; l = dim_out[minor_out];
            if (minor_in == n2o[minor_out] && _s1 + _s2 == 0) { //same candidate index to both the input and output index sets
              if (j > 1 && TENS_TRANSP_BUF_SIZE < _vol_minor * 2) break;
              if (_vol_minor * j > TENS_TRANSP_BUF_SIZE) { s1_ind = minor_in; s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor; _s1++; _s2++; }
              minor_in++; _vol_in *= j; minor_out++; _vol_out *= j; minor++; _vol_minor *= j; m++;
            }
            else { //the input and output index sets consider two different candidates
              if (_vol_minor * j * l <= TENS_TRANSP_BUF_SIZE && _s1 + _s2 == 0) { //accept both, no splitting
                minor_in++; _vol_in *= j; minor_out++; _vol_out *= l; minor += 2; _vol_minor *= (j * l); m++;
              }
              else { //try to accept either one of the two OR both with splitting
                if (j == 1 || l == 1) {
                  if (j == 1 && _s1 == 0) { minor_in++; minor++; m++; }
                  if (l == 1 && _s2 == 0) { minor_out++; minor++; m++; }
                }
                else {
                  if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE && _vol_minor * l > TENS_TRANSP_BUF_SIZE &&
                    _vol_out >= warpSize && _s1 == 0) { //accept the input index, no splitting
                    minor_in++; _vol_in *= j; minor++; _vol_minor *= j; m++;
                  }
                  else if (_vol_minor * j > TENS_TRANSP_BUF_SIZE && _vol_minor * l <= TENS_TRANSP_BUF_SIZE &&
                    _vol_in >= warpSize && _s2 == 0) { //accept the output index, no splitting
                    minor_out++; _vol_out *= l; minor++; _vol_minor *= l; m++;
                  }
                  else { //splitting is unavoidable (both OR one OR none)
                    if (TENS_TRANSP_BUF_SIZE >= _vol_minor * 2) {
                      if (j >= 4 && l >= 4) { //dimension extents are large enough to be split
                        if (_vol_minor * 4 > TENS_TRANSP_BUF_SIZE) { //impossible to split both indices
                          if (_vol_in <= _vol_out && _s1 == 0) { //split the input candidate index
                            s1_ind = minor_in; s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                            minor_in++; _vol_in *= j; minor++; _vol_minor *= j; _s1++; m++;
                          }
                          else { //split the output candidate index
                            if (_s2 == 0) {
                              s1_ind = n2o[minor_out]; s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                              minor_out++; _vol_out *= l; minor++; _vol_minor *= l; _s2++; m++;
                            }
                          }
                        }
                        else { //possible to split both indices
                          i = (int)sqrt(((float)TENS_TRANSP_BUF_SIZE) / (float)_vol_minor); if (i < 2) i = 2; //uniform splitting
                          s1_step = i; s2_step = i; val = (float)_vol_out / (float)_vol_in;
                          if (val < 1.0f) { //scale the initial uniform splitting to reflect the disbalance between _vol_in and _vol_out
                            if (val * (float)i < 1.0f) val = 1.0f / (float)i; if (val * (float)l < (float)i) val = (float)i / (float)l;
                          }
                          else {
                            if (val * (float)i > (float)j) val = (float)j / (float)i; if (val > float(i)) val = (float)i;
                          }
                          s1_step = (int)(((float)i) * val); s2_step = (int)(((float)i) / val);
                          if (s1_step >= 2 && _s1 == 0) { //&& s1_step <= dim_in[minor_in]
                            s1_ind = minor_in; minor_in++; _vol_in *= j; minor++; _vol_minor *= j; _s1++; m++;
                          }
                          else {
                            s1_step = dim_in[s1_ind];
                          }
                          if (s2_step >= 2 && _s2 == 0) { //&& s2_step <= dim_out[minor_out]
                            s2_ind = n2o[minor_out]; minor_out++; _vol_out *= l; minor++; _vol_minor *= l; _s2++; m++;
                          }
                          else {
                            s2_step = dim_in[s2_ind];
                          }
                        }
                      }
                      else if (j >= 4 && l < 4 && _s1 == 0) { //split the input candidate index
                        s1_ind = minor_in; s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                        minor_in++; _vol_in *= j; minor++; _vol_minor *= j; _s1++; m++;
                      }
                      else if (j < 4 && l >= 4 && _s2 == 0) { //split the output candidate index
                        s1_ind = n2o[minor_out]; s1_step = TENS_TRANSP_BUF_SIZE / _vol_minor;
                        minor_out++; _vol_out *= l; minor++; _vol_minor *= l; _s2++; m++;
                      }
                      else { //both candidate indices have too small extent to be split: try to add one of them fully
                        if (_vol_minor * j <= TENS_TRANSP_BUF_SIZE && _s1 == 0) {
                          minor_in++; _vol_in *= j; minor++; _vol_minor *= j; m++;
                        }
                        else if (_vol_minor * l <= TENS_TRANSP_BUF_SIZE && _s2 == 0) {
                          minor_out++; _vol_out *= l; minor++; _vol_minor *= l; m++;
                        }
                      }
                    }
                    else { //unable to add more indices in the minor set
                      break;
                    }
                  }
                }
              }
            }
          }
          if (s1_ind == dim_num - 1 && s2_ind == dim_num - 1) { s2_ind = 0; s2_step = dim_in[0]; } //s1_ind was set while s2_ind was not
        }
        else { //tensor block fits into the shared memory buffer from the beginning
          minor = dim_num; minor_in = dim_num; minor_out = dim_num; _vol_minor = _vol; _vol_in = _vol; _vol_out = _vol;
        }
        // Share the tensor transpose configuration with other threads in each block:
        vol_ext = _vol / _vol_minor; s1_dim = dim_in[s1_ind]; s2_dim = dim_in[s2_ind];
        // Set indexing bases (OUT:{out,in_c,ext_in}_new; IN:{in,out_c,ext_in}_old):
        //  OUTPUT indexing (dim_out[], base_out[]: prioritized new numeration):
        for (i = 0; i < dim_num; i++) { tmp0[i] = dim_out[i]; } //save output dimension extents (new numeration)
        j = 0; for (i = 0; i < minor_out; i++) { pri[j++] = i; } //output minor index set (new numeration))
        for (i = 0; i < dim_num; i++) { if (o2n[i] >= minor_out) pri[j++] = o2n[i]; } //{compl.input minor + external} index set (new numeration)
        j = 1; for (i = 0; i < dim_num; i++) { dim_out[i] = j; j *= tmp0[i]; } //output bases (new numeration)
        for (i = 0; i < dim_num; i++) { base_out[i] = dim_out[pri[i]]; } //output bases (prioritized new numeration)
        for (i = 0; i < dim_num; i++) { dim_out[i] = tmp0[pri[i]]; } //output extents (prioritized new numeration)
        for (i = 0; i < dim_num; i++) { if (n2o[pri[i]] == s1_ind) { s1_ond = i; } else if (n2o[pri[i]] == s2_ind) { s2_ond = i; } } //split indices (prioritized new numeration)
    //  INPUT indexing (dim_in[], base_in[]: prioritized old numeration):
        for (i = 0; i < dim_num; i++) { tmp0[i] = dim_in[i]; } //save input dimension extents (old numeration)
        j = 0; for (i = 0; i < minor_in; i++) { pri[j++] = i; } //input minor index set (old numeration)
        for (i = 0; i < minor_out; i++) { if (n2o[i] >= minor_in) pri[j++] = n2o[i]; } //compl.output minor idex set (old numeration)
        for (i = j; i < dim_num; i++) { pri[i] = n2o[pri[i]]; } //external index set (just convert new numbers to old ones for consistency)
        j = 1; for (i = 0; i < dim_num; i++) { dim_in[i] = j; j *= tmp0[i]; } //input bases (old numeration)
        for (i = 0; i < dim_num; i++) { base_in[i] = dim_in[pri[i]]; } //input bases (prioritized old numeration)
        for (i = 0; i < dim_num; i++) { dim_in[i] = tmp0[pri[i]]; } //input extents (prioritized old numeration)
        for (i = 0; i < dim_num; i++) { if (pri[i] == s1_ind) { _s1 = i; } else if (pri[i] == s2_ind) { _s2 = i; } } //split indices (prioritized old numeration)
        s1_ind = _s1; s2_ind = _s2;
        ns1 = 1 + (s1_dim - 1) / s1_step; //number of segments from the 1st split minor index
        ns2 = 1 + (s2_dim - 1) / s2_step; //number of segments from the 2nd split minor index
    //  Index position correspondence for the minor index set (pri-new --> pri-old):
        j = 0; for (i = 0; i < minor_out; i++) { if (n2o[i] < minor_in) { pri[i] = n2o[i]; } else { pri[i] = (minor_in + j); j++; } }
        j = 0; for (i = 0; i < minor_in; i++) { if (o2n[i] < minor_out) { pri[o2n[i]] = i; } else { pri[minor_out + j] = i; j++; } }
        // Check tensor transpose configuration parameters:
        if (minor <= 0 || minor_in <= 0 || minor_out <= 0 || _vol <= 0 || _vol_minor <= 0) err_code += 5000; //trap
        if (s1_ind >= dim_num || s2_ind >= dim_num || s1_ond >= dim_num || s2_ond >= dim_num ||
          s1_ind == s2_ind || s1_ond == s2_ond || s1_step <= 0 || s2_step <= 0) err_code += 1000; //trap
        if ((s1_step != dim_in[s1_ind] && s1_ind != minor_in - 1 && s1_ond != minor_out - 1) ||
          (s2_step != dim_in[s2_ind] && s2_ind != minor_in - 1 && s2_ond != minor_out - 1)) err_code += 500; //trap
        if ((_vol_minor * s1_step * s2_step) / (s1_dim * s2_dim) > TENS_TRANSP_BUF_SIZE) err_code += 100; //trap
      } //endif: non-trivial permutation
    }
    else {
      err_code = 1 + 2 * blockDim.x % warpSize;
    }
  } //endif: Master thread.
#ifdef DEBUG_GPU
//DEBUG RECORD begin:
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    j = 0; gpu_debug_dump[j++] = dim_num;
    for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = const_args_dims[const_args_pos][i];
    for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = const_args_prmn[const_args_pos][i];
    for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = base_in[i];
    for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = base_out[i];
    gpu_debug_dump[j++] = vol; gpu_debug_dump[j++] = vol_ext; gpu_debug_dump[j++] = vol / vol_ext;
    gpu_debug_dump[j++] = minor; gpu_debug_dump[j++] = minor_in; gpu_debug_dump[j++] = minor_out;
    gpu_debug_dump[j++] = s1_ind; gpu_debug_dump[j++] = s1_ond; gpu_debug_dump[j++] = s1_step; gpu_debug_dump[j++] = s1_dim;
    gpu_debug_dump[j++] = s2_ind; gpu_debug_dump[j++] = s2_ond; gpu_debug_dump[j++] = s2_step; gpu_debug_dump[j++] = s2_dim;
    for (i = 0; i < dim_num; i++) gpu_debug_dump[j++] = pri[i];
    gpu_debug_dump[j++] = err_code; gpu_debug_dump[j++] = -1;
  }
  //DEBUG RECORD end.
#endif /*DEBUG_GPU*/
  __syncthreads();

  //Proceed:
  if (err_code == 0) {
    if (s1_ind > dim_num) { //tag of a trivial permutation
  // Direct copy:
      _vol = vol; j = gridDim.x * blockDim.x; i = blockIdx.x * blockDim.x + threadIdx.x; _addr_in = _vol - _vol % j;
      for (_addr = 0; _addr < _addr_in; _addr += j) {
        _addr_out = _addr + i;
        auto real_part = tens_in_real[_addr_out];
        auto imag_part = tens_in_real[_addr_out + _vol];
        tens_out[_addr_out] = T{ real_part,imag_part };
      }
      _addr_out = _addr_in + i;
      if (_addr_out < _vol) {
        auto real_part = tens_in_real[_addr_out];
        auto imag_part = tens_in_real[_addr_out + _vol];
        tens_out[_addr_out] = T{ real_part,imag_part };
      }
    }
    else { //non-trivial permutation
      l = threadIdx.x / warpSize; //l: warp number
   // Distribute work accross CUDA blocks (external multi-index + splitting):
      for (_work_piece = blockIdx.x; _work_piece < vol_ext * ns1 * ns2; _work_piece += gridDim.x) { //(ns1*ns2*vol_ext) is the total number of independent tasks
        _addr = _work_piece; _addr /= vol_ext; _vol = _work_piece - _addr * vol_ext; _s2 = (int)(_addr / ns1); _s1 = (int)(_addr - _s2 * ns1); //{_addr_ext,_s1,_s2} --> tensor subblock (CUDA block)
    //  Modify dimension extents due to possible dimension splitting:
        if (threadIdx.x == 0) {
          if (_s1 + 1 == ns1) { //last segment of the 1st split index
            j = s1_dim - _s1 * s1_step; dim_in[s1_ind] = j; dim_out[s1_ond] = j;
          }
          else { //internal segment of the 1st split index
            dim_in[s1_ind] = s1_step; dim_out[s1_ond] = s1_step;
          }
          if (_s2 + 1 == ns2) { //last segment of the 2nd split index
            j = s2_dim - _s2 * s2_step; dim_in[s2_ind] = j; dim_out[s2_ond] = j;
          }
          else { //internal segment of the 2nd split index
            dim_in[s2_ind] = s2_step; dim_out[s2_ond] = s2_step;
          }
          j = 1; for (i = 0; i < minor; i++) { tmp0[i] = j; j *= dim_in[i]; } //minor buffer bases (pri-old)
          for (i = 0; i < minor; i++) n2o[i] = tmp0[pri[i]]; //look up table to accelerate further accesses to tmp0[]
        }
        __syncthreads();
        //  Mount input/output volumes and bases:
        _vol_in = dim_in[0]; for (i = 1; i < minor_in; i++) { _vol_in *= dim_in[i]; }
        _vol_out = dim_out[0]; for (i = 1; i < minor_out; i++) { _vol_out *= dim_out[i]; }
        _vol_minor = _vol_out; for (i = minor_out; i < minor; i++) { _vol_minor *= dim_out[i]; }
        _addr_in = (_s1 * s1_step) * base_in[s1_ind] + (_s2 * s2_step) * base_in[s2_ind]; _addr_out = _vol;
        for (i = minor; i < dim_num; i++) { _addr = _vol / dim_in[i]; _addr_in += (_vol - _addr * dim_in[i]) * base_in[i]; _vol = _addr; }
        _vol = _addr_out; _addr_out = (_s1 * s1_step) * base_out[s1_ond] + (_s2 * s2_step) * base_out[s2_ond];
        for (i = minor; i < dim_num; i++) { _addr = _vol / dim_out[i]; _addr_out += (_vol - _addr * dim_out[i]) * base_out[i]; _vol = _addr; }
        if (_vol_out > TENS_TRANSP_TAB_SIZE || _vol_minor > _vol_in * TENS_TRANSP_TAB_SIZE ||
          _vol_minor > _vol_out * TENS_TRANSP_TAB_SIZE) {
          //  Algorithm 0 (slower):
          //   Read the minor volume into the buffer from the input tensor block:
          _vol_minor /= _vol_in; //vol_in_c
          _s1 = 1 + (_vol_in - 1) / warpSize; //number of warps (lines) which fully cover the input volume
          _s2 = blockDim.x / warpSize; //number of whole warps in a thread block (each warp treats one line)
          for (j = l; j < _s1 * _vol_minor; j += _s2) { //j: Line number
            m = j / _s1; _addr = _addr_in; n = m; //n: Input column number (in_c)
            for (i = minor_in; i < minor; i++) { k = m / dim_in[i]; _addr += (m - k * dim_in[i]) * base_in[i]; m = k; }
            //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in the input volume
            m = threadIdx.x + (j - n * _s1 - l) * warpSize; //elemental offset in the input volume (alternative)
            if (m < _vol_in) {
              auto real_part = tens_in_real[_addr + m];
              auto imag_part = tens_in_real[_addr + m + _vol];
              buf0[n * _vol_in + m] = T{ real_part,imag_part };
            }
          }
          __syncthreads();
          //   Write the minor volume from the buffer into the output tensor block:
          _vol_minor = (_vol_minor * _vol_in) / _vol_out; //vol_out_c
          _s1 = 1 + (_vol_out - 1) / warpSize; //number of warps (lines) which fully cover the output volume
          for (j = l; j < _s1 * _vol_minor; j += _s2) { //j: Line number
            n = j / _s1; _addr = _addr_out; _vol = n; _vol_in = 0; //_vol: Output column number (out_c)
      //    for(i=minor_out;i<minor;i++){m=n%dim_out[i]; n/=dim_out[i]; _addr+=m*base_out[i]; _vol_in+=m*tmp0[pri[i]];}
            for (i = minor_out; i < minor; i++) { k = n / dim_out[i]; m = n - k * dim_out[i]; n = k; _addr += m * base_out[i]; _vol_in += m * n2o[i]; }
            //    m=(j%_s1)*warpSize+threadIdx.x%warpSize; //elemental offset in the output volume
            m = threadIdx.x + (j - (int)_vol * _s1 - l) * warpSize; //elemental offset in the output volume (alternative)
            if (m < _vol_out) {
              _addr += m;
              //     for(i=0;i<minor_out;i++){_vol_in+=(m%dim_out[i])*tmp0[pri[i]]; m/=dim_out[i];}
              for (i = 0; i < minor_out; i++) { k = m / dim_out[i]; _vol_in += (m - k * dim_out[i]) * n2o[i]; m = k; }
              tens_out[_addr] = buf0[_vol_in];
            }
          }
          __syncthreads();
        }
        else {
          //  Algorithm 1 (presumably faster):
          //   Create per-block look-up tables:
          m = _vol_minor / _vol_in; //vol_in_c
          for (j = threadIdx.x; j < m; j += blockDim.x) { //column number (input)
            _addr = 0; _s1 = j;
            //    for(i=minor_in;i<minor;i++){_addr+=(_s1%dim_in[i])*base_in[i]; _s1/=dim_in[i];}
            for (i = minor_in; i < minor; i++) { _s2 = _s1 / dim_in[i]; _addr += (_s1 - _s2 * dim_in[i]) * base_in[i]; _s1 = _s2; }
            ftb[j] = _addr;
          }
          m = _vol_minor / _vol_out; //vol_out_c
          for (j = threadIdx.x; j < m; j += blockDim.x) { //column number (output)
            _addr = 0; _s1 = j;
            //    for(i=minor_out;i<minor;i++){_addr+=(_s1%dim_out[i])*base_out[i]; _s1/=dim_out[i];}
            for (i = minor_out; i < minor; i++) { _s2 = _s1 / dim_out[i]; _addr += (_s1 - _s2 * dim_out[i]) * base_out[i]; _s1 = _s2; }
            gtb[j] = _addr;
          }
          for (j = threadIdx.x; j < m; j += blockDim.x) { //column number (output)
            n = 0; _s1 = j;
            //    for(i=minor_out;i<minor;i++){n+=(_s1%dim_out[i])*n2o[i]; _s1/=dim_out[i];}
            for (i = minor_out; i < minor; i++) { _s2 = _s1 / dim_out[i]; n += (_s1 - _s2 * dim_out[i]) * n2o[i]; _s1 = _s2; }
            htb[j] = n;
          }
          for (j = threadIdx.x; j < _vol_out; j += blockDim.x) {
            n = 0; _s1 = j;
            //    for(i=0;i<minor_out;i++){n+=(_s1%dim_out[i])*n2o[i]; _s1/=dim_out[i];}
            for (i = 0; i < minor_out; i++) { _s2 = _s1 / dim_out[i]; n += (_s1 - _s2 * dim_out[i]) * n2o[i]; _s1 = _s2; }
            stb[j] = n;
          }
          __syncthreads();
          //   Read the minor volume into the buffer from the input tensor block:
          _vol_minor /= _vol_in; //vol_in_c
          _s1 = 1 + (_vol_in - 1) / warpSize; //number of warps (lines) which fully cover the input volume
          _s2 = blockDim.x / warpSize; //number of whole warps in a thread block (each warp treats one line)
          for (j = l; j < _s1 * _vol_minor; j += _s2) { //j: Line number
            m = j / _s1; n = threadIdx.x + (j - m * _s1 - l) * warpSize; //m: Input column number (in_c); n: Offset in the column
            if (n < _vol_in) {
              _addr = _addr_in + ftb[m] + n;
              auto real_part = tens_in_real[_addr];
              auto imag_part = tens_in_real[_addr + _vol];
              buf0[m * _vol_in + n] = T{ real_part,imag_part };
            }
          }
          __syncthreads();
          //   Write the minor volume from the buffer into the output tensor block:
          _vol_minor = (_vol_minor * _vol_in) / _vol_out; //vol_out_c
          _s1 = 1 + (_vol_out - 1) / warpSize; //number of warps (lines) which fully cover the output volume
          for (j = l; j < _s1 * _vol_minor; j += _s2) { //j: Line number
            m = j / _s1; n = threadIdx.x + (j - m * _s1 - l) * warpSize; //m: Output column number (out_c); n: Offset in the column
            if (n < _vol_out) {
              _addr = _addr_out + gtb[m] + n; _vol_in = htb[m] + stb[n];
              tens_out[_addr] = buf0[_vol_in];
            }
          }
          __syncthreads();
        }
      } //enddo _work_piece: independent work distribution among thread blocks
    }
  }
  //Record errors if occured (for each block):
  if (threadIdx.x == 0) { if (err_code != 0) i = atomicAdd(&gpu_error_count, 1); }
  return;
}

#endif
