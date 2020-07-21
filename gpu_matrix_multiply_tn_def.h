#pragma once

#ifndef NO_GPU

// MATRIX MULTIPLICATION (slow):
template <typename T>
__global__ void gpu_matrix_multiply_tn__(size_t ll, size_t lr, size_t lc, const T* arg1, const T* arg2, T* arg0, T alpha)
/** arg0(0:ll-1,0:lr-1)+=arg1(0:lc-1,0:ll-1)*arg2(0:lc-1,0:lr-1)*alpha
NOTES:
 # Thread block dimensions (.x and .y) must be equal to MAT_MULT_TILE_DIM(X,Y), respectively.
**/
{
  __shared__ T buf1[MAT_MULT_TILE_DIMX + 1][MAT_MULT_TILE_DIMX + 1], buf2[MAT_MULT_TILE_DIMY + 1][MAT_MULT_TILE_DIMX + 1];
  size_t k, _col, _row, _col_base, _row_base;
  int i, j, l, m;
  T _val;

  if (lc > 0 && ll > 0 && lr > 0 && blockDim.x == MAT_MULT_TILE_DIMX && blockDim.y == MAT_MULT_TILE_DIMY) {
    _val = static_cast<T>(0.0); j = threadIdx.y; i = threadIdx.x;
    _col_base = blockIdx.y * MAT_MULT_TILE_DIMY;
    while (_col_base < lr) {
      _row_base = blockIdx.x * MAT_MULT_TILE_DIMX;
      while (_row_base < ll) {
        for (k = 0; k < lc; k += MAT_MULT_TILE_DIMX) {
          _col = _col_base + j; _row = _row_base + j;
          // Load two blocks into shared memory:
          if (k + MAT_MULT_TILE_DIMX > lc) { m = lc - k; }
          else { m = MAT_MULT_TILE_DIMX; }
          if (i < m) { //(k+i)<lc
            for (l = 0; l < MAT_MULT_TILE_DIMX; l += MAT_MULT_TILE_DIMY) {
              if (_row < ll) { buf1[l + j][i] = arg1[_row * lc + (k + i)] * alpha; } // Load a block of the 1st argument into the shared memory
              _row += MAT_MULT_TILE_DIMY;
            }
            if (_col < lr) { buf2[j][i] = arg2[_col * lc + (k + i)]; } // Load a block of the 2nd argument into the shared memory
          }
          __syncthreads();
          // Multiply the two blocks:
          _row = _row_base + i;
          if (_col < lr) {
            if (_row < ll) {
              _col = _col * ll + _row;
              for (l = 0; l < m; l++) { _val += buf1[i][l] * buf2[j][l]; }
              arg0[_col] += _val; _val = static_cast<T>(0.0);
            }
          }
          __syncthreads();
        }
        _row_base += gridDim.x * MAT_MULT_TILE_DIMX;
      }
      _col_base += gridDim.y * MAT_MULT_TILE_DIMY;
    }
  }
  else {
    if (threadIdx.x == 0 && threadIdx.y == 0) i = atomicAdd(&gpu_error_count, 1); //record an error (for each thread block)
  }
  return;
}

#endif
