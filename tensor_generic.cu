
#include "tensor_algebra.h"

int tens_valid_data_kind(int datk, int* datk_size)
/** Returns YEP if the data kind <datk> is valid in TAL-SH, NOPE otherwise.
    Optionally, the data kind size can be returned in <datk_size>. **/
{
  int datk_sz = -1;
  int ans = NOPE;
  switch (datk) {
  case R4: ans = YEP; datk_sz = sizeof(float); break;    //real float
  case R8: ans = YEP; datk_sz = sizeof(double); break;   //real double
  case C4: ans = YEP; datk_sz = sizeof(float) * 2; break;  //complex float
  case C8: ans = YEP; datk_sz = sizeof(double) * 2; break; //complex double
  case NO_TYPE: ans = YEP; datk_sz = 0; break; //NO_TYPE is a valid data kind
  }
  if (datk_size != NULL) *datk_size = datk_sz;
  return ans;
}

int tens_valid_data_kind_(int datk, int* datk_size) //Fortran binding
{
  return tens_valid_data_kind(datk, datk_size);
}

int permutation_trivial(const int perm_len, const int* perm, const int base)
{
  int trivial = 1;
  for (int i = 0; i < perm_len; ++i) {
    if (perm[i] != (base + i)) { trivial = 0; break; }
  }
  return trivial;
}

#ifdef USE_CUTENSOR
int get_contr_pattern_cutensor(const int* dig_ptrn,
  int drank, int32_t* ptrn_d,
  int lrank, int32_t* ptrn_l,
  int rrank, int32_t* ptrn_r)
  /** Converts a digital tensor contraction pattern used by TAL-SH into the cuTensor digital format. **/
{
  int errc = 0;
  if (drank >= 0 && lrank >= 0 && rrank >= 0) {
    if (lrank + rrank > 0) {
      if (dig_ptrn != NULL) {
        int ci = drank; //contracted indices will have ids: drank+1,drank+2,drank+3,...
        for (int i = 0; i < drank; ++i) ptrn_d[i] = (i + 1); //dtens[1,2,3,4,...]
        for (int i = 0; i < lrank; ++i) {
          int j = dig_ptrn[i];
          if (j > 0) { //uncontracted index
            ptrn_l[i] = j;
          }
          else if (j < 0) { //contracted index
            ptrn_l[i] = ++ci;
            ptrn_r[-j - 1] = ci;
          }
          else {
            errc = -5;
            break;
          }
        }
        if (errc == 0) {
          for (int i = 0; i < rrank; ++i) {
            int j = dig_ptrn[lrank + i];
            if (j > 0) { //uncontracted index
              ptrn_r[i] = j;
            }
            else if (j < 0) { //contracted index
              if (ptrn_r[i] != ptrn_l[-j - 1]) { //already set
                errc = -4;
                break;
              }
            }
            else {
              errc = -3;
              break;
            }
          }
        }
      }
      else {
        errc = -2;
      }
    }
  }
  else {
    errc = -1;
  }
  return errc;
}
#endif /*USE_CUTENSOR*/

size_t tens_elem_offset_f(unsigned int num_dim, const unsigned int* dims, const unsigned int* mlndx)
/** Returns the offset of a tensor element specified by its multi-index with Fortran storage layout.
    Each index in the multi-index has lower bound of zero. **/
{
  unsigned int i;
  size_t offset;

  offset = 0;
  for (i = num_dim - 1; i > 0; --i) { offset += mlndx[i]; offset *= dims[i - 1]; };
  offset += mlndx[0];
  return offset;
}

void tens_elem_mlndx_f(size_t offset, unsigned int num_dim, const unsigned int* dims, unsigned int* mlndx)
/** Returns the multi-index of a tensor element specified by its offset with Fortran storage layout.
    Each index in the multi-index has lower bound of zero. **/
{
  unsigned int i;
  size_t d;

  for (i = 0; i < num_dim; ++i) { d = offset / dims[i]; mlndx[i] = offset - d * dims[i]; offset = d; };
  return;
}

unsigned int argument_coherence_get_value(unsigned int coh_ctrl, unsigned int tot_args, unsigned int arg_num)
/** Given a composite coherence control value, returns an individual component.
    No argument consistency check (0 <= arg_num < tot_args). **/
{
  const unsigned int TWO_BITS_SET = 3;
  unsigned int coh = ((coh_ctrl >> ((tot_args - (arg_num + 1)) * 2)) & (TWO_BITS_SET));
  return coh;
}

int argument_coherence_set_value(unsigned int* coh_ctrl, unsigned int tot_args, unsigned int arg_num, unsigned int coh_val)
/** Sets the coherence value for a specific argument in a composite coherence control value. **/
{
  if (arg_num < tot_args) {
    const unsigned int TWO_BITS_SET = 3;
    if ((coh_val & (~TWO_BITS_SET)) == 0) {
      const unsigned int clear_mask = ((TWO_BITS_SET) << ((tot_args - (arg_num + 1)) * 2));
      const unsigned int set_mask = ((coh_val) << ((tot_args - (arg_num + 1)) * 2));
      const unsigned int coh = (((*coh_ctrl) & (~clear_mask)) | set_mask);
      *coh_ctrl = coh;
    }
    else {
      return 2;
    }
  }
  else {
    return 1;
  }
  return 0;
}
