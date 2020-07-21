
#include "tensor_algebra.h"

#include "cstdlib"

int tensSignature_create(talsh_tens_signature_t** tsigna)
{
  if (tsigna == NULL) return -1;
  *tsigna = (talsh_tens_signature_t*)malloc(sizeof(talsh_tens_signature_t));
  if (*tsigna == NULL) return TRY_LATER;
  return tensSignature_clean(*tsigna);
}

int tensSignature_clean(talsh_tens_signature_t* tsigna)
{
  if (tsigna != NULL) {
    tsigna->num_dim = -1;   //tensor rank
    tsigna->offsets = NULL; //array of offsets
  }
  else {
    return -1;
  }
  return 0;
}

int tensSignature_construct(talsh_tens_signature_t* tsigna, int rank, const size_t* offsets)
{
  int errc = 0;
  if (tsigna != NULL) {
    if (tsigna->num_dim >= 0) errc = tensSignature_destruct(tsigna);
    if (errc == 0) {
      if (rank > 0) {
        if (offsets != NULL) {
          tsigna->offsets = (size_t*)malloc(sizeof(size_t) * rank);
          if (tsigna->offsets == NULL) return TRY_LATER;
          for (int i = 0; i < rank; ++i) tsigna->offsets[i] = offsets[i];
          tsigna->num_dim = rank;
        }
        else {
          errc = -3;
        }
      }
      else if (rank == 0) {
        tsigna->num_dim = rank;
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

int tensSignature_destruct(talsh_tens_signature_t* tsigna)
{
  if (tsigna == NULL) return -1;
  if (tsigna->offsets != NULL) free(tsigna->offsets);
  return tensSignature_clean(tsigna);
}

int tensSignature_destroy(talsh_tens_signature_t* tsigna)
{
  if (tsigna == NULL) return -1;
  int errc = tensSignature_destruct(tsigna);
  free(tsigna);
  return errc;
}