
#include "tensor_algebra.h"

#include <cstdio>
#include <cstdlib>

#include "kernel_auxiliary_data.h"
#include "mem_manager.h"

int tensShape_create(talsh_tens_shape_t** tshape)
/** Creates a tensor shape and cleans it. **/
{
  if (tshape == NULL) return -1;
  *tshape = (talsh_tens_shape_t*)malloc(sizeof(talsh_tens_shape_t));
  if (*tshape == NULL) return TRY_LATER;
  return tensShape_clean(*tshape);
}

int tensShape_clean(talsh_tens_shape_t* tshape)
/** Cleans a tensor shape. A clean (initialized to null) tensor shape has .num_dim=-1.
    A further defined tensor shape has .num_dim >= 0. **/
{
  if (tshape != NULL) {
    tshape->num_dim = -1; //tensor rank
    tshape->dims = NULL;  //tensor dimension extents
    tshape->divs = NULL;  //tensor dimension dividers (segment sizes)
    tshape->grps = NULL;  //tensor dimension groups
  }
  else {
    return -1;
  }
  return 0;
}

int tensShape_construct(talsh_tens_shape_t* tshape, int pinned, int rank, const int* dims, const int* divs, const int* grps)
/** (Re-)defines a tensor shape. It is errorneous to pass an uninitialized tensor shape here,
    that is, the tensor shape *(tshape) must be either clean or previously defined. If <rank> > 0,
    <dims[rank]> must be supplied, whereas <divs[rank]> and <grps[rank]> are always optional.
    If <pinned> = YEP and the tensor shape is clean, then the multi-indices will be allocated
    via the multi-index bank (pinned), otherwise a regular malloc will be called. In case the
    tensor shape is already defined, the previous mutli-index storage entries will be reused,
    regardless whether they were pinned or not (argument <pinned> will not be respected!).
    TRY_LATER or DEVICE_UNABLE return statuses are not errors and in this case the input
    tensor shape will stay unchanged. A return status NOT_CLEAN indicates an unsuccessful
    resource release that can be tolerated in general (the construction will still occur). **/
{
  int i, errc;
  int* mi_dims, * mi_divs, * mi_grps;

  errc = 0;
  //Check arguments:
  if (tshape == NULL) return -1;
  if (rank < 0) return -2;
  if (dims != NULL) { for (i = 0; i < rank; i++) { if (dims[i] < 0) return -3; } }
  if (divs != NULL) { for (i = 0; i < rank; i++) { if (divs[i] < 0) return -4; } }
  if (grps != NULL) { for (i = 0; i < rank; i++) { if (grps[i] < 0) return -5; } }
  if (rank > 0 && dims == NULL) return -6; //dimension extents must be present for rank>0
 //Acquire/release resources if needed:
  mi_dims = NULL; mi_divs = NULL; mi_grps = NULL;
  if (rank > 0 && tshape->num_dim <= 0) { //acquire multi-index resources
    if (tshape->dims != NULL || tshape->divs != NULL || tshape->grps != NULL) return -7; //shape must be clean if .num_dim<0
    if (pinned == NOPE) {
      mi_dims = (int*)malloc(3 * MAX_TENSOR_RANK * sizeof(int));
      if (mi_dims == NULL) return TRY_LATER;
      mi_divs = mi_dims + MAX_TENSOR_RANK;
      mi_grps = mi_divs + MAX_TENSOR_RANK;
    }
    else {
      //Multi-index "Dimension extents":
      errc = mi_entry_get(&mi_dims); //acquire a mi resource
      if (errc != 0) {
        if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; }
        else { return 1; }
      }
      //Multi-index "Dimension dividers":
      errc = mi_entry_get(&mi_divs); //acquire a mi resource
      if (errc != 0) {
        i = mi_entry_release(mi_dims);
        if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; }
        else { return 2; }
      }
      //Multi-index "Dimension groups":
      errc = mi_entry_get(&mi_grps); //acquire a mi resource
      if (errc != 0) {
        i = mi_entry_release(mi_divs); i = mi_entry_release(mi_dims);
        if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; }
        else { return 3; }
      }
    }
    tshape->dims = mi_dims; tshape->divs = mi_divs; tshape->grps = mi_grps;
    errc = 0;
  }
  else if (rank == 0 && tshape->num_dim > 0) { //release multi-index resources
    errc = tensShape_destruct(tshape); if (errc != 0 && errc != NOT_CLEAN) return 4;
  }
  //Define the new tensor shape:
  tshape->num_dim = rank;
  if (dims != NULL) {
    for (i = 0; i < rank; i++) tshape->dims[i] = dims[i];
  }
  if (divs != NULL) {
    for (i = 0; i < rank; i++) tshape->divs[i] = divs[i];
  }
  else {
    for (i = 0; i < rank; i++) tshape->divs[i] = tshape->dims[i]; //default dividers (one segment per dimension)
  }
  if (grps != NULL) {
    for (i = 0; i < rank; i++) tshape->grps[i] = grps[i];
  }
  else {
    for (i = 0; i < rank; i++) tshape->grps[i] = 0; //default groups (all indices belong to the unrestricted group)
  }
  return errc; //either 0 or NOT_CLEAN
}

int tensShape_destruct(talsh_tens_shape_t* tshape)
/** Destructs a defined tensor shape (releases resources and cleans it).
    If the input tensor shape is initialized to null, nothing happens.
    In case of an unsuccessful resource release, a return status NOT_CLEAN
    will be returned, which can be considered as a tolerable error since
    the tensor shape will be cleaned anyway (although a leak can occur). **/
{
  int n, pinned, errc;

  n = 0; //will be incremented upon an unsuccessful resource release
  if (tshape == NULL) return -1;
  if (tshape->num_dim > 0) { //need to release resources
    if (tshape->dims != NULL) {
      pinned = mi_entry_pinned(tshape->dims);
      if (pinned == NOPE) {
        free(tshape->dims); //will free all {dims,divs,grps}
        tshape->dims = NULL; tshape->divs = NULL; tshape->grps = NULL;
      }
      else {
        if (tshape->grps != NULL) { errc = mi_entry_release(tshape->grps); if (errc != 0) n++; tshape->grps = NULL; } //release a mi resource
        if (tshape->divs != NULL) { errc = mi_entry_release(tshape->divs); if (errc != 0) n++; tshape->divs = NULL; } //release a mi resource
        if (tshape->dims != NULL) { errc = mi_entry_release(tshape->dims); if (errc != 0) n++; tshape->dims = NULL; } //release a mi resource
      }
    }
    else {
      return -2;
    }
  }
  if (n != 0) {
    if (VERBOSE) printf("#ERROR(tensShape_destruct): Resource release error %d\n", n);
    n = NOT_CLEAN;
  }
  errc = tensShape_clean(tshape);
  return n; //either 0 or NOT_CLEAN
}

int tensShape_destroy(talsh_tens_shape_t* tshape)
/** Completely destroys a tensor shape. **/
{
  int errc, n;
  if (tshape == NULL) return -1;
  n = 0; errc = tensShape_destruct(tshape); if (errc) n = NOT_CLEAN;
  free(tshape);
  return n; //either 0 (success) or NOT_CLEAN
}

size_t tensShape_volume(const talsh_tens_shape_t* tshape)
/** Returns the volume of a defined tensor shape, or 0 otherwise. **/
{
  int i;
  size_t vol;

  vol = 0;
  if (tshape != NULL) {
    if (tshape->num_dim >= 0 && tshape->num_dim <= MAX_TENSOR_RANK) {
      vol = 1;
      for (i = 0; i < tshape->num_dim; i++) {
        if (tshape->dims[i] > 0) {
          vol *= tshape->dims[i];
        }
        else {
          return 0;
        }
      }
    }
  }
  return vol;
}

int tensShape_rank(const talsh_tens_shape_t* tshape)
/** Returns the tensor shape rank (number of dimensions). **/
{
  return tshape->num_dim;
}

int tensShape_reshape(talsh_tens_shape_t* tshape, int rank, const int* dims, const int* divs, const int* grps)
{
  int tens_rank, pinned, errc;
  size_t vol;

  errc = 0;
  if (tshape == NULL) return -1;
  tens_rank = tensShape_rank(tshape);
  if (tens_rank > 0) {
    if (tshape->dims != NULL) {
      vol = tensShape_volume(tshape);
      pinned = mi_entry_pinned(tshape->dims);
      errc = tensShape_destruct(tshape);
      if (errc == 0) {
        errc = tensShape_construct(tshape, pinned, rank, dims, divs, grps);
        if (errc == 0 && tensShape_volume(tshape) != vol) errc = -2;
      }
    }
    else {
      errc = -3;
    }
  }
  else {
    if (dims != NULL || divs != NULL || grps != NULL) errc = -4;
  }
  return errc;
}

void tensShape_print(const talsh_tens_shape_t* tshape)
/** Prints the tensor shape (dimension extents). **/
{
  int i;

  printf("[");
  for (i = 0; i < (tshape->num_dim); ++i) {
    if (i == (tshape->num_dim) - 1) {
      printf("%d", tshape->dims[i]);
    }
    else {
      printf("%d,", tshape->dims[i]);
    }
  }
  printf("]");
  return;
}
