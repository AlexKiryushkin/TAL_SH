
#include "device_algebra.h"

#include <cstdio>

int tensBlck_create(tensBlck_t** ctens)
/** Creates an empty instance of tensBlck_t and initializes it to null (on Host). **/
{
  *ctens = (tensBlck_t*)malloc(sizeof(tensBlck_t)); if (*ctens == NULL) return TRY_LATER;
  return tensBlck_clean(*ctens);
}

int tensBlck_clean(tensBlck_t* ctens)
/** Cleans an undefined tensBlck_t object. **/
{
  if (ctens == NULL) return -1;
  ctens->data_kind = NO_TYPE;
  ctens->src_rsc = NULL; //source memory resource (where the tensor body is before the operation)
  ctens->dst_rsc = NULL; //destination memory resource (where the tensor body will be after the operation)
  ctens->tmp_rsc = NULL; //temporary memory resource (where the tensor body can be during the operation)
  return tensShape_clean(&(ctens->shape));
}

int tensBlck_destroy(tensBlck_t* ctens)
/** Destroys a defined instance of tensBlck_t (either nullified or shape-defined).
    A return status NOT_CLEAN indicates an unsuccessful resource release, which
    can be considered as a tolerable error (the object will still be destroyed). **/
{
  int n, errc;

  errc = 0; n = 0;
  if (ctens == NULL) return -1;
  errc = tensBlck_destruct(ctens); if (errc) n = NOT_CLEAN;
  if (ctens->tmp_rsc != NULL) { errc = tensDevRsc_destroy(ctens->tmp_rsc); if (errc) n = NOT_CLEAN; }
  if (ctens->dst_rsc != NULL && ctens->dst_rsc != ctens->src_rsc) { errc = tensDevRsc_destroy(ctens->dst_rsc); if (errc) n = NOT_CLEAN; }
  if (ctens->src_rsc != NULL) { errc = tensDevRsc_destroy(ctens->src_rsc); if (errc) n = NOT_CLEAN; }
  ctens->src_rsc = NULL; ctens->dst_rsc = NULL; ctens->tmp_rsc = NULL;
  free(ctens);
  return n;
}

int tensBlck_construct(tensBlck_t* ctens, //pointer to defined tensor block (either nullified or defined to a value)
  int pinned,        //YEP: tensor shape multi-indices will be pinned (for GPU), NOPE: regular malloc (not pinned)
  int trank,         //tensor rank
  const int* dims,   //tensor dimension extents (when trank > 0)
  const int* divs,   //tensor dimension dividers (when trank > 0, optional)
  const int* grps)   //tensor dimension groups (when trank > 0, optional)
/** Constructs (defines/redefines) a tensor block without attaching its body (only the shape).
    If the tensor block is to be used on Nvidia GPUs or other asynchronous devices,
    argument <pinned> must be set to YEP (NOPE will not use pinned memory).
    A return status NOT_CLEAN indicates an unsuccessful resource release, which,
    can be considered as a tolerable error (the object will still be constructed). **/
{
  int n, errc;

  n = 0;
  if (ctens == NULL) return -1;
  if (trank < 0 || trank > MAX_TENSOR_RANK) return -2; //invalid tensor rank
  if (trank > 0 && dims == NULL) return -3; //dimension extents must be present for rank>0 tensors
  errc = tensBlck_destruct(ctens); if (errc != 0) { if (errc == NOT_CLEAN) { n = errc; } else { return 1; } }
  errc = tensShape_construct(&(ctens->shape), pinned, trank, dims, divs, grps);
  if (errc != 0) { if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; } else { return 2; } }
  return n; //either 0 or NOT_CLEAN
}

int tensBlck_attach_body(tensBlck_t* ctens, //pointer to a shape-defined (constructed) tensor block
  int data_kind,     //data kind (R4,R8,C4,C8)
  int dev_id,        //flat device id where the body resides (or should reside): Defaults to Host
  void* body_ptr,    //pointer to the tensor body (global memory of device <dev_id>)
  int buf_entry)     //argument buffer entry handle corresponding to the <body_ptr> (optional)
/** Attaches a body to a shape-defined tensor block (with an empty body). If both <body_ptr> and <buf_entry> are absent,
    a resource will be allocated on device <dev_id> in the device argument buffer (if available). If <buf_entry> is absent,
    a defined <body_ptr> points to an external memory (either pinned or not). If both <body_ptr> and <buf_entry> are defined,
    the external memory is assumed to be within that argument buffer entry. In all cases, the memory resource will be
    associated with the .src_rsc component of tensBlck_t. It is forbidden to attempt allocating/attaching a memory resource
    when an existing memory resource is still in use (this will result in an error). A return status of TRY_LATER or
    DEVICE_UNABLE indicates the current or permanent shortage in the necessary resources and is not an error. **/
{
  int errc, dks;
  size_t vol, body_size;

  if (ctens == NULL) return -1;
  errc = tens_valid_data_kind(data_kind, &dks);
  if (errc != YEP || data_kind == NO_TYPE) return -2;
  if (ctens->shape.num_dim < 0 || ctens->shape.num_dim > MAX_TENSOR_RANK) return -3; //tensor block must be shape-defined
  if (body_ptr == NULL && buf_entry >= 0) return -4; //a defined argument buffer entry must be supplied with the corresponding pointer
  if (dev_id < 0) { dev_id = encode_device_id(DEV_HOST, 0); if (dev_id < 0 || dev_id >= DEV_MAX) return -5; } //dev_id defaults to Host
  if (ctens->src_rsc == NULL) {
    errc = tensDevRsc_create(&(ctens->src_rsc)); if (errc != 0 || ctens->src_rsc == NULL) return 1;
  }
  else {
    if (tensDevRsc_is_empty(ctens->src_rsc) == NOPE) return 2; //source resource is not empty (release it first)
  }
  vol = tensShape_volume(&(ctens->shape)); //tensor body volume (number of elements)
  body_size = vol * ((size_t)dks); //tensor body size in bytes
  if (body_ptr == NULL) { //allocate memory in the argument buffer
    errc = tensDevRsc_allocate_mem(ctens->src_rsc, dev_id, body_size, YEP);
    if (errc != 0) { if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; } else { return 3; } }
  }
  else { //associate memory
    errc = tensDevRsc_attach_mem(ctens->src_rsc, dev_id, body_ptr, buf_entry);
    if (errc != 0) { if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; } else { return 4; } }
  }
  ctens->data_kind = data_kind;
  return 0;
}

int tensBlck_destruct(tensBlck_t* ctens, int release_body, int which_body)
/** Destructs a defined tensor block (releases all resources and initializes the tensor block to null).
    If <release_body> == YEP/NOPE, the global memory resources will be released/kept. Argument <which_body>
    can further regulate which tensor body to be released/kept (SOURCE, DESTINATION, TEMPORARY, EVERYTHING).
    A return status NOT_CLEAN indicates an unsuccessful resource release that may be considered as a
    tolerable error since the tensor block will be nullified anyway. Although device resources are
    released the resource objects themselves are not (they are only destroyed in _destroy method). **/
{
  int n, errc;

  n = 0;
  if (ctens == NULL) return -1;
  if (ctens->shape.num_dim >= 0) { //shape-defined tensor block
    if (ctens->shape.num_dim > MAX_TENSOR_RANK) return -2;
    //Release the TEMPORARY resource:
    if (ctens->tmp_rsc != NULL &&
      ((release_body == YEP && (which_body == EVERYTHING || which_body == TEMPORARY)) ||
        (release_body == NOPE && (which_body != EVERYTHING && which_body != TEMPORARY)))) {
      errc = tensDevRsc_release_all(ctens->tmp_rsc); if (errc != 0) n = NOT_CLEAN; //Note: Resource object is not destroyed!
    }
    ctens->tmp_rsc = NULL;
    //Release the DESTINATION resource (only if different from the SOURCE resource):
    if (ctens->dst_rsc != NULL &&
      ((release_body == YEP && (which_body == EVERYTHING || which_body == DESTINATION)) ||
        (release_body == NOPE && (which_body != EVERYTHING && which_body != DESTINATION)))) {
      if (ctens->dst_rsc != ctens->src_rsc) {
        errc = tensDevRsc_release_all(ctens->dst_rsc); if (errc != 0) n = NOT_CLEAN; //Note: Resource object is not destroyed!
      }
      else {
        ctens->dst_rsc = NULL; //destination resource simply pointed to the source resource
      }
    }
    ctens->dst_rsc = NULL;
    //Release the SOURCE resource:
    if (ctens->src_rsc != NULL &&
      ((release_body == YEP && (which_body == EVERYTHING || which_body == SOURCE)) ||
        (release_body == NOPE && (which_body != EVERYTHING && which_body != SOURCE)))) {
      errc = tensDevRsc_release_all(ctens->src_rsc); if (errc != 0) n = NOT_CLEAN; //Note: Resource object is not destroyed!
    }
    ctens->src_rsc = NULL;
    if (tens_valid_data_kind(ctens->data_kind) != YEP) n = NOT_CLEAN;
  }
  ctens->data_kind = NO_TYPE;
  errc = tensShape_destruct(&(ctens->shape)); if (errc) { if (errc == NOT_CLEAN) { n = NOT_CLEAN; } else { return 1; } }
  return n;
}

int tensBlck_src_dev_id(const tensBlck_t* ctens, int* dev_kind)
/** Returns the device id on which the source data (tensor body) resides.
    If <dev_kind> is provided (!=NULL), the device id will be kind-specific,
    belonging to the device kind <dev_kind>. Otherwise, it will be the flat id.
    A return status DEV_NULL indicates no current source data. A return
    status DEV_MAX indicates a failure (error). **/
{
  int dev_id;

  dev_id = DEV_NULL;
  if (dev_kind != NULL) *dev_kind = DEV_NULL;
  if (ctens == NULL) return DEV_MAX;
  if (ctens->src_rsc != NULL) {
    if (dev_kind == NULL) {
      dev_id = ((*ctens).src_rsc)->dev_id;
    }
    else {
      dev_id = decode_device_id(((*ctens).src_rsc)->dev_id, dev_kind);
    }
  }
  return dev_id;
}

int tensBlck_present(const tensBlck_t* ctens, int dev_id, int dev_kind)
/** Returns YEP/NOPE if the tensor body is present/absent on the device specified by
    a device id <dev_id> and a device kind <dev_kind>. When <dev_id> is present,
    the presence of <dev_kind> determines whether <dev_id> is a flat or kind-specific.
    When <dev_id> is absent but <dev_kind> is present, the presence will be checked
    against the specified device kind. If both <dev_id> and <dev_kind> are absent,
    any presence will be checked (on any device). A return status NVTAL_FAILURE
    indicates invalid arguments. **/
{
  int src_dev, dst_dev, devn, devk;

  if (ctens == NULL) return NVTAL_FAILURE;
  if (ctens->src_rsc != NULL) { src_dev = ctens->src_rsc->dev_id; }
  else { src_dev = DEV_NULL; }
  if (ctens->dst_rsc != NULL) { dst_dev = ctens->dst_rsc->dev_id; }
  else { dst_dev = DEV_NULL; }
  if (dev_kind == DEV_NULL) {
    if (dev_id == DEV_NULL) {
      if (src_dev >= 0 || dst_dev >= 0) return YEP;
    }
    else {
      if (dev_id < 0 || dev_id >= DEV_MAX) return NVTAL_FAILURE;
      if (src_dev == dev_id || dst_dev == dev_id) return YEP;
    }
  }
  else {
    if (valid_device_kind(dev_kind) != YEP) return NVTAL_FAILURE;
    if (dev_id == DEV_NULL) {
      devn = decode_device_id(src_dev, &devk);
      if (devn >= 0 && devk == dev_kind) return YEP;
      devn = decode_device_id(dst_dev, &devk);
      if (devn >= 0 && devk == dev_kind) return YEP;
    }
    else {
      devn = encode_device_id(dev_id, dev_kind);
      if (devn >= DEV_MAX) return NVTAL_FAILURE;
      if (src_dev == devn || dst_dev == devn) return YEP;
    }
  }
  return NOPE;
}

size_t tensBlck_volume(const tensBlck_t* ctens)
/** Returns the volume of a tensor block (number of elements)
    or zero in cases of an empty tensor block or an error. **/
{
  if (ctens == NULL) return 0;
  size_t tvol = tensShape_volume(&(ctens->shape));
  return tvol;
}

void tensBlck_print(const tensBlck_t* ctens)
/** Print info on a given tensor block. **/
{
  if (ctens != NULL) {
    printf("\n#MESSAGE: Printing tensor block info:\n");
    printf(" Tensor block address   : %p\n", ctens);
    printf(" Tensor block data kind : %d\n", ctens->data_kind);
    printf(" Tensor block rank      : %d\n", ctens->shape.num_dim);
    if (ctens->shape.num_dim >= 0 && ctens->shape.num_dim <= MAX_TENSOR_RANK) {
      printf(" Tensor block dimensions:"); for (int i = 0; i < (ctens->shape.num_dim); i++) printf(" %d", ctens->shape.dims[i]);
      printf("\n Tensor block source resource: %p:\n", ctens->src_rsc);
      if (ctens->src_rsc != NULL) {
        printf("  Device ID     : %d\n", ctens->src_rsc->dev_id);
        printf("  Memory address: %p\n", ctens->src_rsc->gmem_p);
        printf("  Buffer entry  : %d\n", ctens->src_rsc->buf_entry);
        printf("  External mem  : %d\n", ctens->src_rsc->mem_attached);
      }
      printf(" Tensor block destination resource: %p:\n", ctens->dst_rsc);
      if (ctens->dst_rsc != NULL) {
        printf("  Device ID     : %d\n", ctens->dst_rsc->dev_id);
        printf("  Memory address: %p\n", ctens->dst_rsc->gmem_p);
        printf("  Buffer entry  : %d\n", ctens->dst_rsc->buf_entry);
        printf("  External mem  : %d\n", ctens->dst_rsc->mem_attached);
      }
      printf(" Tensor block temporary resource: %p:\n", ctens->tmp_rsc);
      if (ctens->tmp_rsc != NULL) {
        printf("  Device ID     : %d\n", ctens->tmp_rsc->dev_id);
        printf("  Memory address: %p\n", ctens->tmp_rsc->gmem_p);
        printf("  Buffer entry  : %d\n", ctens->tmp_rsc->buf_entry);
        printf("  External mem  : %d\n", ctens->tmp_rsc->mem_attached);
      }
    }
    printf("#END OF MESSAGE\n");
  }
  else {
    printf("\n#WARNING(tensor_algebra_gpu_nvidia:tensBlck_print): NULL pointer!\n");
  }
  return;
}

int tensBlck_init_host(tensBlck_t* ctens, double init_val)
/** Initializes a tensor block on Host. **/
{
  int i, dev_kind;
  size_t vol;
  float fval;
  float* fp;
  double* dp;
  if (ctens == NULL) return -1;
  if (ctens->shape.num_dim < 0 || ctens->src_rsc == NULL) return -2;
  if (ctens->src_rsc->gmem_p == NULL) return -3;
  if (tens_valid_data_kind(ctens->data_kind) != YEP || ctens->data_kind == NO_TYPE) return -4;
  i = decode_device_id(ctens->src_rsc->dev_id, &dev_kind); if (dev_kind != DEV_HOST || i != 0) return 1;
  vol = tensBlck_volume(ctens); if (vol == 0) return -5;
  switch (ctens->data_kind) {
  case R4:
    fval = (float)init_val;
    fp = (float*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,fp,fval) schedule(guided)
    for (size_t l = 0; l < vol; l++) fp[l] = fval;
    break;
  case R8:
    dp = (double*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,dp,init_val) schedule(guided)
    for (size_t l = 0; l < vol; l++) dp[l] = init_val;
    break;
  default:
    return 2;
  }
  return 0;
}

double tensBlck_norm2_host(const tensBlck_t* ctens)
/** Computes the squared 2-norm of the tensor block on Host. **/
{
  int i, dev_kind;
  size_t vol;
  double nrm2;
  float* fp;
  double* dp;
  if (ctens == NULL) return -1.;
  if (ctens->shape.num_dim < 0 || ctens->src_rsc == NULL) return -2.;
  if (ctens->src_rsc->gmem_p == NULL) return -3.;
  if (tens_valid_data_kind(ctens->data_kind) != YEP || ctens->data_kind == NO_TYPE) return -4.;
  i = decode_device_id(ctens->src_rsc->dev_id, &dev_kind); if (dev_kind != DEV_HOST || i != 0) return -5.;
  vol = tensBlck_volume(ctens); if (vol == 0) return -6.;
  nrm2 = 0.0;
  switch (ctens->data_kind) {
  case R4:
    fp = (float*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,fp) schedule(guided) reduction(+:nrm2)
    for (size_t l = 0; l < vol; l++) nrm2 += (double)(fp[l] * fp[l]);
    break;
  case R8:
    dp = (double*)(ctens->src_rsc->gmem_p);
#pragma omp parallel for shared(vol,dp) schedule(guided) reduction(+:nrm2)
    for (size_t l = 0; l < vol; l++) nrm2 += dp[l] * dp[l];
    break;
  default:
    return -7.;
  }
  return nrm2;
}