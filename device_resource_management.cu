
#include "tensor_algebra.h"

#include <cstdio>
#include <cstdlib>

#include "kernel_auxiliary_data.h"
#include "mem_manager.h"

int tensDevRsc_create(talsh_dev_rsc_t** drsc)
/** Creates a new device resource descriptor and inits it to null. **/
{
  int errc = 0;
  *drsc = (talsh_dev_rsc_t*)malloc(sizeof(talsh_dev_rsc_t)); if (*drsc == NULL) return TRY_LATER;
  errc = tensDevRsc_clean(*drsc); errc = 0;
  return errc;
}

int tensDevRsc_clean(talsh_dev_rsc_t* drsc)
/** Cleans (initializes to null) a device resource descriptor. **/
{
  if (drsc != NULL) {
    drsc->dev_id = DEV_NULL; //flat device id
    drsc->gmem_p = NULL;     //device global memory pointer (any device)
    drsc->buf_entry = -1;    //device argument buffer entry (any device)
    drsc->mem_attached = 0;  //memory attachement flag (distinguishes between allocated and attached memory)
  }
  else {
    return -1;
  }
  return 0;
}

int tensDevRsc_is_empty(talsh_dev_rsc_t* drsc)
/** Returns YEP if the device resource descriptor is empty, NOPE otherwise.
    Negative return status means an error. **/
{
  int errc = 0;
  if (drsc == NULL) return -1;
  if (drsc->dev_id >= 0 && drsc->dev_id < DEV_MAX) { if (drsc->gmem_p != NULL) return NOPE; }
  errc = tensDevRsc_clean(drsc); errc = YEP;
  return errc;
}

int tensDevRsc_same(const talsh_dev_rsc_t* drsc0, const talsh_dev_rsc_t* drsc1)
/** Returns YEP if two resource descriptors point to the same resources, NOPE otherwise.
    A negative return status indicates an error. **/
{
  if (drsc0 == NULL) return -1;
  if (drsc1 == NULL) return -2;
  if (drsc0->dev_id == drsc1->dev_id &&
    drsc0->gmem_p == drsc1->gmem_p) return YEP;
  return NOPE;
}

int tensDevRsc_clone(const talsh_dev_rsc_t* drsc_in, talsh_dev_rsc_t* drsc_out)
/** Copy constructor for a device resource. **/
{
  if (drsc_in == NULL) return -1;
  if (drsc_out == NULL) return -2;
  drsc_out->dev_id = drsc_in->dev_id;
  drsc_out->gmem_p = drsc_in->gmem_p;
  drsc_out->buf_entry = drsc_in->buf_entry;
  drsc_out->mem_attached = drsc_in->mem_attached;
  return 0;
}

int tensDevRsc_attach_mem(talsh_dev_rsc_t* drsc, int dev_id, void* mem_p, int buf_entry)
/** Attaches a chunk of existing global memory to a device resource descriptor.
    If <buf_entry> >= 0, that means that the global memory is in the argument buffer.
    If the resource descriptor had already been assigned a device, the <dev_id>
    argument must match that one. **/
{
  if (drsc == NULL) return -1;
  if (dev_id < 0 || dev_id >= DEV_MAX) return -2;
  if (mem_p == NULL) return -3;
  if (drsc->dev_id >= 0 && drsc->dev_id != dev_id) return 1; //a non-empty descriptor must be associated with the same device
  if (drsc->gmem_p != NULL || drsc->buf_entry >= 0) return 2; //resource already has global memory attached
  drsc->dev_id = dev_id; drsc->gmem_p = mem_p; drsc->buf_entry = buf_entry; drsc->mem_attached = 1;
  return 0;
}

int tensDevRsc_detach_mem(talsh_dev_rsc_t* drsc)
/** Detaches a chunk of external memory from a device resource descriptor.
    Regardless of the origin, that memory is not released. **/
{
  int errc = 0;
  if (drsc == NULL) return -1;
  if (drsc->dev_id < 0 || drsc->dev_id >= DEV_MAX) return -2; //empty resource descriptor
  if (drsc->gmem_p == NULL || drsc->mem_attached == 0) return 1; //no global memory attached
  drsc->gmem_p = NULL; drsc->buf_entry = -1; drsc->mem_attached = 0;
  errc = tensDevRsc_is_empty(drsc); errc = 0;
  return errc;
}

int tensDevRsc_allocate_mem(talsh_dev_rsc_t* drsc, int dev_id, size_t mem_size, int in_arg_buf)
/** Allocates global memory on device <dev_id> and attaches it to a device resource descriptor.
    If <in_arg_buf> = YEP, the memory will be allocated via that device's argument buffer.
    A return status TRY_LATER or DEVICE_UNABLE indicates the resource shortage and is not an error. **/
{
  int i, devk, devn, errc;
  char* byte_ptr;

  if (drsc == NULL) return -1;
  if (dev_id < 0 || dev_id >= DEV_MAX) return -2;
  if (mem_size <= 0) return -3;
  devn = decode_device_id(dev_id, &devk); if (devn < 0) return -4; //invalid flat device id
  if (drsc->dev_id >= 0 && drsc->dev_id != dev_id) return 1; //resource was assigned to a different device
  if (drsc->gmem_p != NULL || drsc->buf_entry >= 0) return 2; //resource already has global memory attached
  switch (devk) {
  case DEV_HOST:
    if (in_arg_buf == NOPE) {
      errc = host_mem_alloc_pin(&(drsc->gmem_p), mem_size); if (errc != 0) { drsc->gmem_p = NULL; return 3; }
    }
    else {
      errc = get_buf_entry_host(mem_size, &byte_ptr, &i);
      if (errc != 0) { if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; } else { return 4; } }
      drsc->gmem_p = (void*)byte_ptr; drsc->buf_entry = i;
    }
    drsc->mem_attached = 0;
    break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
    if (in_arg_buf == NOPE) {
      errc = gpu_mem_alloc(&(drsc->gmem_p), mem_size, devn); if (errc != 0) { drsc->gmem_p = NULL; return 5; }
    }
    else {
      errc = get_buf_entry_gpu(devn, mem_size, &byte_ptr, &i);
      if (errc != 0) { if (errc == TRY_LATER || errc == DEVICE_UNABLE) { return errc; } else { return 6; } }
      drsc->gmem_p = (void*)byte_ptr; drsc->buf_entry = i;
    }
    drsc->mem_attached = 0;
    break;
#else
    return -5;
#endif
  case DEV_INTEL_MIC:
#ifndef NO_PHI
    //`Future
    break;
#else
    return -6;
#endif
  case DEV_AMD_GPU:
#ifndef NO_AMD
    //`Future
    break;
#else
    return -7;
#endif
  default:
    return -8; //unknown device kind
  }
  drsc->dev_id = dev_id;
  return 0;
}

int tensDevRsc_free_mem(talsh_dev_rsc_t* drsc)
/** Releases global memory referred to by a device resource descriptor.
    An unsuccessful release of the global memory is marked with
    an error status NOT_CLEAN, but the corresponding components of
    the resource descriptor are cleared anyway. **/
{
  int n, devn, devk, errc;

  n = 0;
  if (drsc == NULL) return -1;
  if (drsc->dev_id < 0 || drsc->dev_id >= DEV_MAX) return -2;
  if (drsc->gmem_p == NULL) return -3;
  devn = decode_device_id(drsc->dev_id, &devk); if (devn < 0) return -4; //invalid flat device id
  if (drsc->mem_attached != 0) return 1; //memory was not allocated but attached
  switch (devk) {
  case DEV_HOST:
    if (drsc->buf_entry >= 0) {
      errc = free_buf_entry_host(drsc->buf_entry);
      if (errc != 0) {
        if (VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): free_buf_entry_host error %d\n", errc);
        n = NOT_CLEAN;
      }
      drsc->buf_entry = -1;
    }
    else {
      errc = host_mem_free_pin(drsc->gmem_p);
      if (errc != 0) {
        if (VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): host_mem_free_pin error %d\n", errc);
        n = NOT_CLEAN;
      }
    }
    drsc->gmem_p = NULL;
    break;
  case DEV_NVIDIA_GPU:
#ifndef NO_GPU
    if (drsc->buf_entry >= 0) {
      errc = free_buf_entry_gpu(devn, drsc->buf_entry);
      if (errc != 0) {
        if (VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): free_buf_entry_gpu error %d\n", errc);
        n = NOT_CLEAN;
      }
      drsc->buf_entry = -1;
    }
    else {
      errc = gpu_mem_free(drsc->gmem_p, devn);
      if (errc != 0) {
        if (VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_free_mem): gpu_mem_free error %d\n", errc);
        n = NOT_CLEAN;
      }
    }
    drsc->gmem_p = NULL;
    break;
#else
    return -5;
#endif
  case DEV_INTEL_MIC:
#ifndef NO_PHI
    //`Future
    break;
#else
    return -6;
#endif
  case DEV_AMD_GPU:
#ifndef NO_AMD
    //`Future
    break;
#else
    return -7;
#endif
  default:
    return -8; //invalid device kind
  }
  errc = tensDevRsc_is_empty(drsc);
  return n;
}

int tensDevRsc_get_gmem_ptr(talsh_dev_rsc_t* drsc, void** gmem_p)
/** Returns the pointer to global memory (.gmem_p component) of the device resource. **/
{
  if (drsc == NULL) return -1;
  if (tensDevRsc_is_empty(drsc) == YEP) return 1;
  *gmem_p = drsc->gmem_p;
  return 0;
}

int tensDevRsc_device_id(talsh_dev_rsc_t* drsc)
/** Returns the device id of the resource. **/
{
  return drsc->dev_id;
}

int tensDevRsc_release_all(talsh_dev_rsc_t* drsc)
/** Releases all device resources in <drsc>. An unsuccessful release
    of one or more resources is marked with a return status NOT_CLEAN,
    but the corresponding components of the device resource descriptor
    are cleaned anyway. An empty resource causes no action. **/
{
  int n, errc;

  n = 0;
  if (drsc == NULL) return -1;
  if (drsc->dev_id >= 0 && drsc->dev_id < DEV_MAX) { //resource handle is not empty
 //Release global memory:
    if (drsc->gmem_p != NULL) {
      if (drsc->mem_attached) {
        errc = tensDevRsc_detach_mem(drsc);
        if (errc) {
          if (VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_release_all): tensDevRsc_detach_mem error %d\n", errc);
          n = NOT_CLEAN;
        }
      }
      else {
        errc = tensDevRsc_free_mem(drsc);
        if (errc) {
          if (VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_release_all): tensDevRsc_free_mem error %d\n", errc);
          n = NOT_CLEAN;
        }
      }
    }
  }
  errc = tensDevRsc_clean(drsc);
  if (n != 0 && VERBOSE) printf("#ERROR(NV-TAL:tensDevRsc_release_all): Error %d\n", n);
  return n;
}

int tensDevRsc_destroy(talsh_dev_rsc_t* drsc)
/** Completely destroys a device resource descriptor. A return status NOT_CLEAN
    means that certain resources have not been released cleanly,
    but it is not a critical error in general (however, a leak can occur). **/
{
  int n, errc;
  n = 0;
  if (drsc == NULL) return -1;
  errc = tensDevRsc_release_all(drsc); if (errc) n = NOT_CLEAN;
  free(drsc);
  return n;
}
