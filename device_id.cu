
#include "tensor_algebra.h"

#include <cmath>

int valid_device_kind(int dev_kind)
/** Returns YEP if <dev_kind> is a valid device kind, inlcluding DEV_NULL. NOPE otherwise. **/
{
  if (dev_kind == DEV_NULL ||
    dev_kind == DEV_HOST ||
    dev_kind == DEV_NVIDIA_GPU ||
    dev_kind == DEV_INTEL_MIC ||
    dev_kind == DEV_AMD_GPU) return YEP;
  return NOPE;
}

int encode_device_id(int dev_kind, int dev_num)
/** Given a device ID <dev_num> within its kind <dev_kind>, returns the flat device ID.
    DEV_MAX value on return means that the arguments were invalid. **/
{
  int dev_id = DEV_MAX; //Return of this value (= outside devices range) will mean that the arguments were invalid
  switch (dev_kind) {
  case DEV_HOST: if (dev_num == 0) dev_id = 0; break;
  case DEV_NVIDIA_GPU: if (dev_num >= 0 && dev_num < MAX_GPUS_PER_NODE) dev_id = 1 + dev_num; break;
  case DEV_INTEL_MIC: if (dev_num >= 0 && dev_num < MAX_MICS_PER_NODE) dev_id = 1 + MAX_GPUS_PER_NODE + dev_num; break;
  case DEV_AMD_GPU: if (dev_num >= 0 && dev_num < MAX_AMDS_PER_NODE) dev_id = 1 + MAX_GPUS_PER_NODE + MAX_MICS_PER_NODE + dev_num; break;
  default: dev_id = DEV_MAX; //unknown device kind
  }
  return dev_id;
}

int decode_device_id(int dev_id, int* dev_kind)
/** Given a flat device ID <dev_id>, returns the device kind <dev_kind> (optional)
    and the kind-specific device ID (>=0) as the return value.
    A negative return status (DEV_NULL) indicates an invalid <dev_id>. **/
{
  int dvn, dvid;

  dvn = DEV_NULL; //negative return value will correspond to an invalid <dev_id>
  if (dev_kind != NULL) *dev_kind = DEV_NULL;
  dvid = abs(dev_id); //flat device id is defined up to a sign
  if (dvid == 0) { //Host
    if (dev_kind != NULL) *dev_kind = DEV_HOST;
    dvn = 0;
  }
  else if (dvid >= 1 && dvid <= MAX_GPUS_PER_NODE) { //Nvidia GPU
    if (dev_kind != NULL) *dev_kind = DEV_NVIDIA_GPU;
    dvn = dvid - 1;
  }
  else if (dvid >= 1 + MAX_GPUS_PER_NODE && dvid <= MAX_GPUS_PER_NODE + MAX_MICS_PER_NODE) { //Intel MIC
    if (dev_kind != NULL) *dev_kind = DEV_INTEL_MIC;
    dvn = dvid - 1 - MAX_GPUS_PER_NODE;
  }
  else if (dvid >= 1 + MAX_GPUS_PER_NODE + MAX_MICS_PER_NODE && dvid <= MAX_GPUS_PER_NODE + MAX_MICS_PER_NODE + MAX_AMDS_PER_NODE) { //AMD GPU
    if (dev_kind != NULL) *dev_kind = DEV_AMD_GPU;
    dvn = dvid - 1 - MAX_GPUS_PER_NODE - MAX_MICS_PER_NODE;
  }
  return dvn; //ID of the device within its kind
}
