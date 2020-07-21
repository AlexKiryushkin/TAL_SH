
#include "auxiliary_functions.h"
#include "kernel_auxiliary_data.h"
#include "mem_manager.h"

int prmn_convert(int n, const int* o2n, int* n2o)
/** Converts an O2N permutation into N2O (length = n). Both permutations
    are sign-free and the numeration starts from 1. **/
{
  int i, j;
  if (n >= 0) {
    for (i = 0; i < n; i++) { j = o2n[i] - 1; if (j >= 0 && j < n) { n2o[j] = i + 1; } else { return 1; } }
  }
  else {
    return 2;
  }
  return 0;
}

int non_trivial_prmn(int n, const int* prm)
/** Returns NOPE if the permutation prm[0:n-1] is trivial, YEP otherwise.
    The permutation is sign-free and the numeration starts from 1. No error check. **/
{
  int i, f = NOPE;
  for (i = 0; i < n; i++) { if (prm[i] != i + 1) { f = YEP; break; } }
  return f;
}

#ifndef NO_GPU

int cuda_stream_get(int gpu_num, int* cuda_stream_handle)
/** For GPU#gpu_num, returns a usable CUDA stream handle <cuda_stream_handle>.
Non-zero return status means an error, except the return status TRY_LATER means
no free resources are currently available (not an error). **/
{
  *cuda_stream_handle = -1;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (CUDAStreamFFE[gpu_num] > 0) { //number of free handles left on GPU#gpu_num
        *cuda_stream_handle = CUDAStreamFreeHandle[gpu_num][--CUDAStreamFFE[gpu_num]];
        if (*cuda_stream_handle < 0 || *cuda_stream_handle >= MAX_CUDA_TASKS) {
          *cuda_stream_handle = -1; return 3; //invalid handle: corruption
        }
      }
      else {
        return TRY_LATER; //all handles are currently busy
      }
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

int cuda_stream_release(int gpu_num, int cuda_stream_handle)
/** For GPU#gpu_num, releases a CUDA stream handle <cuda_stream_handle>.
Non-zero return status means an error. **/
{
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (cuda_stream_handle >= 0 && cuda_stream_handle < MAX_CUDA_TASKS) {
        if (CUDAStreamFFE[gpu_num] < 0 || CUDAStreamFFE[gpu_num] > MAX_CUDA_TASKS) return 5; //corrupted
        if (CUDAStreamFFE[gpu_num] < MAX_CUDA_TASKS) {
          CUDAStreamFreeHandle[gpu_num][CUDAStreamFFE[gpu_num]++] = cuda_stream_handle;
        }
        else {
          return 4; //an attempt to release a non-existing handle
        }
      }
      else {
        return 3;
      }
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

cudaStream_t* cuda_stream_ptr(int gpu_num, int cuda_stream_handle)
{
  /** Returns a pointer to a valid CUDA stream handle. **/
  if (gpu_num < 0 || gpu_num >= MAX_GPUS_PER_NODE) return NULL;
  if (cuda_stream_handle < 0 || cuda_stream_handle >= MAX_CUDA_TASKS) return NULL;
  if (gpu_is_mine(gpu_num) > GPU_OFF) return &(CUDAStreamBank[gpu_num][cuda_stream_handle]);
  return NULL;
}

int cuda_event_get(int gpu_num, int* cuda_event_handle)
/** For GPU#gpu_num, returns a usable CUDA event handle <cuda_event_handle>.
Non-zero return status means an error, except the return status TRY_LATER means
no free resources are currently available (not an error). **/
{
  *cuda_event_handle = -1;
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (CUDAEventFFE[gpu_num] > 0) { //number of free handles left on GPU#gpu_num
        *cuda_event_handle = CUDAEventFreeHandle[gpu_num][--CUDAEventFFE[gpu_num]];
        if (*cuda_event_handle < 0 || *cuda_event_handle >= MAX_CUDA_EVENTS) {
          *cuda_event_handle = -1; return 3; //invalid handle: corruption
        }
      }
      else {
        return TRY_LATER; //all handles are currently busy
      }
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

int cuda_event_release(int gpu_num, int cuda_event_handle)
/** For GPU#gpu_num, releases a CUDA event handle <cuda_event_handle>.
Non-zero return status means an error. **/
{
  if (gpu_num >= 0 && gpu_num < MAX_GPUS_PER_NODE) {
    if (gpu_is_mine(gpu_num) > GPU_OFF) {
      if (cuda_event_handle >= 0 && cuda_event_handle < MAX_CUDA_EVENTS) {
        if (CUDAEventFFE[gpu_num] < 0 || CUDAEventFFE[gpu_num] > MAX_CUDA_EVENTS) return 5; //corrupted
        if (CUDAEventFFE[gpu_num] < MAX_CUDA_EVENTS) {
          CUDAEventFreeHandle[gpu_num][CUDAEventFFE[gpu_num]++] = cuda_event_handle;
        }
        else {
          return 4; //an attempt to release a non-existing handle
        }
      }
      else {
        return 3;
      }
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

cudaEvent_t* cuda_event_ptr(int gpu_num, int cuda_event_handle)
{
  /** Returns a pointer to a valid CUDA event handle. **/
  if (gpu_num < 0 || gpu_num >= MAX_GPUS_PER_NODE) return NULL;
  if (cuda_event_handle < 0 || cuda_event_handle >= MAX_CUDA_EVENTS) return NULL;
  if (gpu_is_mine(gpu_num) > GPU_OFF) return &(CUDAEventBank[gpu_num][cuda_event_handle]);
  return NULL;
}

void limit_cuda_blocks2d(int max_blocks, int* bx, int* by)
/** Limits the number of CUDA blocks in a 2d grid to <max_blocks>.
    No argument validity check! **/
{
  if (max_blocks > 1) {
    double rdc = ((double)max_blocks) / (((double)(*bx)) * ((double)(*by)));
    if (rdc < 1.0) {
      rdc = sqrt(rdc);
      if (*bx > * by) {
        *by = (int)(rdc * ((double)(*by))); if (*by < 1) { *by = 1; *bx = max_blocks; return; }
        *bx = (int)(rdc * ((double)(*bx)));
      }
      else {
        *bx = (int)(rdc * ((double)(*bx))); if (*bx < 1) { *bx = 1; *by = max_blocks; return; }
        *by = (int)(rdc * ((double)(*by)));
      }
      if ((*bx) * (*by) > max_blocks) {
        if (*bx > * by) { (*bx)--; }
        else { (*by)--; }
      }
    }
  }
  else {
    *bx = 1; *by = 1;
  }
  return;
}

int tens_op_best_gpu(const tensBlck_t* tens0, const tensBlck_t* tens1, const tensBlck_t* tens2)
/** Returns the optimal GPU for a given set of tensor arguments (from the data locality point of view).
    A negative return status means an error. All arguments are optional. **/
{
  int gpu, dev_kind, gpu0, gpu1, gpu2, s0, s1, s2;

  gpu = -1;
  if (tens0 != NULL) {
    if (tens0->src_rsc == NULL) return -1;
    gpu0 = decode_device_id((tens0->src_rsc)->dev_id, &dev_kind);
    if (dev_kind != DEV_NVIDIA_GPU) gpu0 = -1;
    if (tens1 != NULL) {
      if (tens1->src_rsc == NULL) return -1;
      gpu1 = decode_device_id((tens1->src_rsc)->dev_id, &dev_kind);
      if (dev_kind != DEV_NVIDIA_GPU) gpu1 = -1;
      if (gpu1 >= 0 && gpu1 == gpu0) {
        gpu = gpu1;
      }
      else {
        if (tens2 != NULL) {
          if (tens2->src_rsc == NULL) return -1;
          gpu2 = decode_device_id((tens2->src_rsc)->dev_id, &dev_kind);
          if (dev_kind != DEV_NVIDIA_GPU) gpu2 = -1;
          if (gpu2 >= 0 && (gpu2 == gpu1 || gpu2 == gpu0)) {
            gpu = gpu2;
          }
          else {
            s0 = 0; s1 = 0; s2 = 0;
            if (gpu0 >= 0) s0 = gpu_stats[gpu0].tasks_submitted - (gpu_stats[gpu0].tasks_completed + gpu_stats[gpu0].tasks_deferred + gpu_stats[gpu0].tasks_failed);
            if (gpu1 >= 0) s1 = gpu_stats[gpu1].tasks_submitted - (gpu_stats[gpu1].tasks_completed + gpu_stats[gpu1].tasks_deferred + gpu_stats[gpu1].tasks_failed);
            if (gpu2 >= 0) s2 = gpu_stats[gpu2].tasks_submitted - (gpu_stats[gpu2].tasks_completed + gpu_stats[gpu2].tasks_deferred + gpu_stats[gpu2].tasks_failed);
            if (gpu0 >= 0 && (gpu1 < 0 || s0 <= s1) && (gpu2 < 0 || s0 <= s2)) {
              gpu = gpu0;
            }
            else if (gpu1 >= 0 && (gpu0 < 0 || s1 <= s0) && (gpu2 < 0 || s1 <= s2)) {
              gpu = gpu1;
            }
            else if (gpu2 >= 0 && (gpu1 < 0 || s2 <= s1) && (gpu0 < 0 || s2 <= s0)) {
              gpu = gpu2;
            }
          }
        }
        else {
          s0 = 0; s1 = 0;
          if (gpu0 >= 0) s0 = gpu_stats[gpu0].tasks_submitted - (gpu_stats[gpu0].tasks_completed + gpu_stats[gpu0].tasks_deferred + gpu_stats[gpu0].tasks_failed);
          if (gpu1 >= 0) s1 = gpu_stats[gpu1].tasks_submitted - (gpu_stats[gpu1].tasks_completed + gpu_stats[gpu1].tasks_deferred + gpu_stats[gpu1].tasks_failed);
          if (gpu0 >= 0 && (gpu1 < 0 || s0 <= s1)) {
            gpu = gpu0;
          }
          else if (gpu1 >= 0 && (gpu0 < 0 || s1 <= s0)) {
            gpu = gpu1;
          }
        }
      }
    }
    else {
      gpu = gpu0;
    }
  }
  if (gpu < 0 || gpu >= MAX_GPUS_PER_NODE) gpu = gpu_busy_least();
  if (gpu_is_mine(gpu) <= GPU_OFF) gpu = -1; //for safety
  return gpu;
}

int cuda_task_set_arg(cudaTask_t* cuda_task, unsigned int arg_num, tensBlck_t* tens_p)
/** Sets a specific tensor argument in a CUDA task. The tensor argument is associated with
    the provided tensor block and the required temporary multi-index entries are acquired.
    If the multi-index resources cannot be acquired at this time, TRY_LATER is returned. **/
{
  int cae, errc, i;
  unsigned int n;
#ifdef USE_CUTENSOR
  cutensorStatus_t err_cutensor;
  int64_t exts[MAX_TENSOR_RANK];
#endif

  if (cuda_task == NULL) return -1;
  if (cuda_task->task_error >= 0 || cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //finished or empty CUDA task
  if (arg_num >= MAX_TENSOR_OPERANDS) return -3; //[0..MAX_TENSOR_OPERANDS-1]
  if (tens_p == NULL) return -4;
  if (gpu_is_mine(cuda_task->gpu_id) > GPU_OFF) {
    //Associate the tensor block:
    cuda_task->tens_args[arg_num].tens_p = tens_p; //no checks, just do it
  //Acquire a multi-index entry in pinned Host memory:
    errc = mi_entry_get(&(cuda_task->tens_args[arg_num].prmn_p));
    if (errc) { cuda_task->tens_args[arg_num].prmn_p = NULL; cuda_task->tens_args[arg_num].tens_p = NULL; return TRY_LATER; }
    //Acquire a paired multi-index entry in GPU constant memory:
    errc = const_args_entry_get(cuda_task->gpu_id, &cae);
    if (errc == 0) {
      cuda_task->tens_args[arg_num].const_mem_entry = cae;
    }
    else {
      cuda_task->tens_args[arg_num].prmn_p = NULL; cuda_task->tens_args[arg_num].tens_p = NULL;
      return TRY_LATER;
    }
#ifdef USE_CUTENSOR
    //Acquire cuTensor tensor descriptor:
    n = tens_p->shape.num_dim; for (i = 0; i < n; ++i) exts[i] = (tens_p->shape.dims)[i];
    switch (tens_p->data_kind) {
    case R4:
      err_cutensor = cutensorInitTensorDescriptor(&(cutensor_handle[cuda_task->gpu_id]),
        &((cuda_task->tens_cudesc)[arg_num]), (uint32_t)n, exts, NULL,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY);
      if (err_cutensor != CUTENSOR_STATUS_SUCCESS) return 5;
      break;
    case R8:
      err_cutensor = cutensorInitTensorDescriptor(&(cutensor_handle[cuda_task->gpu_id]),
        &((cuda_task->tens_cudesc)[arg_num]), (uint32_t)n, exts, NULL,
        CUDA_R_64F, CUTENSOR_OP_IDENTITY);
      if (err_cutensor != CUTENSOR_STATUS_SUCCESS) return 4;
      break;
    case C4:
      err_cutensor = cutensorInitTensorDescriptor(&(cutensor_handle[cuda_task->gpu_id]),
        &((cuda_task->tens_cudesc)[arg_num]), (uint32_t)n, exts, NULL,
        CUDA_C_32F, CUTENSOR_OP_IDENTITY);
      if (err_cutensor != CUTENSOR_STATUS_SUCCESS) return 3;
      break;
    case C8:
      err_cutensor = cutensorInitTensorDescriptor(&(cutensor_handle[cuda_task->gpu_id]),
        &((cuda_task->tens_cudesc)[arg_num]), (uint32_t)n, exts, NULL,
        CUDA_C_64F, CUTENSOR_OP_IDENTITY);
      if (err_cutensor != CUTENSOR_STATUS_SUCCESS) return 2;
      break;
    default:
      return -5;
    }
#endif
    //Update number of arguments:
    cuda_task->num_args = MAX(cuda_task->num_args, arg_num + 1); //it is user's responsibility to set all preceding arguments
  }
  else {
    return 1;
  }
  return 0;
}

int cuda_task_set_prefactor(cudaTask_t* cuda_task, talshComplex4 prefactor)
/** Sets a complex prefactor for the tensor operation in a CUDA task (single precision). **/
{
  int errc;
  void* pref_p;

  if (cuda_task == NULL) return -1;
  if (cuda_task->task_error >= 0 || cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //finished or empty CUDA task
  errc = slab_entry_get(&prefactors, &pref_p); if (errc != 0) return -3;
  cuda_task->pref_ptr = pref_p;
  *((talshComplex4*)(cuda_task->pref_ptr)) = prefactor;
  return 0;
}

int cuda_task_set_prefactor(cudaTask_t* cuda_task, talshComplex8 prefactor)
/** Sets a complex prefactor for the tensor operation in a CUDA task (double precision). **/
{
  int errc;
  void* pref_p;

  if (cuda_task == NULL) return -1;
  if (cuda_task->task_error >= 0 || cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //finished or empty CUDA task
  errc = slab_entry_get(&prefactors, &pref_p); if (errc != 0) return -3;
  cuda_task->pref_ptr = pref_p;
  *((talshComplex8*)(cuda_task->pref_ptr)) = prefactor;
  return 0;
}

int cuda_task_record(cudaTask_t* cuda_task, unsigned int coh_ctrl, unsigned int err_code)
/** Records a scheduled CUDA task. A successfully scheduled CUDA task has <err_code>=0,
    otherwise a positive <err_code> indicates a task scheduling failure. In the latter
    case, the CUDA task will be finalized here. Special error code NVTAL_DEFERRED is a
    non-critical task scheduling failure, not considered as an error.  **/
{
  int i, errc;

  if (cuda_task == NULL) return -1;
  if (cuda_task->task_error >= 0) return -2; //CUDA task is not clean
  if (cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -3; //GPU ID is out of range or CUDA task is not clean
  if (cuda_task->num_args == 0 || cuda_task->num_args > MAX_TENSOR_OPERANDS) return -4; //no operands associated with the task
  for (i = 0; i < cuda_task->num_args; i++) { if (cuda_task->tens_args[i].tens_p == NULL) return -5; } //all tensor arguments must be set
  if (err_code == 0) { //successfully scheduled CUDA task
    if (gpu_is_mine(cuda_task->gpu_id) > GPU_OFF) {
      cuda_task->task_error = -1; cuda_task->coherence = coh_ctrl;
    }
    else {
      cuda_task->task_error = 13; cuda_task->coherence = coh_ctrl; //GPU is not mine
      errc = cuda_task_finalize(cuda_task); gpu_stats[cuda_task->gpu_id].tasks_failed++;
    }
  }
  else { //CUDA task that failed scheduling
    cuda_task->task_error = err_code; cuda_task->coherence = coh_ctrl;
    errc = cuda_task_finalize(cuda_task);
    if (err_code == NVTAL_DEFERRED) {
      gpu_stats[cuda_task->gpu_id].tasks_deferred++;
    }
    else {
      gpu_stats[cuda_task->gpu_id].tasks_failed++;
    }
  }
  return 0;
}

int cuda_task_finalize(cudaTask_t* cuda_task) //do not call this function in tensor operations
/** Releases unneeded (temporary and other) memory resources right after a CUDA task
    has completed or failed. In case the resources cannot be released cleanly,
    it returns NOT_CLEAN just as a warning, but the CUDA task is finalized anyway.
    It also applies the coherence control protocol (for successfully completed tasks only).
    Note that the CUDA task is not destructed here, namely CUDA stream/event resources and the
    .tens_p component of .tens_args[] are unmodified (.prmn_p and .const_mem_entry are released). **/
{
  const unsigned int TWO_BITS_SET = 3; //two right bits are set: {0:D,1:M,2:T,3:K}
  unsigned int bts, coh, s_d_same;
  int i, ret_stat, errc;
  cudaTensArg_t* tens_arg;

  if (cuda_task == NULL) return -1;
  if (cuda_task->task_error < 0) return 1; //unfinished or empty CUDA task cannot be finalized
  if (cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return 2; //invalid GPU id or empty
  if (cuda_task->num_args > MAX_TENSOR_OPERANDS) return 3; //invalid number of tensor arguments
  ret_stat = 0; coh = cuda_task->coherence;
  //Release resources for tensor arguments:
  for (i = cuda_task->num_args - 1; i >= 0; i--) { //last argument corresponds to the first (minor) two bits
    bts = (coh) & (TWO_BITS_SET);
    tens_arg = &(cuda_task->tens_args[i]);
    if (tens_arg->tens_p != NULL) { //pointer to the tensor block associated with this argument
      if (tens_arg->tens_p->src_rsc == NULL) return -2; //source must always be present
      if (tens_arg->tens_p->dst_rsc != NULL) {
        if (tens_arg->tens_p->src_rsc->dev_id == tens_arg->tens_p->dst_rsc->dev_id) { s_d_same = YEP; }
        else { s_d_same = NOPE; };
      }
      else {
        if (cuda_task->task_error == 0) return -3; //destination resource must be present for successfully completed CUDA tasks
        s_d_same = NOPE; //no destination resource (failed CUDA tasks only)
      }
      // Release temporary resources (always):
      if (tens_arg->tens_p->tmp_rsc != NULL) {
        errc = tensDevRsc_release_all(tens_arg->tens_p->tmp_rsc);
        if (errc) {
          if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): tmp_rsc resource release error %d\n", errc);
          ret_stat = NOT_CLEAN;
        }
      }
      // Release source/destination resources if needed:
      if (tens_arg->tens_p->dst_rsc == tens_arg->tens_p->src_rsc) tens_arg->tens_p->dst_rsc = NULL;
      if (cuda_task->task_error == 0) { //coherence control for successfully completed CUDA tasks
        if (bts < 2) {
          if (s_d_same == NOPE) {
            errc = tensDevRsc_release_all(tens_arg->tens_p->src_rsc);
            if (errc) {
              if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): src_rsc resource release error %d\n", errc);
              ret_stat = NOT_CLEAN;
            }
          }
          if (bts == 0 && tens_arg->tens_p->dst_rsc != NULL) {
            errc = tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
            if (errc) {
              if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): dst_rsc resource release error %d\n", errc);
              ret_stat = NOT_CLEAN;
            }
          }
        }
        else if (bts == 2) {
          if (s_d_same == NOPE && tens_arg->tens_p->dst_rsc != NULL) {
            errc = tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
            if (errc) {
              if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): dst_rsc resource release error %d\n", errc);
              ret_stat = NOT_CLEAN;
            }
          }
        }
      }
      else { //failed CUDA task
        if (tens_arg->tens_p->dst_rsc != NULL) {
          errc = tensDevRsc_release_all(tens_arg->tens_p->dst_rsc);
          if (errc) {
            if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): dst_rsc resource release error %d\n", errc);
            ret_stat = NOT_CLEAN;
          }
        }
      }
      // Release multi-index entries if any:
      if (tens_arg->prmn_p != NULL) { //if .prmn_p is not from the internal pinned slab nothing will be done:
        if (mi_entry_pinned(tens_arg->prmn_p) == YEP) {
          errc = mi_entry_release(tens_arg->prmn_p);
          if (errc) {
            if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): permutation entry release error %d\n", errc);
            ret_stat = NOT_CLEAN;
          }
          tens_arg->prmn_p = NULL;
        }
      }
      if (tens_arg->const_mem_entry >= 0) {
        errc = const_args_entry_free(cuda_task->gpu_id, tens_arg->const_mem_entry);
        if (errc) {
          if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): constant memory resource release error %d\n", errc);
          ret_stat = NOT_CLEAN;
        }
        tens_arg->const_mem_entry = 0;
      }
      //printf("\n#DEBUG(NV-TAL::cuda_task_finalize): tensBlck_t argument %d end state:\n",i); tensBlck_print(tens_arg->tens_p); //debug
    }
    else {
      if (cuda_task->task_error == 0) return -4; //successfully completed CUDA tasks must have all tensor arguments associated
    }
    coh = coh >> 2; //select the 2-bits for the next argument
  }
  //Release prefactor resource, if needed:
  if (cuda_task->pref_ptr != NULL) {
    errc = slab_entry_release(&prefactors, cuda_task->pref_ptr);
    if (errc) {
      if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_finalize): prefactor release error %d\n", errc);
      ret_stat = NOT_CLEAN;
    }
    cuda_task->pref_ptr = NULL;
  }
  return ret_stat;
}

#endif


