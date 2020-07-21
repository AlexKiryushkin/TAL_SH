
#include "device_algebra.h"

#include <cstdio>

#include "auxiliary_functions.h"
#include "kernel_auxiliary_data.h"
#include "mem_manager.h"

#ifndef NO_GPU

//CUDA TASK API:
__host__ int cuda_task_create(cudaTask_t** cuda_task)
/** Creates an empty instance of cudaTask_t. An unsuccessful attempt
    to allocate memory for the CUDA task returns status TRY_LATER. **/
{
  int errc = 0;
  //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:cuda_task_create): New CUDA task: sizeof(cudaTask_t) = %d",sizeof(cudaTask_t)); //debug
  *cuda_task = (cudaTask_t*)malloc(sizeof(cudaTask_t)); if (*cuda_task == NULL) return TRY_LATER;
  errc = cuda_task_clean(*cuda_task); errc = 0;
  return errc;
}

__host__ int cuda_task_clean(cudaTask_t* cuda_task)
/** Cleans (initializes to null) a freshly allocated CUDA task. **/
{
  if (cuda_task == NULL) return -1;
  cuda_task->task_error = -1; cuda_task->gpu_id = -1; cuda_task->num_args = 0;
  cuda_task->stream_hl = -1;
  cuda_task->event_start_hl = -1; cuda_task->event_comput_hl = -1;
  cuda_task->event_output_hl = -1; cuda_task->event_finish_hl = -1;
#ifdef GPU_FINE_TIMING
  cuda_task->event_mmbeg_hl = -1; cuda_task->event_mmend_hl = -1;
#endif
  for (int i = 0; i < MAX_TENSOR_OPERANDS; ++i) {
    cuda_task->tens_args[i].tens_p = NULL;
    cuda_task->tens_args[i].prmn_p = NULL;
    cuda_task->tens_args[i].const_mem_entry = -1;
  }
  cuda_task->pref_ptr = NULL;
  return 0;
}

__host__ int cuda_task_construct(cudaTask_t* cuda_task, int gpu_id)
/** Constructs a CUDA task ready for recording on GPU#gpu_id (acquires resources).
    If <gpu_id> is not passed here (negative), the currently active GPU will be used.
    Returns TRY_LATER or DEVICE_UNABLE in case of temporary or permanent
    shortage of GPU resources, respectively (CUDA task is left clean). **/
{
  int i, errc;

  errc = 0;
  if (cuda_task == NULL) return -1;
  if (cuda_task->task_error >= 0 || cuda_task->gpu_id >= 0 || cuda_task->num_args > 0) return 1; //CUDA task is not clean: Destruct/clean it first
  i = cuda_task_clean(cuda_task); //just in case
  if (gpu_id < 0) gpu_id = gpu_in_focus();
  if (gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE) return 2; //gpu_id is out of range
  if (gpu_is_mine(gpu_id) > GPU_OFF) {
    errc = cuda_stream_get(gpu_id, &(cuda_task->stream_hl));
    if (errc != 0) {
      cuda_task->stream_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 3;
    }
    else {
      errc = cuda_event_get(gpu_id, &(cuda_task->event_start_hl));
      if (errc != 0) {
        cuda_task->event_start_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 4;
      }
      else {
        errc = cuda_event_get(gpu_id, &(cuda_task->event_comput_hl));
        if (errc != 0) {
          cuda_task->event_comput_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 5;
        }
        else {
          errc = cuda_event_get(gpu_id, &(cuda_task->event_output_hl));
          if (errc != 0) {
            cuda_task->event_output_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 6;
          }
          else {
            errc = cuda_event_get(gpu_id, &(cuda_task->event_finish_hl));
            if (errc != 0) {
              cuda_task->event_finish_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 7;
#ifdef GPU_FINE_TIMING
            }
            else {
              errc = cuda_event_get(gpu_id, &(cuda_task->event_mmbeg_hl));
              if (errc != 0) {
                cuda_task->event_mmbeg_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 8;
              }
              else {
                errc = cuda_event_get(gpu_id, &(cuda_task->event_mmend_hl));
                if (errc != 0) {
                  cuda_task->event_mmend_hl = -1; if (errc != TRY_LATER && errc != DEVICE_UNABLE) errc = 9;
                }
              }
#endif
            }
          }
        }
      }
    }
    if (errc == 0) {
      cuda_task->task_error = -1; cuda_task->gpu_id = gpu_id;
    }
    else {
#ifdef GPU_FINE_TIMING
      i = cuda_event_release(gpu_id, cuda_task->event_mmbeg_hl); cuda_task->event_mmbeg_hl = -1;
      i = cuda_event_release(gpu_id, cuda_task->event_mmend_hl); cuda_task->event_mmend_hl = -1;
#endif
      i = cuda_event_release(gpu_id, cuda_task->event_finish_hl); cuda_task->event_finish_hl = -1;
      i = cuda_event_release(gpu_id, cuda_task->event_output_hl); cuda_task->event_output_hl = -1;
      i = cuda_event_release(gpu_id, cuda_task->event_comput_hl); cuda_task->event_comput_hl = -1;
      i = cuda_event_release(gpu_id, cuda_task->event_start_hl); cuda_task->event_start_hl = -1;
      i = cuda_stream_release(gpu_id, cuda_task->stream_hl); cuda_task->stream_hl = -1;
      i = cuda_task_clean(cuda_task);
    }
  }
  else {
    return DEVICE_UNABLE;
  }
  return errc;
}

__host__ int cuda_task_destruct(cudaTask_t* cuda_task)
/** Destructs a defined completed CUDA task or does nothing. If the CUDA task
    is defined but not completed, a return status TRY_LATER is returned.
    If any of the resources used by the CUDA task cannot be released cleanly,
    a return status NOT_CLEAN is returned. Nevertheless, the CUDA task will be
    clean at the end. **/
{
  int n, errc;

  if (cuda_task == NULL) return -1;
  errc = cuda_task_completed(cuda_task); //CUDA task is finalized there (if completed or failed)
  if (errc == CUDA_TASK_EMPTY) return 0;
  n = 0; //number of unsuccessful resource releases
  if (errc == CUDA_TASK_COMPLETED || errc == CUDA_TASK_ERROR) {
    if (cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -2; //GPU id is out of allowed range
    if (cuda_task == LastTask[cuda_task->gpu_id]) LastTask[cuda_task->gpu_id] = NULL; //clear task dependency
  // Release CUDA resources:
    errc = cuda_stream_release(cuda_task->gpu_id, cuda_task->stream_hl); cuda_task->stream_hl = -1; if (errc != 0) n++;
    errc = cuda_event_release(cuda_task->gpu_id, cuda_task->event_start_hl); cuda_task->event_start_hl = -1; if (errc != 0) n++;
    errc = cuda_event_release(cuda_task->gpu_id, cuda_task->event_comput_hl); cuda_task->event_comput_hl = -1; if (errc != 0) n++;
    errc = cuda_event_release(cuda_task->gpu_id, cuda_task->event_output_hl); cuda_task->event_output_hl = -1; if (errc != 0) n++;
    errc = cuda_event_release(cuda_task->gpu_id, cuda_task->event_finish_hl); cuda_task->event_finish_hl = -1; if (errc != 0) n++;
#ifdef GPU_FINE_TIMING
    errc = cuda_event_release(cuda_task->gpu_id, cuda_task->event_mmbeg_hl); cuda_task->event_mmbeg_hl = -1; if (errc != 0) n++;
    errc = cuda_event_release(cuda_task->gpu_id, cuda_task->event_mmend_hl); cuda_task->event_mmend_hl = -1; if (errc != 0) n++;
#endif
    // Release prefactor entry, if needed:
    if (cuda_task->pref_ptr != NULL) {
      errc = slab_entry_release(&prefactors, cuda_task->pref_ptr); if (errc != 0) n++;
    }
    // Clean the CUDA task:
    errc = cuda_task_clean(cuda_task);
  }
  else {
    return TRY_LATER; //CUDA task is still in progress
  }
  if (n != 0) n = NOT_CLEAN;
  return n;
}

__host__ int cuda_task_destroy(cudaTask_t* cuda_task)
/** Destroys an instance of cudaTask_t if the CUDA task has completed or empty.
    If the CUDA task is still in progress, a return status TRY_LATER is returned.
    If any of the CUDA task resources could not be released cleanly, a return
    status NOT_CLEAN will be returned but the CUDA task will still be destroyed. **/
{
  int n, errc;

  n = 0;
  if (cuda_task == NULL) return -1;
  errc = cuda_task_completed(cuda_task); //CUDA task is finalized there (if completed or failed)
  if (errc == CUDA_TASK_COMPLETED || errc == CUDA_TASK_ERROR) {
    errc = cuda_task_destruct(cuda_task); if (errc != 0) n = NOT_CLEAN;
  }
  else {
    if (errc != CUDA_TASK_EMPTY) return TRY_LATER; //CUDA task is still in progress
  }
  free(cuda_task);
  return n;
}

__host__ int cuda_task_gpu_id(const cudaTask_t* cuda_task)
/** Returns the GPU id associated with a CUDA task. A negative
    return value means a null or empty task was passed here. **/
{
  if (cuda_task == NULL) return -2;
  if (cuda_task->gpu_id >= 0 && cuda_task->gpu_id < MAX_GPUS_PER_NODE) return cuda_task->gpu_id;
  return -1;
}

__host__ int cuda_task_status(cudaTask_t* cuda_task)
/** Checks the status of a CUDA task. Possible status values are listed in tensor_algebra.h
    and tensor_algebra.inc (keep them consistent!). Both CUDA_TASK_COMPLETED (no errors) and
    CUDA_TASK_ERROR (error occurred) suggest a completion of the CUDA task. An unsuccessful
    attempt to find out the status of the CUDA task results in a return status NVTAL_FAILURE. **/
{
  int task_stat, cur_gpu, errc;
  cudaEvent_t* evnt_p;
  cudaError_t err;

  if (cuda_task == NULL) return CUDA_TASK_EMPTY; //NULL task pointer is treated as an empty task here
  if (cuda_task->task_error < 0 && cuda_task->gpu_id < 0) return CUDA_TASK_EMPTY; //empty CUDA task
  if (cuda_task->task_error >= 0 && cuda_task->gpu_id < 0) return NVTAL_FAILURE; //completed task without an assigned GPU
  if (cuda_task->task_error == 0) return CUDA_TASK_COMPLETED; //CUDA task had completed successfully
  if (cuda_task->task_error > 0) return CUDA_TASK_ERROR; //CUDA task error had been registered
  cur_gpu = gpu_in_focus(); if (cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE) return NVTAL_FAILURE; //get current GPU
  errc = gpu_activate(cuda_task->gpu_id); if (errc != 0) return NVTAL_FAILURE; //could not activate the CUDA task GPU
  evnt_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_finish_hl); if (evnt_p == NULL) return NVTAL_FAILURE;
  err = cudaEventQuery(*evnt_p);
  if (err == cudaSuccess) {
    cuda_task->task_error = 0; errc = cuda_task_finalize(cuda_task); //release unneeded memory resources occupied by the task arguments
    if (errc == 0) {
      cuda_task->task_error = 0; task_stat = CUDA_TASK_COMPLETED; //CUDA task completed, memory released cleanly
    }
    else {
      if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_status): cuda_task_finalize error %d\n", errc);
      cuda_task->task_error = 127; task_stat = CUDA_TASK_ERROR; //CUDA task completed, memory could not be released cleanly
    }
    gpu_stats[cuda_task->gpu_id].tasks_completed++;
  }
  else {
    evnt_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_output_hl); if (evnt_p == NULL) return NVTAL_FAILURE;
    err = cudaEventQuery(*evnt_p);
    if (err == cudaSuccess) {
      task_stat = CUDA_TASK_OUTPUT_THERE; //computing kernel has finished
    }
    else {
      evnt_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_comput_hl); if (evnt_p == NULL) return NVTAL_FAILURE;
      err = cudaEventQuery(*evnt_p);
      if (err == cudaSuccess) {
        task_stat = CUDA_TASK_INPUT_THERE; //computation started, input data is on device (can be reused later)
      }
      else {
        evnt_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_start_hl); if (evnt_p == NULL) return NVTAL_FAILURE;
        err = cudaEventQuery(*evnt_p);
        if (err == cudaSuccess) {
          task_stat = CUDA_TASK_STARTED; //task started
        }
        else {
          task_stat = CUDA_TASK_SCHEDULED; //task has not started yet
        }
      }
    }
  }
  errc = gpu_activate(cur_gpu);
  return task_stat;
}

__host__ int cuda_task_completed(cudaTask_t* cuda_task)
/** Returns CUDA_TASK_COMPLETED or CUDA_TASK_ERROR if an existing CUDA task <cuda_task>
    has completed successfully or due to a scheduling/execution failure, respectively.
    Note that having had successfully checked the CUDA task for completion before will immediately
    suggest completion later (without further querying)! Other possible outputs: CUDA_TASK_EMPTY, CUDA_TASK_SCHEDULED.
    An inability to check the completion status of the CUDA task results in return status NVTAL_FAILURE. **/
{
  int cur_gpu, ret_stat, errc;
  cudaStream_t* strm_p;
  cudaError_t err;

  if (cuda_task == NULL) return CUDA_TASK_EMPTY; //null CUDA task is treated as empty
  if (cuda_task->gpu_id < 0) return CUDA_TASK_EMPTY;
  if (cuda_task->task_error == 0) return CUDA_TASK_COMPLETED; //successful completion had occurred
  if (cuda_task->task_error > 0) return CUDA_TASK_ERROR; //completion due to an error had occurred
  cur_gpu = gpu_in_focus(); if (cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE) return NVTAL_FAILURE;
  errc = gpu_activate(cuda_task->gpu_id); if (errc != 0) return NVTAL_FAILURE;
  strm_p = cuda_stream_ptr(cuda_task->gpu_id, cuda_task->stream_hl); if (strm_p == NULL) return NVTAL_FAILURE;
  err = cudaStreamQuery(*strm_p);
  if (err != cudaSuccess && err != cudaErrorInvalidResourceHandle) { //task is still in progress
    ret_stat = CUDA_TASK_SCHEDULED;
  }
  else { //task completed successfully or has never been scheduled
    if (err == cudaErrorInvalidResourceHandle) { //stream does not exist
      ret_stat = CUDA_TASK_EMPTY;
    }
    else {
      ret_stat = CUDA_TASK_COMPLETED;
      if (cuda_task->task_error < 0) { cuda_task->task_error = 0; gpu_stats[cuda_task->gpu_id].tasks_completed++; }
    }
  }
  if (ret_stat == CUDA_TASK_COMPLETED) {
    errc = cuda_task_finalize(cuda_task);
    if (errc != 0) {
      if (VERBOSE) printf("#ERROR(NV-TAL:cuda_task_completed): cuda_task_finalize error %d\n", errc);
      cuda_task->task_error = 127; //resources could not be released properly
    }
  }
  errc = gpu_activate(cur_gpu);
  return ret_stat;
}

__host__ int cuda_task_wait(cudaTask_t* cuda_task)
/** Waits upon completion of a CUDA task: Returns the output of cuda_task_completed(..).
    Possible returns are CUDA_TASK_COMPLETED, CUDA_TASK_ERROR, CUDA_TASK_SCHEDULED, CUDA_TASK_EMPTY.
    In case the completion of a CUDA task cannot be determined, a return status NVTAL_FAILURE is returned. **/
{
  int i, j;

  i = CUDA_TASK_SCHEDULED; j = 1;
  while (j > 0) {
    i = cuda_task_completed(cuda_task); if (i != CUDA_TASK_SCHEDULED) j--;
  }
  return i;
}

__host__ int cuda_tasks_wait(unsigned int num_tasks, cudaTask_t** cuda_tasks, int* task_stats)
/** Waits upon completion of a series of CUDA tasks. Returns zero on success, non-zero on error.
    On success, <task_stats> will contain the completion status for each task. Note that
    <cuda_tasks> points to an array of CUDA task pointers. **/
{
  int i, j, n;

  if (num_tasks > 0) {
    if (cuda_tasks != NULL && task_stats != NULL) {
      for (i = 0; i < num_tasks; i++) { task_stats[i] = CUDA_TASK_SCHEDULED; }
      n = num_tasks;
      while (n > 0) {
        for (i = 0; i < num_tasks; i++) {
          if (task_stats[i] == CUDA_TASK_SCHEDULED) {
            if (cuda_tasks[i] != NULL) {
              j = cuda_task_completed(cuda_tasks[i]); task_stats[i] = j;
              if (j != CUDA_TASK_SCHEDULED) n--;
            }
            else {
              return 1;
            }
          }
        }
      }
    }
    else {
      return 2;
    }
  }
  return 0;
}

__host__ int cuda_task_error_code(const cudaTask_t* cuda_task)
/** Returns the current .task_error member variable. **/
{
  return cuda_task->task_error;
}

__host__ int cuda_task_dev_rsc_copy(const cudaTask_t* cuda_task, unsigned int arg_num, char which, talsh_dev_rsc_t* dev_rsc)
/** Clones the device resource object from a tensor argument of a CUDA task into <dev_rsc>:
    <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
  int errc;
  tensBlck_t* ctens;

  if (cuda_task == NULL) return -1;
  if (dev_rsc == NULL) return -2;
  if (arg_num >= cuda_task->num_args) return 1;
  ctens = cuda_task->tens_args[arg_num].tens_p;
  if (ctens) {
    switch (which) {
    case 's': errc = tensDevRsc_clone(ctens->src_rsc, dev_rsc); break;
    case 't': errc = tensDevRsc_clone(ctens->tmp_rsc, dev_rsc); break;
    case 'd': errc = tensDevRsc_clone(ctens->dst_rsc, dev_rsc); break;
    default: errc = 2;
    }
  }
  else {
    errc = 3;
  }
  return errc;
}

__host__ int cuda_task_dev_rsc_move(cudaTask_t* cuda_task, unsigned int arg_num, char which, talsh_dev_rsc_t* dev_rsc)
/** Moves the device resource object from a tensor argument of a CUDA task into <dev_rsc>:
    <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
  int errc;
  tensBlck_t* ctens;

  if (cuda_task == NULL) return -1;
  if (dev_rsc == NULL) return -2;
  if (arg_num >= cuda_task->num_args) return 1;
  ctens = cuda_task->tens_args[arg_num].tens_p;
  if (ctens) {
    switch (which) {
    case 's': errc = tensDevRsc_clone(ctens->src_rsc, dev_rsc); if (errc == 0) { free(ctens->src_rsc); ctens->src_rsc = NULL; } break;
    case 't': errc = tensDevRsc_clone(ctens->tmp_rsc, dev_rsc); if (errc == 0) { free(ctens->tmp_rsc); ctens->tmp_rsc = NULL; } break;
    case 'd': errc = tensDevRsc_clone(ctens->dst_rsc, dev_rsc); if (errc == 0) { free(ctens->dst_rsc); ctens->dst_rsc = NULL; } break;
    default: errc = 2;
    }
  }
  else {
    errc = 3;
  }
  return errc;
}

__host__ int cuda_task_arg_has_resource(cudaTask_t* cuda_task, unsigned int arg_num, char which, int* ierr)
/** Queries the existence of a CUDA task resource for tensor argument <arg_num>.
    <which> selects bewteen 's':source, 't':temporary, 'd':destination (resource). **/
{
  int ans;
  tensBlck_t* ctens;

  ans = NOPE; *ierr = 0;
  if (cuda_task == NULL) { *ierr = -1; return ans; }
  if (arg_num >= cuda_task->num_args) { *ierr = 1; return ans; }
  ctens = cuda_task->tens_args[arg_num].tens_p;
  if (ctens == NULL) { *ierr = 2; return ans; }
  switch (which) {
  case 's': if (ctens->src_rsc != NULL) ans = YEP; break;
  case 't': if (ctens->tmp_rsc != NULL) ans = YEP; break;
  case 'd': if (ctens->dst_rsc != NULL) ans = YEP; break;
  default: *ierr = 3;
  }
  return ans;
}

__host__ int cuda_task_arg_destroy(cudaTask_t* cuda_task, int arg_num) //internal use only
/** Destroys a specific <tensBlck_t> argument in a CUDA task. If <arg_num> is not
    specified (negative), all arguments of the CUDA task will be destroyed. **/
{
  int i, errc;

  errc = 0;
  if (cuda_task == NULL) return -1;
  if (arg_num >= cuda_task->num_args) return 1;
  if (arg_num < 0) { //destroy all tensor arguments
    while (cuda_task->num_args > 0) {
      i = tensBlck_destroy(cuda_task->tens_args[cuda_task->num_args - 1].tens_p);
      if ((i == 0 || i == NOT_CLEAN) && errc == 0) { errc = i; }
      else { errc = 2; }
      cuda_task->tens_args[--(cuda_task->num_args)].tens_p = NULL;
    }
  }
  else { //destroy a specific tensor argument
    i = tensBlck_destroy(cuda_task->tens_args[arg_num].tens_p);
    if ((i == 0 || i == NOT_CLEAN) && errc == 0) { errc = i; }
    else { errc = 3; }
    cuda_task->tens_args[arg_num].tens_p = NULL;
  }
  return errc;
}

__host__ float cuda_task_time(const cudaTask_t* cuda_task, float* in_copy, float* out_copy, float* comp, float* mmul)
/** Returns the time (in seconds) the CUDA task took to complete. Also, <in_copy> is the input copying time,
    <out_copy> is the output copying time, <comp> is the computing time, and <mmul> is the matrix
    multiplication time in seconds. A negative return value means an error occurred. **/
{
  int cur_gpu, errc;
  float time_ms;
  cudaEvent_t* evnt0_p, * evnt1_p, * evnt2_p, * evnt3_p;
#ifdef GPU_FINE_TIMING
  cudaEvent_t* evnt4_p, * evnt5_p;
#endif
  cudaError_t err;

  if (cuda_task != NULL) {
    if (cuda_task->task_error < 0) return -10.0f; //unfinished or empty task
    if (cuda_task->gpu_id < 0 || cuda_task->gpu_id >= MAX_GPUS_PER_NODE) return -9.0f;
    cur_gpu = gpu_in_focus(); if (cur_gpu < 0 || cur_gpu >= MAX_GPUS_PER_NODE) return -8.0f;
    errc = gpu_activate(cuda_task->gpu_id); if (errc != 0) return -7.0f;
    evnt0_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_start_hl); if (evnt0_p == NULL) return -6.0f;
    evnt1_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_comput_hl); if (evnt1_p == NULL) return -5.0f;
    evnt2_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_output_hl); if (evnt2_p == NULL) return -4.0f;
    evnt3_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_finish_hl); if (evnt3_p == NULL) return -3.0f;
#ifdef GPU_FINE_TIMING
    evnt4_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_mmbeg_hl); if (evnt4_p == NULL) return -2.0f;
    evnt5_p = cuda_event_ptr(cuda_task->gpu_id, cuda_task->event_mmend_hl); if (evnt5_p == NULL) return -1.0f;
#endif
    if (in_copy != NULL) {
      err = cudaEventElapsedTime(&time_ms, *evnt0_p, *evnt1_p); //time in miliseconds
      if (err == cudaSuccess) { *in_copy = time_ms / 1000.0f; }
      else { *in_copy = -1.0f; }
    }
    if (comp != NULL) {
      err = cudaEventElapsedTime(&time_ms, *evnt1_p, *evnt2_p); //time in miliseconds
      if (err == cudaSuccess) { *comp = time_ms / 1000.0f; }
      else { *comp = -1.0f; }
    }
    if (out_copy != NULL) {
      err = cudaEventElapsedTime(&time_ms, *evnt2_p, *evnt3_p); //time in miliseconds
      if (err == cudaSuccess) { *out_copy = time_ms / 1000.0f; }
      else { *out_copy = -1.0f; }
    }
#ifdef GPU_FINE_TIMING
    if (mmul != NULL) {
      err = cudaEventElapsedTime(&time_ms, *evnt4_p, *evnt5_p); //time in miliseconds
      if (err == cudaSuccess) { *mmul = time_ms / 1000.0f; }
      else { *mmul = -1.0f; }
    }
#endif
    err = cudaEventElapsedTime(&time_ms, *evnt0_p, *evnt3_p); //time in miliseconds
    if (err == cudaSuccess) { time_ms /= 1000.0f; }
    else { time_ms = -1.0f; } //time in seconds
    errc = gpu_activate(cur_gpu);
    return time_ms;
  }
  else {
    return -13.666f; //null task
  }
}

__host__ float cuda_task_time_(const cudaTask_t* cuda_task, float* in_copy, float* out_copy, float* comp, float* mmul)
{
  return cuda_task_time(cuda_task, in_copy, out_copy, comp, mmul);
}

void cuda_task_print(const cudaTask_t* cuda_task)
/** Prints CUDA task info. **/
{
  if (cuda_task != NULL) {
    printf("\n#MESSAGE: Printing CUDA task info:\n");
    printf(" CUDA task status             : %d\n", cuda_task->task_error);
    printf(" CUDA task GPU id             : %d\n", cuda_task->gpu_id);
    printf(" CUDA task stream handle      : %d\n", cuda_task->stream_hl);
    printf(" CUDA task event_start handle : %d\n", cuda_task->event_start_hl);
    printf(" CUDA task event_comput handle: %d\n", cuda_task->event_comput_hl);
    printf(" CUDA task event_output handle: %d\n", cuda_task->event_output_hl);
    printf(" CUDA task event_finish handle: %d\n", cuda_task->event_finish_hl);
#ifdef GPU_FINE_TIMING
    printf(" CUDA task event_mmbeg handle : %d\n", cuda_task->event_mmbeg_hl);
    printf(" CUDA task event_mmend handle : %d\n", cuda_task->event_mmend_hl);
#endif
    printf(" CUDA task coherence_var      : %u\n", cuda_task->coherence);
    printf(" CUDA task num_args           : %u\n", cuda_task->num_args);
    if (cuda_task->num_args <= MAX_TENSOR_OPERANDS) {
      for (int i = 0; i < cuda_task->num_args; ++i) {
        printf("  Tensor argument #%d address: %p\n", i, cuda_task->tens_args[i].tens_p);
        tensBlck_print(cuda_task->tens_args[i].tens_p);
      }
    }
    else {
      printf(" ERROR: Invalid number of arguments!!!\n");
    }
    printf("#END OF MESSAGE\n");
  }
  else {
    printf("\n#WARNING(tensor_algebra_gpu_nvidia:cuda_task_print): NULL pointer!\n");
  }
  return;
}

#endif
