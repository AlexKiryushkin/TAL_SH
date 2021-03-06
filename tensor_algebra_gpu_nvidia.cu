/** Tensor Algebra Library for NVidia GPU: NV-TAL (CUDA based).
AUTHOR: Dmitry I. Lyakh (Liakh): quant4me@gmail.com, liakhdi@ornl.gov
REVISION: 2020/04/12

Copyright (C) 2014-2020 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2020 Oak Ridge National Laboratory (UT-Battelle)

This file is part of ExaTensor.

ExaTensor is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ExaTensor is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
OPTIONS:
 # -D CUDA_ARCH=350: target GPU compute capability (default is 130);
 # -D NO_GPU: disables GPU usage;
 # -D NO_BLAS: disables cuBLAS calls, they will be replaced by in-house routines (slower);
 # -D USE_CUTT: enables an optimized tensor transpose via the cuTT library;
 # -D DEBUG_GPU: collection of debugging information will be activated;
NOTES:
 # Minimal required compute capability is 1.1 (1.3 for double precision).
 # cuBLAS.v2 is required when BLAS is enabled.
 # Non-blocking tensor algebra functions carry an additional output argument <cuda_task> (task handle).
 # Non-blocking tensor algebra functions carry an additional input argument <coherence_ctrl>
   which controls the tensor data consistency synchronization accross different devices
   after the tensor operation has completed successfully.
FOR DEVELOPERS ONLY:
 # Currently used device resources:
    - Global memory pointer (any device);
    - Argument buffer entry handle (any device);
    - Multi-index entry * (Host pinned memory, entry length = MAX_TENSOR_RANK);
    - GPU constant-memory entry handle (Nvidia GPU);
    - CUDA stream handle (Nvidia GPU);
    - CUDA event handle (Nvidia GPU).
 # A life cycle of a C object (for example, tensBlck_t):
    a) Allocate memory for the object itself, if needed: Suffix _alloc or _create (includes cleaning);
    b) Clean (initialize to null) an allocated (empty) object: Suffix _clean (normally included in _create);
    c) Construct (define or redefine) an existing object (resources will be acquired/released): Suffix _construct;
    d) Destruct a defined object (resources will be released, the object will be reset to clean): Suffix _destruct;
    e) Free the memory occupied by an object: Suffix _free or _destroy (may include _destruct, if needed).
   Thus, as a rule, the device resource acquisition/release occurs solely in _construct and _destruct functions.
 # A state of a C object:
    a) Undefined: After the memory allocation (either dynamic or static);
    b) Defined-empty (clean): After cleaning or destruction (dynamic object creation produces a clean object);
    c) Defined to a value (value-defined): After construction;
    d) Dead: After memory deallocation (if it was allocated dynamically).
 # Resource acquisition/release:
    - Tensor block constructor/destructor acquires/releases global memory resources, including
      both pointers and buffer entries, as well as multi-index bank entries (pinned Host memory).
    - CUDA task constructor/destructor acquires/releases CUDA resources (stream, events).
    - Tensor operation scheduling functions acquire GPU global memory resources,
      GPU constant memory resources, Host pinned multi-index entries.
    - CUDA task completion/error query functions release GPU global memory resources,
      GPU constant memory resources, Host pinned multi-index entries.
    - Coherence control is only applied to successfully finished CUDA tasks.
 # Functions which construct tensor blocks or perform asynchronous operations on them
   allocate resources (global/constant memory, etc). In case the corresponding resource
   allocator returns TRY_LATER or DEVICE_UNABLE (or an error), the corresponding function
   must clean the partially created tensor block or the CUDA task before returning:
   The corresponding object will be kept in its initial state if no SUCCESS.
 # Some CUDA kernels operating on two or more arguments assume no aliases
   for GPU pointers (__restrict__). Check each specific operation to see whether
   it is ok for the two tensor arguments to refer to the same tensor body.
TO BE FIXED:
 # In tensor operation scheduling functions, if a scheduling error occurs after
   the data transfer or CUDA kernel has been scheduled, the CUDA task finalization
   must not begin until the partially scheduled CUDA task has completed on GPU.
   Insert cudaStreamSynchronize in the finalization procedure.
 # Invoke cudaDeviceCanAccessPeer() in tensor operations to check whether
   two devices of the same kind can access each other's memory.
 # Account for implicit data transfers to/from peer GPUs in their statistics.
 # User-provided Alpha factors for gpu_tensor_block_contract() and
   gpu_tensor_block_add() reside on Host, thus requiring a slab in GPU
   memory (either global or constant) as a temporary for BLAS references.
**/

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "auxiliary_functions.h"
#include "tensor_algebra.h"
#include "device_algebra.h"
#include "mem_manager.h"
#include "talsh_complex.h"
#include "kernels.h"

//TENSOR BLOCK API:

#ifndef NO_GPU

//-------------------------------------------------
//EXPORTED FUNCTIONS (callable from C/C++/Fortran):
//---------------------------------------------------------------------------
// MATRIX MULTIPLICATION 'TN' (blocking, slow):
template <typename T>
__host__ int gpu_matrix_multiply_tn(size_t ll, size_t lr, size_t lc,
                                    const T * lmat, const T * rmat, T * dmat)
/** dmat(0:ll-1,0:lr-1)+=lmat(0:lc-1,0:ll-1)*rmat(0:lc-1,0:lr-1)
All matrices are in Host memory. Executed on the currently set GPU device. **/
{
 size_t dsize,lsize,rsize;
 T *dptr,*lptr,*rptr;
 int bx,by,err_code;
 const char *err_msg;
 cudaError_t err;

 if(lc > 0 && ll > 0 && lr > 0 && lmat != NULL && rmat != NULL && dmat != NULL){
  err=cudaGetLastError(); err=cudaSuccess;
  dsize=ll*lr*sizeof(T); lsize=lc*ll*sizeof(T); rsize=lc*lr*sizeof(T);
  err_code=gpu_mem_alloc((void**)&dptr,dsize); if(err_code != 0) return 1;
  err_code=gpu_mem_alloc((void**)&lptr,lsize); if(err_code != 0) return 2;
  err_code=gpu_mem_alloc((void**)&rptr,rsize); if(err_code != 0) return 3;
  err=cudaMemcpy((void*)dptr,(void*)dmat,dsize,cudaMemcpyHostToDevice); if(err != cudaSuccess) return 4;
  err=cudaMemcpy((void*)lptr,(void*)lmat,lsize,cudaMemcpyHostToDevice); if(err != cudaSuccess) return 5;
  err=cudaMemcpy((void*)rptr,(void*)rmat,rsize,cudaMemcpyHostToDevice); if(err != cudaSuccess) return 6;
  err_code=gpu_get_error_count();
  bx=1+(ll-1)/MAT_MULT_TILE_DIMX; by=1+(lr-1)/MAT_MULT_TILE_DIMY; limit_cuda_blocks2d(MAX_CUDA_BLOCKS,&bx,&by);
  dim3 blcks(bx,by); dim3 thrds(MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY);
  //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_matrix_multiply_tn): Running GPU kernel ..."); //debug
  gpu_matrix_multiply_tn__<<<blcks,thrds>>>(ll,lr,lc,lptr,rptr,dptr,(T)(1.0));
  err=cudaDeviceSynchronize(); if(err != cudaSuccess) return 7;
  err=cudaGetLastError();
  if(err!=cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_matrix_multiply_tn): Kernel error: %s\n",err_msg);
   return 8;
  }
  if(gpu_get_error_count() > err_code) return 9;
  //if(DEBUG) printf("Done: %d",err); //debug
  err=cudaMemcpy((void*)dmat,(void*)dptr,dsize,cudaMemcpyDeviceToHost); if(err != cudaSuccess) return 10;
  err=cudaDeviceSynchronize(); if(err != cudaSuccess) return 11;
  err_code=gpu_mem_free((void*)rptr); if(err_code != 0) return 12;
  err_code=gpu_mem_free((void*)lptr); if(err_code != 0) return 13;
  err_code=gpu_mem_free((void*)dptr); if(err_code != 0) return 14;
  err=cudaDeviceSynchronize(); if(err != cudaSuccess) return 15;
 }else{
  return 16;
 }
 return 0;
}
//-----------------------------------------------------------------------------------------------------------------------------
// TENSOR BODY CLONING (non-blocking):
__host__ int gpu_tensor_block_place(tensBlck_t *ctens, int gpu_id, unsigned int coh_ctrl, cudaTask_t *cuda_task, void *dev_mem)
/** Copies/moves the tensor body to a different GPU (gpu_id >= 0) or Host (gpu_id < 0).
    If <dev_mem> is a valid target device memory pointer, it will be used for storage, otherwise buffer memory will be allocated.
    A non-zero return status indicates an error. If the error code is negative, the CUDA task was not recorded.
    For positive error codes, the CUDA task was recorded. If the source device where the tensor body resides
    coincides with the destination device, no transfer will be scheduled.
    The source tensor body must reside either on Host or on Nvidia GPU. **/
{
 int j,tds,gpu_ex,src_gpu,devk,cur_gpu,devid,nclean,errc;
 size_t tvol,tsize;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
 cudaError_t err;
 const char *err_msg;

 errc=0; nclean=0;
 //Argument check:
 if(ctens == NULL) return -1;
 if(cuda_task == NULL) return -2;
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
 if(tens_valid_data_kind(ctens->data_kind,&tds) != YEP) return -4;
 if(tensBlck_present(ctens,DEV_NULL,DEV_NVIDIA_GPU) == YEP || tensBlck_present(ctens,DEV_NULL,DEV_HOST) == YEP){
  //Determine the id of the transfer executing GPU:
  cur_gpu=gpu_in_focus(); //save the current GPU
  gpu_ex=DEV_NULL; //executing GPU
  src_gpu=tensBlck_src_dev_id(ctens,&devk); //source GPU (or Host)
  if(devk == DEV_HOST){src_gpu=DEV_NULL;}else{if(devk != DEV_NVIDIA_GPU) return -5;} //src_gpu: source GPU (-1:Host)
  if(gpu_id >= 0 && gpu_id < MAX_GPUS_PER_NODE){ //destination is a GPU
   gpu_ex=gpu_id; if(gpu_is_mine(gpu_ex) <= GPU_OFF) return -6;
  }else if(gpu_id < 0){ //destination is Host
   if(src_gpu >= 0){gpu_ex=src_gpu; if(gpu_is_mine(gpu_ex) <= GPU_OFF) return -7;}
  }else{
   return -8; //invalid gpu_id
  }
  //Construct the CUDA task:
  if(gpu_ex < 0){ //Host-to-self transfer requested (no transfer)
   errc=cuda_task_construct(cuda_task);
  }else{ //Host-to-GPU, GPU-to-Host, GPU-to-GPU
   gpu_stats[gpu_ex].tasks_submitted++;
   //Check peer access if appropriate:
   if(src_gpu >= 0 && src_gpu != gpu_ex){
    err=cudaDeviceCanAccessPeer(&j,gpu_ex,src_gpu);
    if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
   }
   //Activate the transfer executing GPU:
   if(gpu_ex != cur_gpu){j=gpu_activate(gpu_ex); if(j){j=gpu_activate(cur_gpu); return -9;}} //activate the target GPU
   err=cudaGetLastError();
   if(err != cudaSuccess){
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Previous error detected: %s\n",cudaGetErrorString(err));
    ++nclean; err=cudaSuccess; //clear the GPU error status (sets NOT_CLEAN on exit)
   }
   errc=cuda_task_construct(cuda_task,gpu_ex); if(errc) j=gpu_activate(cur_gpu);
  }
  if(errc){if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -10;}}

  // *** From this point all error codes must be positive and the CUDA task must be recorded! ***
  //Set the CUDA task argument(s):
  errc=cuda_task_set_arg(cuda_task,0,ctens);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    j=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); j=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
    return errc;
   }else{
    j=cuda_task_record(cuda_task,coh_ctrl,1); j=gpu_activate(cur_gpu);
    return 1;
   }
  }
  //Determine the volume/size of the tensor block:
  tvol=tensBlck_volume(ctens); tsize=tvol*tds;
  if(tvol == 0){errc=cuda_task_record(cuda_task,coh_ctrl,2); errc=gpu_activate(cur_gpu); return 2;}
  //Associate CUDA stream and event pointers locally for convenience:
  cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
  if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,3); errc=gpu_activate(cur_gpu); return 3;}
  cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
  if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
  cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
  if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
  cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
  if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
  cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
  if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
  //Acquire global memory resources (destination resource):
  if(gpu_id >= 0){devid=encode_device_id(DEV_NVIDIA_GPU,gpu_id);}else{devid=encode_device_id(DEV_HOST,0);} //flat device id of the destination
  if(ctens->dst_rsc == ctens->src_rsc) ctens->dst_rsc=NULL;
  if(gpu_ex >= 0 && gpu_id != src_gpu){ //data is on a different GPU device or Host
   if(ctens->dst_rsc == NULL){
    errc=tensDevRsc_create(&(ctens->dst_rsc)); if(errc){j=cuda_task_record(cuda_task,coh_ctrl,8); j=gpu_activate(cur_gpu); return 8;}
   }else{
    if(tensDevRsc_is_empty(ctens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ctens->dst_rsc); if(errc) ++nclean;}
   }
   if(dev_mem == NULL){
    errc=tensDevRsc_allocate_mem(ctens->dst_rsc,devid,tsize,YEP); //device memory is allocated in the device argument buffer
   }else{
    errc=tensDevRsc_attach_mem(ctens->dst_rsc,devid,dev_mem); //externally provided device memory will be used for storage
   }
   if(errc){
    if(errc == TRY_LATER || errc == DEVICE_UNABLE){
     j=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); j=gpu_activate(cur_gpu);
     return errc;
    }else{
     j=cuda_task_record(cuda_task,coh_ctrl,9); j=gpu_activate(cur_gpu);
     return 9;
    }
   }
  }else{
   if(ctens->dst_rsc != NULL){
    if(tensDevRsc_is_empty(ctens->dst_rsc) == NOPE){j=tensDevRsc_release_all(ctens->dst_rsc); if(j) ++nclean;}
   }
   ctens->dst_rsc=ctens->src_rsc; //destination and source resources are the same (because the data is already on the executing GPU or Host)
  }
  //Record the start event:
  err=cudaEventRecord(*cuda_start,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the start event: %s\n",err_msg);
   j=cuda_task_record(cuda_task,coh_ctrl,10); j=gpu_activate(cur_gpu); return 10;
  }
  //Schedule the data transfer:
  if(gpu_ex >= 0 && gpu_id != src_gpu){
   //Make sure the data transfer does not begin before the data transfer from the previous task has finished:
   if(LastTask[gpu_ex] != NULL){ //`This should be done atomically for thread safety
    dep_event=cuda_event_ptr(LastTask[gpu_ex]->gpu_id,LastTask[gpu_ex]->event_comput_hl);
    err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
    if(err != cudaSuccess){
     err_msg=cudaGetErrorString(err);
     if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to create a task dependency: %s\n",err_msg);
     j=cuda_task_record(cuda_task,coh_ctrl,11); j=gpu_activate(cur_gpu); return 11;
    }
   }
   //Transfer:
   err=cudaMemcpyAsync(ctens->dst_rsc->gmem_p,ctens->src_rsc->gmem_p,tsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Tensor body transfer failed: %s\n",err_msg);
    j=cuda_task_record(cuda_task,coh_ctrl,12); j=gpu_activate(cur_gpu); return 12;
   }
   if(gpu_id >= 0){ //incoming traffic
    gpu_stats[gpu_ex].traffic_in+=tsize;
   }else{ //outgoing traffic (to Host)
    gpu_stats[gpu_ex].traffic_out+=tsize;
   }
  }
  //Record other events:
  err=cudaEventRecord(*cuda_comput,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the compute event: %s\n",err_msg);
   j=cuda_task_record(cuda_task,coh_ctrl,13); j=gpu_activate(cur_gpu); return 13;
  }
  err=cudaEventRecord(*cuda_output,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the output event: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,14); errc=gpu_activate(cur_gpu); return 14;
  }
  err=cudaEventRecord(*cuda_finish,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_place): Unable to record the finish event: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,15); errc=gpu_activate(cur_gpu); return 15;
  }
  //Record the successfully scheduled CUDA task, update the Last Task, and restore the original GPU:
  errc=cuda_task_record(cuda_task,coh_ctrl,0);
  if(gpu_ex >= 0 && gpu_ex != src_gpu) LastTask[gpu_ex]=cuda_task;
  if(gpu_ex >= 0 && gpu_ex != cur_gpu) j=gpu_activate(cur_gpu);
 }else{
  return -11; //tensor block is neither present on Host nor on any Nvidia GPU
 }
 if(nclean > 0 && errc == 0) errc=NOT_CLEAN;
 return errc;
}
//------------------------------------------------------------------------------------------
// TENSOR INITIALIZATION (non-blocking):
__host__ int gpu_tensor_block_init(tensBlck_t *dtens, double val_real, double val_imag,
                                   unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id)
/**
dtens(:)=scalar_value
INPUT:
 # (val_real,val_imag) - initialization value;
 # coh_ctrl - one of the COPY_X parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
OUTPUT:
 # dtens - initialized destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,tds_d,gpu_d,gpu_num,cur_gpu,targ_dev,bx,errc,stat;
 size_t vol_d,dsize;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg;
 float fval;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): GPU Tensor Initialization:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(dtens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP) return -2; //tensor block must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP) return -5; //tds_d: destination tensor element size in bytes
 if(dtens->data_kind <= 0) return -6; //tensor must have been previsously allocated with a certain data kind
 if(dtens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8; //source resource must be present (tensor body)
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#endif
 vol_d=tensBlck_volume(dtens); //tensor block volume
 dsize=vol_d*tds_d;            //tensor argument size in bytes
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu);
    return 12;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_init):\n");
  printf(" Const args (d)     : %d\n",cuda_task->tens_args[0].const_mem_entry); //debug
  printf(" Block sizes (d)    : %lu\n",dsize); //debug
  printf(" Block ranks (d)    : %d\n",dtens->shape.num_dim); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,23); errc=gpu_activate(cur_gpu); return 23;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,24); errc=gpu_activate(cur_gpu); return 24;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Destination tensor:
 if(cuda_task->tens_args[0].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(dtens->shape.dims),sizeof(int)*((size_t)drank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Destination tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,33); errc=gpu_activate(cur_gpu); return 33;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  if(gpu_d != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(dtens->dst_rsc->gmem_p,dtens->src_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Destination tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,35); errc=gpu_activate(cur_gpu); return 35;
   }
   gpu_stats[gpu_num].traffic_in+=dsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,36); errc=gpu_activate(cur_gpu); return 36;
 }
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
 }
//Destination tensor argument does not need transposing:
 darg=dtens->dst_rsc->gmem_p;
//Schedule the appropriate computation kernel:
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;
 }
#endif
// Initialization kernel:
 bx=1+(vol_d-1)/THRDS_ARRAY_INIT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
 switch(dtens->data_kind){
  case R4:
   fval=(float)val_real;
   gpu_array_init__<<<bx,THRDS_ARRAY_INIT,0,*cuda_stream>>>(vol_d,(float*)darg,fval);
   break;
  case R8:
   gpu_array_init__<<<bx,THRDS_ARRAY_INIT,0,*cuda_stream>>>(vol_d,(double*)darg,val_real);
   break;
  case C4:
   gpu_array_init__<<<bx,THRDS_ARRAY_INIT,0,*cuda_stream>>>(vol_d,(talshComplex4*)darg,
                                                            talshComplex4Set((float)val_real,(float)val_imag));
   break;
  case C8:
   gpu_array_init__<<<bx,THRDS_ARRAY_INIT,0,*cuda_stream>>>(vol_d,(talshComplex8*)darg,
                                                            talshComplex8Set(val_real,val_imag));
   break;
  default:
   errc=cuda_task_record(cuda_task,coh_ctrl,48); errc=gpu_activate(cur_gpu); return 48;
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
 }
#endif
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl)&(TWO_BITS_SET); //select bits 0,1 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_init): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,58); errc=gpu_activate(cur_gpu); return 58;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}
//-------------------------------------------------------------------------------------------------------------
// TENSOR SLICING (non-blocking):
__host__ int gpu_tensor_block_slice(tensBlck_t *ltens, tensBlck_t *dtens, const int *offsets,
                                    unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id, int accumulative)
{
 //`Implement
 printf("\n#FATAL(tensor_algebra_gpu_nvidia:gpu_tensor_block_slice): Operation not implemented!\n");
 return -1;
}
//--------------------------------------------------------------------------------------------------------------
// TENSOR INSERTION (non-blocking):
__host__ int gpu_tensor_block_insert(tensBlck_t *ltens, tensBlck_t *dtens, const int *offsets,
                                     unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id, int accumulative)
{
 //`Implement
 printf("\n#FATAL(tensor_algebra_gpu_nvidia:gpu_tensor_block_insert): Operation not implemented!\n");
 return -1;
}
//---------------------------------------------------------------------------------------------------------------
// TENSOR COPY/PERMUTATION (non-blocking):
__host__ int gpu_tensor_block_copy(const int *cptrn, tensBlck_t *ltens, tensBlck_t *dtens, unsigned int coh_ctrl,
                                   cudaTask_t *cuda_task, int gpu_id, int conj_bits)
/**
dtens(:)=ltens(:permuted)
INPUT:
 # cptrn(1:lrank) - permutation pattern (O2N): Position correspondence:
                    Uncontracted indices are positive, no contracted indices;
 # ltens - left tensor argument (initialized!);
 # dtens - destination tensor argument;
 # coh_ctrl - one of the COPY_XX parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
 # conj_bits - tensor argument complex conjugation bits, one bit per argument: {0:D,1:L};
OUTPUT:
 # dtens - updated destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,lrank,tds_d,tds_l,gpu_d,gpu_l,gpu_num,cur_gpu,targ_dev,bx,errc,stat,conj_l;
 int lprm[1+MAX_TENSOR_RANK];
 size_t vol_d,vol_l,dsize,lsize;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg,*larg;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;
#ifdef USE_CUTT
 cuttHandle cutt_d,cutt_l;
 cuttResult cutt_err;
#endif

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): GPU Tensor Copy:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(cptrn == NULL || dtens == NULL || ltens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP) return -2; //tensor blocks must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 lrank=(ltens->shape).num_dim; //left tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK ||
    lrank < 0 || lrank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP ||          //tds_d: destination tensor element size in bytes
    tens_valid_data_kind(ltens->data_kind,&tds_l) != YEP) return -5; //tds_l: left tensor element size in bytes
 if(!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind)) return -6; //data kind mismatch
 if(dtens->src_rsc == NULL || ltens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8; //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(ltens->src_rsc) != NOPE) return -9; //source resource must be present (tensor body)
//Check the contraction pattern and dimension extent correspondence:
 for(i=0;i<drank;i++) lprm[i]=0;
 for(i=0;i<lrank;i++){ //position in ltens
  j=cptrn[i];
  if(j > 0){ //position in dtens
   if(j > drank) return -11;
   if((dtens->shape).dims[j-1] != (ltens->shape).dims[i]) return -12;
   if(lprm[j-1] == 0){lprm[j-1]=1;}else{return -13;}
  }else{
   return -18;
  }
 }
 for(i=0;i<drank;i++) if(lprm[i] != 1) return -27;
//Check argument complex conjugation bits:
 conj_bits = conj_bits & 3; //keep only first two bits, one per tensor argument {0:D,1:L}
 if(conj_bits & 1){ //destination tensor argument conjugation = inverse conjugation of the left argument
  conj_bits = conj_bits ^ 3; //XOR with 0b11 will invert bits
 }
 conj_l = 0; if((conj_bits & 2) != 0) conj_l = 1; //left argument conjugation flag
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens,ltens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_l=decode_device_id(ltens->src_rsc->dev_id,&j); if(gpu_l < 0) return -30; //left tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_l != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_l); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_l=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
// Left argument:
 errc=cuda_task_set_arg(cuda_task,1,ltens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,2); i=gpu_activate(cur_gpu);
   return 2;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#endif
//Determine the volume and required matricization permutation for each tensor argument:
 for(i=0;i<drank;i++) cuda_task->tens_args[0].prmn_p[i]=(1+i); //trivial permutation
 for(i=0;i<lrank;i++) cuda_task->tens_args[1].prmn_p[i]=cptrn[i]; //required O2N permutation
 vol_d=tensBlck_volume(dtens); vol_l=tensBlck_volume(ltens); //tensor block volumes
 dsize=vol_d*tds_d; lsize=vol_l*tds_l; //tensor argument sizes in bytes
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu);
    return 12;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Left tensor:
 if(ltens->dst_rsc == ltens->src_rsc) ltens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_l != gpu_num){ //data is on a different GPU device or Host
  if(ltens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,13); i=gpu_activate(cur_gpu); return 13;}
  }else{
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->dst_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,14); i=gpu_activate(cur_gpu);
    return 14;
   }
  }
 }else{
  if(ltens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  ltens->dst_rsc=ltens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy):\n");
  printf(" Const args (d,l)   : %d %d\n",cuda_task->tens_args[0].const_mem_entry,
                                         cuda_task->tens_args[1].const_mem_entry); //debug
  printf(" Block sizes (d,l)  : %lu %lu\n",dsize,lsize); //debug
  printf(" Block ranks (d,l)  : %d %d\n",dtens->shape.num_dim,ltens->shape.num_dim); //debug
  printf(" Contraction pattern:"); for(i=0;i<(ltens->shape.num_dim);i++) printf(" %d",cptrn[i]); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,23); errc=gpu_activate(cur_gpu); return 23;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,24); errc=gpu_activate(cur_gpu); return 24;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Left tensor:
 if(cuda_task->tens_args[1].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(ltens->shape.dims),sizeof(int)*((size_t)lrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Left tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,25); errc=gpu_activate(cur_gpu); return 25;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[1].prmn_p),sizeof(int)*((size_t)lrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor permutation
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Left tensor prmn H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,26); errc=gpu_activate(cur_gpu); return 26;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  if(gpu_l != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(ltens->dst_rsc->gmem_p,ltens->src_rsc->gmem_p,lsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Left tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,27); errc=gpu_activate(cur_gpu); return 27;
   }
   gpu_stats[gpu_num].traffic_in+=lsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,28); errc=gpu_activate(cur_gpu); return 28;
 }
//Use the destination resource pointers for each tensor argument:
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
 }
 darg=dtens->dst_rsc->gmem_p;
 larg=ltens->dst_rsc->gmem_p;
//Schedule the appropriate computation kernel:
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;
 }
#endif
// Permutation kernel:
 if(TRANS_SHMEM == EFF_TRN_ON){
  bx=1+(vol_l-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(ltens->data_kind){
   case R4:
    gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(larg),(float*)(darg));
    break;
   case R8:
    gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(larg),(double*)(darg));
    break;
   case C4:
    gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
      (talshComplex4*)(larg),(talshComplex4*)(darg));
    break;
   case C8:
    gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
      (talshComplex8*)(larg),(talshComplex8*)(darg));
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,40); errc=gpu_activate(cur_gpu); return 40;
  }
 }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
  errc=prmn_convert(lrank,cuda_task->tens_args[1].prmn_p,lprm); for(i=0;i<lrank;++i) --(lprm[i]);
  cutt_err=cuttPlan(&cutt_l,lrank,(ltens->shape).dims,lprm,((size_t)tds_l),*cuda_stream);
  if(cutt_err == CUTT_SUCCESS){
   cutt_err=cuttExecute(cutt_l,larg,darg);
   if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;};
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
  }
#else
  errc=cuda_task_record(cuda_task,coh_ctrl,68); errc=gpu_activate(cur_gpu); return 68;
#endif
 }else if(TRANS_SHMEM == EFF_TRN_OFF){
  bx=1+(vol_l-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(ltens->data_kind){
   case R4:
    gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(larg),(float*)(darg));
    break;
   case R8:
    gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(larg),(double*)(darg));
    break;
   case C4:
    gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
      (talshComplex4*)(larg),(talshComplex4*)(darg));
    break;
   case C8:
    gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
     (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
      (talshComplex8*)(larg),(talshComplex8*)(darg));
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,41); errc=gpu_activate(cur_gpu); return 41;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,60); errc=gpu_activate(cur_gpu); return 60;
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
 }
#endif
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl>>2)&(TWO_BITS_SET); //select bits 2,3 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_copy): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,58); errc=gpu_activate(cur_gpu); return 58;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}
//-----------------------------------------------------------------------------------------------------------------------
// TENSOR ADDITION (non-blocking):
__host__ int gpu_tensor_block_add(const int *cptrn, tensBlck_t *ltens, tensBlck_t *dtens, unsigned int coh_ctrl,
                                  cudaTask_t *cuda_task, int gpu_id, double scale_real, double scale_imag, int conj_bits)
/**
dtens(:)+=ltens(:)*scalar
INPUT:
 # cptrn(1:lrank) - addition pattern: Position correspondence:
                    Uncontracted indices are positive, no contracted indices;
 # ltens - left tensor argument (initialized!);
 # dtens - destination tensor argument (initialized!);
 # coh_ctrl - one of the COPY_XX parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
 # scale_real - real part of the GEMM alpha coefficient (defaults to 1.0);
 # scale_imag - imaginary part of the GEMM alpha coefficient (defaults to 0.0);
 # conj_bits - tensor argument complex conjugation bits, one bit per argument: {0:D,1:L};
OUTPUT:
 # dtens - updated destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,lrank,tds_d,tds_l,gpu_d,gpu_l,perm_d,perm_l,ncd,nlu,nru,gpu_num,cur_gpu,targ_dev,bx,errc,stat,conj_l;
 int dprm[1+MAX_TENSOR_RANK],lprm[1+MAX_TENSOR_RANK],rprm[1]; //the 1st element is the sign of the permutation
 size_t vol_d,vol_l,dsize,lsize;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg,*larg;
 talshComplex4 scale_cmplx4;
 talshComplex8 scale_cmplx8;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;
#ifdef USE_CUTT
 cuttHandle cutt_d,cutt_l;
 cuttResult cutt_err;
#endif

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): GPU Tensor Addition:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(cptrn == NULL || dtens == NULL || ltens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP) return -2; //tensor blocks must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 lrank=(ltens->shape).num_dim; //left tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK ||
    lrank < 0 || lrank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP ||          //tds_d: destination tensor element size in bytes
    tens_valid_data_kind(ltens->data_kind,&tds_l) != YEP) return -5; //tds_l: left tensor element size in bytes
 if(!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind)) return -6; //data kind mismatch
 if(dtens->src_rsc == NULL || ltens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8; //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(ltens->src_rsc) != NOPE) return -9; //source resource must be present (tensor body)
//Check the contraction pattern and dimension extent correspondence:
 for(i=0;i<drank;i++) dprm[i]=0; for(i=0;i<lrank;i++) lprm[i]=0;
 for(i=0;i<lrank;i++){ //position in ltens
  j=cptrn[i];
  if(j > 0){ //position in dtens
   if(j > drank) return -11;
   if((dtens->shape).dims[j-1] != (ltens->shape).dims[i]) return -12;
   if(dprm[j-1] == 0){dprm[j-1]=1;}else{return -13;}
  }else{
   return -18;
  }
 }
 for(i=0;i<drank;i++) if(dprm[i] != 1) return -27;
//Check argument complex conjugation bits:
 conj_bits = conj_bits & 3; //keep only first two bits, one per tensor argument {0:D,1:L}
 if(conj_bits & 1){ //destination tensor argument conjugation = inverse conjugation of the left argument
  conj_bits = conj_bits ^ 3; //XOR with 0b11 will invert bits
 }
 conj_l = 0; if((conj_bits & 2) != 0) conj_l = 1; //left argument conjugation flag
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens,ltens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_l=decode_device_id(ltens->src_rsc->dev_id,&j); if(gpu_l < 0) return -30; //left tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_l != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_l); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_l=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
// Left argument:
 errc=cuda_task_set_arg(cuda_task,1,ltens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,2); i=gpu_activate(cur_gpu);
   return 2;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#endif
//Determine the volume and required matricization permutation for each tensor argument:
 get_contr_permutations(1,0,lrank,0,cptrn,0,dprm,lprm,rprm,&ncd,&nlu,&nru,&errc); //permutations and numbers of dimensions
 if(errc){i=cuda_task_record(cuda_task,coh_ctrl,9); i=gpu_activate(cur_gpu); return 9;}
 for(i=0;i<drank;i++) cuda_task->tens_args[0].prmn_p[i]=dprm[1+i]; //ignore the permutaion sign
 perm_d=non_trivial_prmn(drank,cuda_task->tens_args[0].prmn_p);    //trivial or not
 for(i=0;i<lrank;i++) cuda_task->tens_args[1].prmn_p[i]=lprm[1+i]; //ignore the permutaion sign
 perm_l=non_trivial_prmn(lrank,cuda_task->tens_args[1].prmn_p);    //trivial or not
 vol_d=tensBlck_volume(dtens); vol_l=tensBlck_volume(ltens);       //tensor block volumes
 dsize=vol_d*tds_d; lsize=vol_l*tds_l;                             //tensor argument sizes in bytes
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu);
    return 12;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Left tensor:
 if(ltens->dst_rsc == ltens->src_rsc) ltens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_l != gpu_num){ //data is on a different GPU device or Host
  if(ltens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,13); i=gpu_activate(cur_gpu); return 13;}
  }else{
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->dst_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,14); i=gpu_activate(cur_gpu);
    return 14;
   }
  }
 }else{
  if(ltens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  ltens->dst_rsc=ltens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
// Set up temporary memory resources in all tensors if needed (because of out-of-place tensor transpose):
//  Destination tensor:
 if(perm_d == YEP){
  if(dtens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,17); i=gpu_activate(cur_gpu); return 17;}
  }else{
   if(tensDevRsc_is_empty(dtens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->tmp_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,18); i=gpu_activate(cur_gpu);
    return 18;
   }
  }
 }
//  Left tensor:
 if(perm_l == YEP){
  if(ltens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,19); i=gpu_activate(cur_gpu); return 19;}
  }else{
   if(tensDevRsc_is_empty(ltens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->tmp_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,20); i=gpu_activate(cur_gpu);
    return 20;
   }
  }
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_add):\n");
  printf(" Const args (d,l)   : %d %d\n",cuda_task->tens_args[0].const_mem_entry,
                                         cuda_task->tens_args[1].const_mem_entry); //debug
  printf(" Block sizes (d,l)  : %lu %lu\n",dsize,lsize); //debug
  printf(" Block ranks (d,l)  : %d %d\n",dtens->shape.num_dim,ltens->shape.num_dim); //debug
  printf(" Contraction pattern:"); for(i=0;i<(ltens->shape.num_dim);i++) printf(" %d",cptrn[i]); //debug
  printf("\n Contr/uncontr/lens : %d %d %d",ncd,nlu,nru); //debug
  printf("\n D-permutation      :"); for(i=0;i<dtens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[0].prmn_p[i]); //debug
  printf("\n L-permutation      :"); for(i=0;i<ltens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[1].prmn_p[i]); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,23); errc=gpu_activate(cur_gpu); return 23;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,24); errc=gpu_activate(cur_gpu); return 24;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Left tensor:
 if(cuda_task->tens_args[1].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(ltens->shape.dims),sizeof(int)*((size_t)lrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Left tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,25); errc=gpu_activate(cur_gpu); return 25;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  if(perm_l == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[1].prmn_p),sizeof(int)*((size_t)lrank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Left tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,26); errc=gpu_activate(cur_gpu); return 26;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  }
  if(gpu_l != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(ltens->dst_rsc->gmem_p,ltens->src_rsc->gmem_p,lsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Left tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,27); errc=gpu_activate(cur_gpu); return 27;
   }
   gpu_stats[gpu_num].traffic_in+=lsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,28); errc=gpu_activate(cur_gpu); return 28;
 }
// Destination tensor:
 if(cuda_task->tens_args[0].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(dtens->shape.dims),sizeof(int)*((size_t)drank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,33); errc=gpu_activate(cur_gpu); return 33;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  if(perm_d == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[0].prmn_p),sizeof(int)*((size_t)drank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,34); errc=gpu_activate(cur_gpu); return 34;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  }
  if(gpu_d != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(dtens->dst_rsc->gmem_p,dtens->src_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,35); errc=gpu_activate(cur_gpu); return 35;
   }
   gpu_stats[gpu_num].traffic_in+=dsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,36); errc=gpu_activate(cur_gpu); return 36;
 }
//Schedule tensor transposes if needed:
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
 }
// Destination tensor transpose (it should not happen actually):
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,38); errc=gpu_activate(cur_gpu); return 38;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   for(i=0;i<drank;++i) dprm[i]=cuda_task->tens_args[0].prmn_p[i]-1;
   cutt_err=cuttPlan(&cutt_d,drank,(dtens->shape).dims,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->dst_rsc->gmem_p,dtens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,63); errc=gpu_activate(cur_gpu); return 63;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,64); errc=gpu_activate(cur_gpu); return 64;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,65); errc=gpu_activate(cur_gpu); return 65;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,39); errc=gpu_activate(cur_gpu); return 39;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,59); errc=gpu_activate(cur_gpu); return 59;
  }
  darg=dtens->tmp_rsc->gmem_p;
 }else{
  darg=dtens->dst_rsc->gmem_p;
 }
// Left tensor transpose:
 if(perm_l == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,40); errc=gpu_activate(cur_gpu); return 40;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(lrank,cuda_task->tens_args[1].prmn_p,lprm); for(i=0;i<lrank;++i) --(lprm[i]);
   cutt_err=cuttPlan(&cutt_l,lrank,(ltens->shape).dims,lprm,((size_t)tds_l),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_l,ltens->dst_rsc->gmem_p,ltens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,68); errc=gpu_activate(cur_gpu); return 68;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,41); errc=gpu_activate(cur_gpu); return 41;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,60); errc=gpu_activate(cur_gpu); return 60;
  }
  larg=ltens->tmp_rsc->gmem_p;
 }else{
  larg=ltens->dst_rsc->gmem_p;
 }
//Schedule the appropriate computation kernel:
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,66); errc=gpu_activate(cur_gpu); return 66;
 }
#endif
// Addition kernel:
 bx=1+(vol_d-1)/THRDS_ARRAY_ADD; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
 switch(dtens->data_kind){
  case R4:
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(float*)darg,(float*)larg,(float)scale_real);
   gpu_stats[gpu_num].flops+=2.0*((double)(dsize)); //1 mul, 1 add SP
   break;
  case R8:
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(double*)darg,(double*)larg,scale_real);
   gpu_stats[gpu_num].flops+=2.0*((double)(dsize)); //1 mul, 1 add DP
   break;
  case C4:
   scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex4*)darg,(talshComplex4*)larg,scale_cmplx4,conj_l);
   gpu_stats[gpu_num].flops+=8.0*((double)(dsize)); //4 mul, 4 add SP
   break;
  case C8:
   scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
   gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex8*)darg,(talshComplex8*)larg,scale_cmplx8,conj_l);
   gpu_stats[gpu_num].flops+=8.0*((double)(dsize)); //4 mul, 4 add DP
   break;
  default:
   errc=cuda_task_record(cuda_task,coh_ctrl,48); errc=gpu_activate(cur_gpu); return 48;
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
 }
#endif
//Schedule the inverse tensor transpose for the destination tensor (should not happen actually):
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,54); errc=gpu_activate(cur_gpu); return 54;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(drank,cuda_task->tens_args[0].prmn_p,dprm); for(i=0;i<drank;++i) --(dprm[i]);
   for(i=0;i<drank;++i) rprm[i]=(dtens->shape).dims[drank-i-1]; //inversed dimension order
   cutt_err=cuttPlan(&cutt_d,drank,rprm,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->tmp_rsc->gmem_p,dtens->dst_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,63); errc=gpu_activate(cur_gpu); return 63;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,64); errc=gpu_activate(cur_gpu); return 64;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,65); errc=gpu_activate(cur_gpu); return 65;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,55); errc=gpu_activate(cur_gpu); return 55;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,62); errc=gpu_activate(cur_gpu); return 62;
  }
 }
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl>>2)&(TWO_BITS_SET); //select bits 2,3 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_add): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,58); errc=gpu_activate(cur_gpu); return 58;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}
//-------------------------------------------------------------------------------------------------------------------
// TENSOR CONTRACTION (non-blocking):
__host__ int gpu_tensor_block_contract_dlf(const int *cptrn, tensBlck_t *ltens, tensBlck_t *rtens, tensBlck_t *dtens,
                                           unsigned int coh_ctrl, cudaTask_t *cuda_task, int gpu_id,
                                           double scale_real, double scale_imag, int conj_bits, int accumulative)
/**
dtens(:)+=ltens(:)*rtens(:)
INPUT:
 # cptrn(1:lrank+rrank) - contraction pattern: Position correspondence:
                          Uncontracted indices are positive, contracted are negative;
 # ltens - left tensor argument (initialized!);
 # rtens - right tensor argument (initialized!);
 # dtens - destination tensor argument (initialized!);
 # coh_ctrl - one of the COPY_XXX parameters regulating the data presence for each tensor argument;
 # cuda_task - pointer to an empty (clean) CUDA task;
 # gpu_id - suggested GPU ID on which the operation is to be scheduled (-1: defaults to the optimal one);
 # scale_real - real part of the GEMM alpha coefficient (defaults to 1.0);
 # scale_imag - imaginary part of the GEMM alpha coefficient (defaults to 0.0);
 # conj_bits - tensor argument complex conjugation bits, one bit per argument: {0:D,1:L,2:R};
 # accumulative - accumulate in (default) VS overwrite destination tensor: [YEP|NOPE];
OUTPUT:
 # dtens - updated destination tensor;
 # cuda_task - recorded CUDA task (either successfully scheduled or failed).
NOTES:
 # If the tensor operation has been scheduled successfully, a recorded (active) CUDA task
   will be returned along with zero return status. A scheduling error results in either
   a negative (at early stages) or positive (at later stages) return status. In the former case
   the CUDA task is left clean, while at the latter case it will be recorded as failed (error).
 # Special return statuses TRY_LATER and DEVICE_UNABLE are not errors but merely indicators
   of the current or permanent lack of resources, respectively. However, the CUDA task status
   in these cases will still be set to an error (always check the function return status!).
 # If <gpu_id> is out of the legitimate GPU range, it will be replaced by an optimal one,
   based on argument residence and the current load of GPU(s).
**/
{
 int i,j,drank,lrank,rrank,tds_d,tds_l,tds_r,gpu_d,gpu_l,gpu_r,perm_d,perm_l,perm_r;
 int ncd,nlu,nru,gpu_num,cur_gpu,targ_dev,bx,by,errc,stat,conj_l,conj_r,fast_math;
 int dprm[1+MAX_TENSOR_RANK],lprm[1+MAX_TENSOR_RANK],rprm[1+MAX_TENSOR_RANK]; //the 1st element is the sign of the permutation
 size_t vol_d,vol_l,vol_r,dsize,lsize,rsize,lc,ll,lr,pofs;
 unsigned int coh;
 const unsigned int TWO_BITS_SET = 3; //two right bits are set
 void *darg,*larg,*rarg,*alpha_plus_p,*alpha_minus_p,*beta_p,*beta_one_p;
 talshComplex4 scale_cmplx4;
 talshComplex8 scale_cmplx8;
 cudaStream_t *cuda_stream;
 cudaEvent_t *cuda_start,*cuda_comput,*cuda_output,*cuda_finish,*dep_event;
#ifdef GPU_FINE_TIMING
 cudaEvent_t *cuda_mmbeg,*cuda_mmend;
#endif
 cudaError_t err;
 const char *err_msg;
#ifndef NO_BLAS
 cublasStatus_t err_cublas;
 cublasOperation_t left_conj,right_conj;
#endif
#ifdef USE_CUTT
 cuttHandle cutt_d,cutt_l,cutt_r;
 cuttResult cutt_err;
#endif
#ifdef USE_CUTENSOR
 cutensorStatus_t err_cutensor;
 cutensorContractionPlan_t plan_cudesc;
 cutensorContractionFind_t find_cudesc;
 cutensorContractionDescriptor_t contr_cudesc;
 uint32_t align_d,align_l,align_r;
 int32_t cumod_d[MAX_TENSOR_RANK],cumod_l[MAX_TENSOR_RANK],cumod_r[MAX_TENSOR_RANK];
#endif

 //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): GPU Tensor Contraction:\n"); //debug
 stat=0; //return status in case of successful scheduling
//Check function arguments:
 if(cptrn == NULL || dtens == NULL || ltens == NULL || rtens == NULL || cuda_task == NULL) return -1;
 if(tensBlck_present(dtens) != YEP || tensBlck_present(ltens) != YEP || tensBlck_present(rtens) != YEP) return -2; //tensor blocks must reside in some device memory
 if(cuda_task_gpu_id(cuda_task) >= 0) return -3; //CUDA task is not clean (destruct/clean it first)
//Check tensor arguments:
 drank=(dtens->shape).num_dim; //destination tensor rank
 lrank=(ltens->shape).num_dim; //left tensor rank
 rrank=(rtens->shape).num_dim; //right tensor rank
 if(drank < 0 || drank > MAX_TENSOR_RANK ||
    lrank < 0 || lrank > MAX_TENSOR_RANK ||
    rrank < 0 || rrank > MAX_TENSOR_RANK) return -4;
 if(tens_valid_data_kind(dtens->data_kind,&tds_d) != YEP ||          //tds_d: destination tensor element size in bytes
    tens_valid_data_kind(ltens->data_kind,&tds_l) != YEP ||          //tds_l: left tensor element size in bytes
    tens_valid_data_kind(rtens->data_kind,&tds_r) != YEP) return -5; //tds_r: right tensor element size in bytes
 if(!(dtens->data_kind > 0 && ltens->data_kind == dtens->data_kind && rtens->data_kind == dtens->data_kind)) return -6; //data kind mismatch
 if(dtens->src_rsc == NULL || ltens->src_rsc == NULL || rtens->src_rsc == NULL) return -7; //source resource must always be present
 if(tensDevRsc_is_empty(dtens->src_rsc) != NOPE) return -8;  //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(ltens->src_rsc) != NOPE) return -9;  //source resource must be present (tensor body)
 if(tensDevRsc_is_empty(rtens->src_rsc) != NOPE) return -10; //source resource must be present (tensor body)
//Check the contraction pattern and tensor dimension extent correspondence:
 for(i=0;i<drank;i++) dprm[i]=0; for(i=0;i<lrank;i++) lprm[i]=0; for(i=0;i<rrank;i++) rprm[i]=0;
 for(i=0;i<lrank;i++){ //position in ltens
  j=cptrn[i];
  if(j > 0){ //position in dtens
   if(j > drank) return -11;
   if((dtens->shape).dims[j-1] != (ltens->shape).dims[i]) return -12;
   if(dprm[j-1] == 0){dprm[j-1]=1;}else{return -13;}
  }else if(j < 0){ //position in rtens
   if(-j > rrank) return -14;
   if((rtens->shape).dims[-j-1] != (ltens->shape).dims[i]) return -15;
   if(cptrn[lrank+(-j-1)] != -(i+1)) return -16;
   if(rprm[-j-1] == 0){rprm[-j-1]=1;}else{return -17;}
  }else{
   return -18;
  }
 }
 for(i=0;i<rrank;i++){ //position in rtens
  j=cptrn[lrank+i];
  if(j > 0){ //position in dtens
   if(j > drank) return -19;
   if((dtens->shape).dims[j-1] != (rtens->shape).dims[i]) return -20;
   if(dprm[j-1] == 0){dprm[j-1]=1;}else{return -21;}
  }else if(j < 0){ //position in ltens
   if(-j > lrank) return -22;
   if((ltens->shape).dims[-j-1] != (rtens->shape).dims[i]) return -23;
   if(cptrn[-j-1] != -(i+1)) return -24;
   if(lprm[-j-1] == 0){lprm[-j-1]=1;}else{return -25;}
  }else{
   return -26;
  }
 }
 for(i=0;i<drank;i++) if(dprm[i] != 1) return -27;
#ifdef USE_CUTENSOR
 if(get_contr_pattern_cutensor(cptrn,drank,cumod_d,lrank,cumod_l,rrank,cumod_r) != 0) return -27;
#endif
//Check argument complex conjugation bits:
#ifndef NO_BLAS
 left_conj=CUBLAS_OP_T; right_conj=CUBLAS_OP_N; //default is TN GEMM
#endif
 conj_bits = conj_bits & 7; //keep only first three bits, one per tensor argument {0:D,1:L,2:R}
 if(conj_bits & 1){ //destination tensor argument conjugation = inverse conjugation of left and right tensor arguments
  conj_bits = conj_bits ^ 7; //XOR with 0b111 will invert bits
 }
 if(dtens->data_kind == C4 || dtens->data_kind == C8){ //conjugation may apply to complex data kinds
  conj_l = 0; if((conj_bits & 2) != 0) conj_l=1; //left tensor argument conjugation flag
  conj_r = 0; if((conj_bits & 4) != 0) conj_r=1; //right tensor argument conjugation flag
#ifndef NO_BLAS
  if(conj_l != 0) left_conj = CUBLAS_OP_C;
  if(conj_r != 0) right_conj = CUBLAS_OP_C;
#endif
 }else{
  conj_bits = 0; conj_l = 0; conj_r = 0; //no conjugation for real data kinds
 }
//Activate the right GPU:
 if(gpu_id < 0 || gpu_id >= MAX_GPUS_PER_NODE){gpu_num=tens_op_best_gpu(dtens,ltens,rtens);}else{gpu_num=gpu_id;}
 if(gpu_is_mine(gpu_num) <= GPU_OFF) return -28; //GPU is not mine or error
 gpu_stats[gpu_num].tasks_submitted++;
 gpu_d=decode_device_id(dtens->src_rsc->dev_id,&j); if(gpu_d < 0) return -29; //destination tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_d != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_d); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_d=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_l=decode_device_id(ltens->src_rsc->dev_id,&j); if(gpu_l < 0) return -30; //left tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_l != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_l); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_l=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 gpu_r=decode_device_id(rtens->src_rsc->dev_id,&j); if(gpu_r < 0) return -31; //right tensor source device id
 if(j == DEV_NVIDIA_GPU){
  if(gpu_r != gpu_num){
   err=cudaDeviceCanAccessPeer(&j,gpu_num,gpu_r); if(err != cudaSuccess || j == 0) return DEVICE_UNABLE; //peer access impossible for this GPU device
  }
 }else if(j == DEV_HOST){
  gpu_r=-1; //data is in Host memory
 }else{
  return DEVICE_UNABLE; //data is not in Host or GPU memory
 }
 cur_gpu=gpu_in_focus(); //save the current GPU
 if(gpu_num != cur_gpu){errc=gpu_activate(gpu_num); if(errc){errc=gpu_activate(cur_gpu); return -32;}} //activate the target GPU
 err=cudaGetLastError(); err=cudaSuccess; //clear the GPU error status
 targ_dev=encode_device_id(DEV_NVIDIA_GPU,gpu_num); //flat device id
//Construct a CUDA task (acquire CUDA resources) for the target GPU:
 errc=cuda_task_construct(cuda_task,gpu_num);
 if(errc){i=gpu_activate(cur_gpu); if(errc == TRY_LATER || errc == DEVICE_UNABLE){return errc;}else{return -33;}}

// *** From this point all error codes must be positive and the CUDA task must be recorded! ***
//Set up tensor arguments (allocates additional resources for each tensor argument):
// Destination argument:
 errc=cuda_task_set_arg(cuda_task,0,dtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,1); i=gpu_activate(cur_gpu);
   return 1;
  }
 }
// Left argument:
 errc=cuda_task_set_arg(cuda_task,1,ltens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,2); i=gpu_activate(cur_gpu);
   return 2;
  }
 }
// Right argument:
 errc=cuda_task_set_arg(cuda_task,2,rtens);
 if(errc){
  if(errc == TRY_LATER || errc == DEVICE_UNABLE){
   i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu); //not an error if TRY_LATER or DEVICE_UNABLE are returned
   return errc;
  }else{
   i=cuda_task_record(cuda_task,coh_ctrl,3); i=gpu_activate(cur_gpu);
   return 3;
  }
 }
//Associate CUDA stream and event pointers locally for convenience:
 cuda_stream=cuda_stream_ptr(cuda_task->gpu_id,cuda_task->stream_hl);
 if(cuda_stream == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,4); errc=gpu_activate(cur_gpu); return 4;}
 cuda_start=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_start_hl);
 if(cuda_start == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,5); errc=gpu_activate(cur_gpu); return 5;}
 cuda_comput=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_comput_hl);
 if(cuda_comput == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,6); errc=gpu_activate(cur_gpu); return 6;}
 cuda_output=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_output_hl);
 if(cuda_output == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,7); errc=gpu_activate(cur_gpu); return 7;}
 cuda_finish=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_finish_hl);
 if(cuda_finish == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,8); errc=gpu_activate(cur_gpu); return 8;}
#ifdef GPU_FINE_TIMING
 cuda_mmbeg=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmbeg_hl);
 if(cuda_mmbeg == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,9); errc=gpu_activate(cur_gpu); return 9;}
 cuda_mmend=cuda_event_ptr(cuda_task->gpu_id,cuda_task->event_mmend_hl);
 if(cuda_mmend == NULL){errc=cuda_task_record(cuda_task,coh_ctrl,10); errc=gpu_activate(cur_gpu); return 10;}
#endif
//Determine the volume and required matricization permutation for each tensor argument:
 if(drank > 0 && lrank > 0 && rrank > 0 && drank < (lrank + rrank)){ //GEMM mapped tensor contraction: {TN,NT,NN,TT}
  get_contr_permutations(1,0,lrank,rrank,cptrn,conj_bits,dprm,lprm,rprm,&ncd,&nlu,&nru,&errc); //permutations and numbers of dimensions
 }else{ //custom kernel mapped tensor contraction (complex conjugation does not require modified permutations)
  get_contr_permutations(1,0,lrank,rrank,cptrn,0,dprm,lprm,rprm,&ncd,&nlu,&nru,&errc); //permutations and numbers of dimensions
 }
// Get permutations:
 if(errc){i=cuda_task_record(cuda_task,coh_ctrl,11); i=gpu_activate(cur_gpu); return 11;}
 for(i=0;i<drank;i++) cuda_task->tens_args[0].prmn_p[i]=dprm[1+i]; //ignore the permutaion sign
 perm_d=non_trivial_prmn(drank,cuda_task->tens_args[0].prmn_p);    //trivial or not
 for(i=0;i<lrank;i++) cuda_task->tens_args[1].prmn_p[i]=lprm[1+i]; //ignore the permutaion sign
 perm_l=non_trivial_prmn(lrank,cuda_task->tens_args[1].prmn_p);    //trivial or not
 for(i=0;i<rrank;i++) cuda_task->tens_args[2].prmn_p[i]=rprm[1+i]; //ignore the permutaion sign
 perm_r=non_trivial_prmn(rrank,cuda_task->tens_args[2].prmn_p);    //trivial or not
// Get tensor volumes, sizes and matrix attributes:
 vol_d=tensBlck_volume(dtens); vol_l=tensBlck_volume(ltens); vol_r=tensBlck_volume(rtens); //tensor block volumes
 lc=1; ll=1;
 for(i=0;i<lrank;i++){
  if(cuda_task->tens_args[1].prmn_p[i] <= ncd){lc*=((ltens->shape).dims[i]);}else{ll*=((ltens->shape).dims[i]);}
 }
 lr=vol_d/ll;
 if(vol_l <= 0 || vol_r <= 0 || vol_d <= 0 || vol_d%ll != 0 || vol_r%lr != 0 || vol_r/lr != lc){
  i=cuda_task_record(cuda_task,coh_ctrl,12); i=gpu_activate(cur_gpu); return 12; //invalid matrix dimensions obtained
 }
 dsize=vol_d*tds_d; lsize=vol_l*tds_l; rsize=vol_r*tds_r; //tensor argument sizes in bytes
// Check fast math requirements:
 fast_math=NOPE;
#ifdef USE_CUTENSOR
 if(DISABLE_BLAS == 0 && gpu_is_mine(gpu_num) >= GPU_MINE_CUBLAS){
  if(drank > 0 && lrank > 0 && rrank > 0){ //`Remove this restriction
   perm_d=NOPE; perm_l=NOPE; perm_r=NOPE; //cuTensor does not need tensor permutations
  }
 }
#else
 if(gpu_query_fast_math(gpu_num) == YEP){
  if(dtens->data_kind == R4 || dtens->data_kind == C4){ //`Will require extension if new hardware
   if(lr%WMMA_ALIGN == 0 && ll%WMMA_ALIGN == 0 && lc%WMMA_ALIGN == 0){
    if(TRANS_SHMEM == EFF_TRN_ON){
     if(dtens->data_kind == C4 || dtens->data_kind == C8){ //complex data types will require real/imag split
      if(scale_real == 1.0 && scale_imag == 0.0){ //`Lift this restriction in future (requires better handling of prefactors)
       perm_d=YEP; perm_l=YEP; perm_r=YEP;
       fast_math=YEP;
      }
     }else{
      fast_math=YEP;
     }
    }
   }
  }
 }
#endif
//Acquire global memory resources for tensor arguments if needed:
// Set up destination memory resources in all tensors:
//  Destination tensor:
 if(dtens->dst_rsc == dtens->src_rsc) dtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_d != gpu_num){ //data is on a different GPU device or Host
  if(dtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,13); i=gpu_activate(cur_gpu); return 13;}
  }else{
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->dst_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,14); i=gpu_activate(cur_gpu);
    return 14;
   }
  }
 }else{
  if(dtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(dtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  dtens->dst_rsc=dtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Left tensor:
 if(ltens->dst_rsc == ltens->src_rsc) ltens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_l != gpu_num){ //data is on a different GPU device or Host
  if(ltens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,15); i=gpu_activate(cur_gpu); return 15;}
  }else{
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->dst_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,16); i=gpu_activate(cur_gpu);
    return 16;
   }
  }
 }else{
  if(ltens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(ltens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  ltens->dst_rsc=ltens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
//  Right tensor:
 if(rtens->dst_rsc == rtens->src_rsc) rtens->dst_rsc = NULL; //destination resource was pointing to the source resource
 if(gpu_r != gpu_num){ //data is on a different GPU device or Host
  if(rtens->dst_rsc == NULL){
   errc=tensDevRsc_create(&(rtens->dst_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,17); i=gpu_activate(cur_gpu); return 17;}
  }else{
   if(tensDevRsc_is_empty(rtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(rtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(rtens->dst_rsc,targ_dev,rsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,18); i=gpu_activate(cur_gpu);
    return 18;
   }
  }
 }else{
  if(rtens->dst_rsc != NULL){
   if(tensDevRsc_is_empty(rtens->dst_rsc) == NOPE){errc=tensDevRsc_release_all(rtens->dst_rsc); if(errc) stat=NOT_CLEAN;}
  }
  rtens->dst_rsc=rtens->src_rsc; //destination and source resources are the same (because the data is already on the computing GPU)
 }
// Set up temporary memory resources in all tensors if needed (because of out-of-place tensor transpose):
//  Destination tensor:
 if(perm_d == YEP){
  if(dtens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(dtens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,19); i=gpu_activate(cur_gpu); return 19;}
  }else{
   if(tensDevRsc_is_empty(dtens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(dtens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(dtens->tmp_rsc,targ_dev,dsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,20); i=gpu_activate(cur_gpu);
    return 20;
   }
  }
 }
//  Left tensor:
 if(perm_l == YEP){
  if(ltens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(ltens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,21); i=gpu_activate(cur_gpu); return 21;}
  }else{
   if(tensDevRsc_is_empty(ltens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(ltens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(ltens->tmp_rsc,targ_dev,lsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,22); i=gpu_activate(cur_gpu);
    return 22;
   }
  }
 }
//  Right tensor:
 if(perm_r == YEP){
  if(rtens->tmp_rsc == NULL){
   errc=tensDevRsc_create(&(rtens->tmp_rsc)); if(errc){i=cuda_task_record(cuda_task,coh_ctrl,23); i=gpu_activate(cur_gpu); return 23;}
  }else{
   if(tensDevRsc_is_empty(rtens->tmp_rsc) == NOPE){errc=tensDevRsc_release_all(rtens->tmp_rsc); if(errc) stat=NOT_CLEAN;}
  }
  errc=tensDevRsc_allocate_mem(rtens->tmp_rsc,targ_dev,rsize,YEP);
  if(errc){
   if(errc == TRY_LATER || errc == DEVICE_UNABLE){
    i=cuda_task_record(cuda_task,coh_ctrl,NVTAL_DEFERRED); i=gpu_activate(cur_gpu);
    return errc;
   }else{
    i=cuda_task_record(cuda_task,coh_ctrl,24); i=gpu_activate(cur_gpu);
    return 24;
   }
  }
 }
#ifdef DEBUG_GPU
//DEBUG begin:
 if(DEBUG){
  printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf):\n");
  printf(" Const args (d,l,r) : %d %d %d\n",cuda_task->tens_args[0].const_mem_entry,
                                            cuda_task->tens_args[1].const_mem_entry,
                                            cuda_task->tens_args[2].const_mem_entry); //debug
  printf(" Block sizes (d,l,r): %lu %lu %lu\n",dsize,lsize,rsize); //debug
  printf(" Block ranks (d,l,r): %d %d %d\n",dtens->shape.num_dim,ltens->shape.num_dim,rtens->shape.num_dim); //debug
  printf(" Contraction pattern:"); for(i=0;i<(ltens->shape.num_dim+rtens->shape.num_dim);i++) printf(" %d",cptrn[i]); //debug
  printf("\n Contr/uncontr/lens : %d %d %d: %lu %lu %lu\n",ncd,nlu,nru,lc,ll,lr); //debug
  printf(" D-permutation      :"); for(i=0;i<dtens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[0].prmn_p[i]); //debug
  printf("\n L-permutation      :"); for(i=0;i<ltens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[1].prmn_p[i]); //debug
  printf("\n R-permutation      :"); for(i=0;i<rtens->shape.num_dim;i++) printf(" %d",cuda_task->tens_args[2].prmn_p[i]); //debug
  printf("\n#END OF DEBUG\n");
 }
//DEBUG end.
#endif /*DEBUG_GPU*/
//Start scheduling CUDA calls:
 err=cudaEventRecord(*cuda_start,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the start event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,25); errc=gpu_activate(cur_gpu); return 25;
 }
 if(LastTask[gpu_num] != NULL){ //`This should be done atomically for thread safety
  dep_event=cuda_event_ptr(LastTask[gpu_num]->gpu_id,LastTask[gpu_num]->event_comput_hl);
  err=cudaStreamWaitEvent(*cuda_stream,*dep_event,0); //input transfers should only begin after the previous task input transfers have completed
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to create a task dependency: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,26); errc=gpu_activate(cur_gpu); return 26;
  }
 }
//Schedule forward data transfers for all tensors if needed:
// Left tensor:
 if(cuda_task->tens_args[1].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(ltens->shape.dims),sizeof(int)*((size_t)lrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Left tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,27); errc=gpu_activate(cur_gpu); return 27;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  if(perm_l == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[1].prmn_p),sizeof(int)*((size_t)lrank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[1].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Left tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,28); errc=gpu_activate(cur_gpu); return 28;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)lrank);
  }
  if(gpu_l != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(ltens->dst_rsc->gmem_p,ltens->src_rsc->gmem_p,lsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Left tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,29); errc=gpu_activate(cur_gpu); return 29;
   }
   gpu_stats[gpu_num].traffic_in+=lsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,30); errc=gpu_activate(cur_gpu); return 30;
 }
// Right tensor:
 if(cuda_task->tens_args[2].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(rtens->shape.dims),sizeof(int)*((size_t)rrank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[2].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Right tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,31); errc=gpu_activate(cur_gpu); return 31;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)rrank);
  if(perm_r == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[2].prmn_p),sizeof(int)*((size_t)rrank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[2].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Right tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,32); errc=gpu_activate(cur_gpu); return 32;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)rrank);
  }
  if(gpu_r != gpu_num){ //data is not on the computing GPU
   err=cudaMemcpyAsync(rtens->dst_rsc->gmem_p,rtens->src_rsc->gmem_p,rsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Right tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,33); errc=gpu_activate(cur_gpu); return 33;
   }
   gpu_stats[gpu_num].traffic_in+=rsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,34); errc=gpu_activate(cur_gpu); return 34;
 }
// Destination tensor:
 if(cuda_task->tens_args[0].const_mem_entry >= 0){ //GPU constant memory entry will contain tensor dimension extents and the matricization permutation (if any)
  err=cudaMemcpyToSymbolAsync(const_args_dims,(void*)(dtens->shape.dims),sizeof(int)*((size_t)drank),
                sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor dimension extents
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor dims H2D copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,35); errc=gpu_activate(cur_gpu); return 35;
  }
  gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  if(perm_d == YEP){
   err=cudaMemcpyToSymbolAsync(const_args_prmn,(void*)(cuda_task->tens_args[0].prmn_p),sizeof(int)*((size_t)drank),
                 sizeof(int)*((size_t)(MAX_TENSOR_RANK*(cuda_task->tens_args[0].const_mem_entry))),cudaMemcpyHostToDevice,*cuda_stream); //tensor matricization permutation
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor prmn H2D copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,36); errc=gpu_activate(cur_gpu); return 36;
   }
   gpu_stats[gpu_num].traffic_in+=sizeof(int)*((size_t)drank);
  }
  if(gpu_d != gpu_num && accumulative != NOPE){ //data is not on the computing GPU
   err=cudaMemcpyAsync(dtens->dst_rsc->gmem_p,dtens->src_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor body copy failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,37); errc=gpu_activate(cur_gpu); return 37;
   }
   gpu_stats[gpu_num].traffic_in+=dsize;
  }
 }else{
  errc=cuda_task_record(cuda_task,coh_ctrl,38); errc=gpu_activate(cur_gpu); return 38;
 }
//Schedule tensor transposes if needed:
// Record a CUDA event:
 err=cudaEventRecord(*cuda_comput,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the compute event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,39); errc=gpu_activate(cur_gpu); return 39;
 }
// Destination tensor transpose:
 if(perm_d == YEP){
  if(accumulative != NOPE){
   if(TRANS_SHMEM == EFF_TRN_ON){
    bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
    switch(dtens->data_kind){
     case R4:
      gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
      break;
     case R8:
      gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
      break;
     case C4:
      if(fast_math == YEP){
       gpu_tensor_block_copy_cmplx_split_out_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
        (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
         (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
      }else{
       gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
        (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
         (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
      }
      break;
     case C8:
      gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
        (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
      break;
     default:
      errc=cuda_task_record(cuda_task,coh_ctrl,40); errc=gpu_activate(cur_gpu); return 40;
    }
   }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
    for(i=0;i<drank;++i) dprm[i]=cuda_task->tens_args[0].prmn_p[i]-1;
    cutt_err=cuttPlan(&cutt_d,drank,(dtens->shape).dims,dprm,((size_t)tds_d),*cuda_stream);
    if(cutt_err == CUTT_SUCCESS){
     cutt_err=cuttExecute(cutt_d,dtens->dst_rsc->gmem_p,dtens->tmp_rsc->gmem_p);
     if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,41); errc=gpu_activate(cur_gpu); return 41;};
    }else{
     errc=cuda_task_record(cuda_task,coh_ctrl,42); errc=gpu_activate(cur_gpu); return 42;
    }
#else
    errc=cuda_task_record(cuda_task,coh_ctrl,43); errc=gpu_activate(cur_gpu); return 43;
#endif
   }else if(TRANS_SHMEM == EFF_TRN_OFF){
    bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
    switch(dtens->data_kind){
     case R4:
      gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->dst_rsc->gmem_p),(float*)(dtens->tmp_rsc->gmem_p));
      break;
     case R8:
      gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->dst_rsc->gmem_p),(double*)(dtens->tmp_rsc->gmem_p));
      break;
     case C4:
      gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
        (talshComplex4*)(dtens->dst_rsc->gmem_p),(talshComplex4*)(dtens->tmp_rsc->gmem_p));
      break;
     case C8:
      gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
       (0,1,drank,cuda_task->tens_args[0].const_mem_entry,
        (talshComplex8*)(dtens->dst_rsc->gmem_p),(talshComplex8*)(dtens->tmp_rsc->gmem_p));
      break;
     default:
      errc=cuda_task_record(cuda_task,coh_ctrl,44); errc=gpu_activate(cur_gpu); return 44;
    }
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,45); errc=gpu_activate(cur_gpu); return 45;
   }
  }else{
   err=cudaMemsetAsync(dtens->tmp_rsc->gmem_p,0,dsize,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor memset failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,46); errc=gpu_activate(cur_gpu); return 46;
   }
  }
  darg=dtens->tmp_rsc->gmem_p;
 }else{
  if(accumulative == NOPE){
   err=cudaMemsetAsync(dtens->dst_rsc->gmem_p,0,dsize,*cuda_stream);
   if(err != cudaSuccess){
    err_msg=cudaGetErrorString(err);
    if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor memset failed: %s\n",err_msg);
    errc=cuda_task_record(cuda_task,coh_ctrl,47); errc=gpu_activate(cur_gpu); return 47;
   }
  }
  darg=dtens->dst_rsc->gmem_p;
 }
// Left tensor transpose:
 if(perm_l == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     if(fast_math == YEP){
      gpu_tensor_block_copy_cmplx_split_out_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
        (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     }else{
      gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
        (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     }
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,48); errc=gpu_activate(cur_gpu); return 48;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(lrank,cuda_task->tens_args[1].prmn_p,lprm); for(i=0;i<lrank;++i) --(lprm[i]);
   cutt_err=cuttPlan(&cutt_l,lrank,(ltens->shape).dims,lprm,((size_t)tds_l),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_l,ltens->dst_rsc->gmem_p,ltens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,49); errc=gpu_activate(cur_gpu); return 49;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,50); errc=gpu_activate(cur_gpu); return 50;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,51); errc=gpu_activate(cur_gpu); return 51;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_l-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(ltens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(float*)(ltens->dst_rsc->gmem_p),(float*)(ltens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,(double*)(ltens->dst_rsc->gmem_p),(double*)(ltens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex4*)(ltens->dst_rsc->gmem_p),(talshComplex4*)(ltens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,lrank,cuda_task->tens_args[1].const_mem_entry,
       (talshComplex8*)(ltens->dst_rsc->gmem_p),(talshComplex8*)(ltens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,52); errc=gpu_activate(cur_gpu); return 52;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,53); errc=gpu_activate(cur_gpu); return 53;
  }
  larg=ltens->tmp_rsc->gmem_p;
 }else{
  larg=ltens->dst_rsc->gmem_p;
 }
// Right tensor transpose:
 if(perm_r == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_r-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(rtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(float*)(rtens->dst_rsc->gmem_p),(float*)(rtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(double*)(rtens->dst_rsc->gmem_p),(double*)(rtens->tmp_rsc->gmem_p));
     break;
    case C4:
     if(fast_math == YEP){
      gpu_tensor_block_copy_cmplx_split_out_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
        (talshComplex4*)(rtens->dst_rsc->gmem_p),(talshComplex4*)(rtens->tmp_rsc->gmem_p));
     }else{
      gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
        (talshComplex4*)(rtens->dst_rsc->gmem_p),(talshComplex4*)(rtens->tmp_rsc->gmem_p));
     }
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex8*)(rtens->dst_rsc->gmem_p),(talshComplex8*)(rtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,54); errc=gpu_activate(cur_gpu); return 54;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(rrank,cuda_task->tens_args[2].prmn_p,rprm); for(i=0;i<rrank;++i) --(rprm[i]);
   cutt_err=cuttPlan(&cutt_r,rrank,(rtens->shape).dims,rprm,((size_t)tds_r),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_r,rtens->dst_rsc->gmem_p,rtens->tmp_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,55); errc=gpu_activate(cur_gpu); return 55;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,56); errc=gpu_activate(cur_gpu); return 56;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,57); errc=gpu_activate(cur_gpu); return 57;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_r-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(rtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(float*)(rtens->dst_rsc->gmem_p),(float*)(rtens->tmp_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,(double*)(rtens->dst_rsc->gmem_p),(double*)(rtens->tmp_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex4*)(rtens->dst_rsc->gmem_p),(talshComplex4*)(rtens->tmp_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (0,0,rrank,cuda_task->tens_args[2].const_mem_entry,
       (talshComplex8*)(rtens->dst_rsc->gmem_p),(talshComplex8*)(rtens->tmp_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,58); errc=gpu_activate(cur_gpu); return 58;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,59); errc=gpu_activate(cur_gpu); return 59;
  }
  rarg=rtens->tmp_rsc->gmem_p;
 }else{
  rarg=rtens->dst_rsc->gmem_p;
 }
//Schedule the appropriate computation kernel:
// Set up the scaling prefactor (in mapped Host memory):
 errc=0;
 switch(dtens->data_kind){
  case R4:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex4Set((float)scale_real,(float)scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,60); j=gpu_activate(cur_gpu); return 60;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_plus_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_plus_p,sgemm_alpha_plus); if(err != cudaSuccess) errc++;
    err=cudaGetSymbolAddress(&alpha_minus_p,sgemm_alpha_minus); if(err != cudaSuccess) errc++;
   }
   if(accumulative == NOPE){
    err=cudaGetSymbolAddress(&beta_p,sgemm_beta_zero); if(err != cudaSuccess) errc++;
   }else{
    err=cudaGetSymbolAddress(&beta_p,sgemm_beta_one); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_one_p,sgemm_beta_one); if(err != cudaSuccess) errc++;
   break;
  case R8:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex8Set(scale_real,scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,61); j=gpu_activate(cur_gpu); return 61;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_plus_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_plus_p,dgemm_alpha_plus); if(err != cudaSuccess) errc++;
    err=cudaGetSymbolAddress(&alpha_minus_p,dgemm_alpha_minus); if(err != cudaSuccess) errc++;
   }
   if(accumulative == NOPE){
    err=cudaGetSymbolAddress(&beta_p,dgemm_beta_zero); if(err != cudaSuccess) errc++;
   }else{
    err=cudaGetSymbolAddress(&beta_p,dgemm_beta_one); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_one_p,dgemm_beta_one); if(err != cudaSuccess) errc++;
   break;
  case C4:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex4Set((float)scale_real,(float)scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,62); j=gpu_activate(cur_gpu); return 62;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_plus_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_plus_p,cgemm_alpha_plus); if(err != cudaSuccess) errc++;
    err=cudaGetSymbolAddress(&alpha_minus_p,cgemm_alpha_minus); if(err != cudaSuccess) errc++;
   }
   if(accumulative == NOPE){
    err=cudaGetSymbolAddress(&beta_p,cgemm_beta_zero); if(err != cudaSuccess) errc++;
   }else{
    err=cudaGetSymbolAddress(&beta_p,cgemm_beta_one); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_one_p,cgemm_beta_one); if(err != cudaSuccess) errc++;
   break;
  case C8:
   if(scale_real != 1.0 || scale_imag != 0.0){
    errc=cuda_task_set_prefactor(cuda_task,talshComplex8Set(scale_real,scale_imag));
    if(errc){j=cuda_task_record(cuda_task,coh_ctrl,63); j=gpu_activate(cur_gpu); return 63;}
    j=slab_get_entry_offset(&prefactors,cuda_task->pref_ptr,&pofs); if(j != 0) errc++;
    alpha_plus_p=(void*)&(((char*)(gpu_prefs_base_ptr))[pofs]);
   }else{
    err=cudaGetSymbolAddress(&alpha_plus_p,zgemm_alpha_plus); if(err != cudaSuccess) errc++;
    err=cudaGetSymbolAddress(&alpha_minus_p,zgemm_alpha_minus); if(err != cudaSuccess) errc++;
   }
   if(accumulative == NOPE){
    err=cudaGetSymbolAddress(&beta_p,zgemm_beta_zero); if(err != cudaSuccess) errc++;
   }else{
    err=cudaGetSymbolAddress(&beta_p,zgemm_beta_one); if(err != cudaSuccess) errc++;
   }
   err=cudaGetSymbolAddress(&beta_one_p,zgemm_beta_one); if(err != cudaSuccess) errc++;
   break;
  default:
   errc++;
 }
 if(errc){errc=cuda_task_record(cuda_task,coh_ctrl,64); errc=gpu_activate(cur_gpu); return 64;}
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmbeg,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the mmbeg event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,65); errc=gpu_activate(cur_gpu); return 65;
 }
#endif
// Scalar multiplication:
 if(drank == 0 && lrank == 0 && rrank == 0){
  switch(dtens->data_kind){
   case R4:
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((float*)larg,(float*)rarg,(float*)darg,(float)scale_real);
    break;
   case R8:
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((double*)larg,(double*)rarg,(double*)darg,scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((talshComplex4*)larg,(talshComplex4*)rarg,(talshComplex4*)darg,
                                                  scale_cmplx4,conj_l,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_scalar_multiply__<<<1,1,0,*cuda_stream>>>((talshComplex8*)larg,(talshComplex8*)rarg,(talshComplex8*)darg,
                                                  scale_cmplx8,conj_l,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,67); errc=gpu_activate(cur_gpu); return 67;
  }
// Right tensor rescaled addition:
 }else if(lrank == 0 && rrank > 0){
  bx=1+(vol_d-1)/THRDS_ARRAY_ADD; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(dtens->data_kind){
   case R4:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(float*)(darg),(float*)(rarg),(float*)(larg),(float)scale_real);
    break;
   case R8:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(double*)(darg),(double*)(rarg),(double*)(larg),scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex4*)(darg),(talshComplex4*)(rarg),
                                                           (talshComplex4*)(larg),scale_cmplx4,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex8*)(darg),(talshComplex8*)(rarg),
                                                           (talshComplex8*)(larg),scale_cmplx8,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,68); errc=gpu_activate(cur_gpu); return 68;
  }
// Left tensor rescaled addition:
 }else if(lrank > 0 && rrank == 0){
  bx=1+(vol_d-1)/THRDS_ARRAY_ADD; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(dtens->data_kind){
   case R4:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(float*)(darg),(float*)(larg),(float*)(rarg),(float)scale_real);
    break;
   case R8:
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(double*)(darg),(double*)(larg),(double*)(rarg),scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex4*)(darg),(talshComplex4*)(larg),
                                                           (talshComplex4*)(rarg),scale_cmplx4,conj_l);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_add__<<<bx,THRDS_ARRAY_ADD,0,*cuda_stream>>>(vol_d,(talshComplex8*)(darg),(talshComplex8*)(larg),
                                                           (talshComplex8*)(rarg),scale_cmplx8,conj_l);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,69); errc=gpu_activate(cur_gpu); return 69;
  }
// Full tensor contraction (via vector dot-product):
 }else if(drank == 0 && lrank > 0 && rrank == lrank){
  bx=1+(vol_l-1)/THRDS_ARRAY_SCALE; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
  switch(ltens->data_kind){
   case R4:
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(float),*cuda_stream>>>
                             (vol_l,(float*)larg,(float*)rarg,(float*)darg,(float)scale_real);
    break;
   case R8:
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(double),*cuda_stream>>>
                             (vol_l,(double*)larg,(double*)rarg,(double*)darg,scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(talshComplex4),*cuda_stream>>>
                             (vol_l,(talshComplex4*)larg,(talshComplex4*)rarg,(talshComplex4*)darg,scale_cmplx4,conj_l,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_dot_product__<<<bx,THRDS_ARRAY_SCALE,THRDS_ARRAY_SCALE*sizeof(talshComplex8),*cuda_stream>>>
                             (vol_l,(talshComplex8*)larg,(talshComplex8*)rarg,(talshComplex8*)darg,scale_cmplx8,conj_l,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,70); errc=gpu_activate(cur_gpu); return 70;
  }
// Tensor product (no contracted indices):
 }else if(drank > 0 && drank == lrank + rrank){
  bx=1+(vol_l-1)/THRDS_ARRAY_PRODUCT; by=1+(vol_r-1)/THRDS_ARRAY_PRODUCT;
  limit_cuda_blocks2d(MAX_CUDA_BLOCKS,&bx,&by); dim3 blcks(bx,by);
  switch(dtens->data_kind){
   case R4:
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(float*)larg,vol_r,(float*)rarg,(float*)darg,(float)scale_real);
    break;
   case R8:
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(double*)larg,vol_r,(double*)rarg,(double*)darg,scale_real);
    break;
   case C4:
    scale_cmplx4 = talshComplex4Set((float)scale_real,(float)scale_imag);
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(talshComplex4*)larg,vol_r,(talshComplex4*)rarg,(talshComplex4*)darg,scale_cmplx4,conj_l,conj_r);
    break;
   case C8:
    scale_cmplx8 = talshComplex8Set(scale_real,scale_imag);
    gpu_array_product__<<<blcks,THRDS_ARRAY_PRODUCT,0,*cuda_stream>>>
                          (vol_l,(talshComplex8*)larg,vol_r,(talshComplex8*)rarg,(talshComplex8*)darg,scale_cmplx8,conj_l,conj_r);
    break;
   default:
    errc=cuda_task_record(cuda_task,coh_ctrl,71); errc=gpu_activate(cur_gpu); return 71;
  }
// Partial tensor contraction (via TN matrix multiplication):
 }else{
#ifndef NO_BLAS
  if(DISABLE_BLAS == 0 && gpu_is_mine(gpu_num) >= GPU_MINE_CUBLAS){ //BLAS is enabled
   err_cublas=cublasSetStream(cublas_handle[gpu_num],*cuda_stream);
   if(err_cublas != CUBLAS_STATUS_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,72); errc=gpu_activate(cur_gpu); return 72;}
#ifdef USE_CUTENSOR
   err_cutensor=cutensorGetAlignmentRequirement(&(cutensor_handle[gpu_num]),(const void*)darg,
                                                &(cuda_task->tens_cudesc[0]),&align_d);
   if(err_cutensor == CUTENSOR_STATUS_SUCCESS){
    err_cutensor=cutensorGetAlignmentRequirement(&(cutensor_handle[gpu_num]),(const void*)larg,
                                                 &(cuda_task->tens_cudesc[1]),&align_l);
    if(err_cutensor == CUTENSOR_STATUS_SUCCESS){
     err_cutensor=cutensorGetAlignmentRequirement(&(cutensor_handle[gpu_num]),(const void*)rarg,
                                                  &(cuda_task->tens_cudesc[2]),&align_r);
    }
   }
   if(err_cutensor != CUTENSOR_STATUS_SUCCESS){
    if(VERBOSE){
     err_msg=cutensorGetErrorString(err_cutensor);
     if(err_msg != NULL) printf("#ERROR(gpu_tensor_block_contract_dlf): cuTensor error: %s\n",err_msg);
    }
    errc=cuda_task_record(cuda_task,coh_ctrl,72); errc=gpu_activate(cur_gpu); return 72;
   }
#endif
   switch(dtens->data_kind){
    case R4:
#ifdef USE_CUTENSOR
     if(cuda_task->pref_ptr == NULL) cuda_task->pref_ptr=&h_sgemm_beta_one;
     err_cutensor=cutensorInitContractionDescriptor(&(cutensor_handle[gpu_num]),&contr_cudesc,
                                                    &(cuda_task->tens_cudesc[1]),cumod_l,align_l,
                                                    &(cuda_task->tens_cudesc[2]),cumod_r,align_r,
                                                    &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                    &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                    CUTENSOR_R_MIN_32F);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorInitContractionFind(&(cutensor_handle[gpu_num]),&find_cudesc,CUTENSOR_ALGO_DEFAULT);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorInitContractionPlan(&(cutensor_handle[gpu_num]),&plan_cudesc,&contr_cudesc,&find_cudesc,
                                              (uint64_t)(cutensor_worksize[gpu_num]));
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorContraction(&(cutensor_handle[gpu_num]),&plan_cudesc,
                                      cuda_task->pref_ptr,larg,rarg,
                                      (const void*)&h_sgemm_beta_one,darg,darg,
                                      cutensor_workspace[gpu_num],(uint64_t)(cutensor_worksize[gpu_num]),
                                      *cuda_stream);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     if(cuda_task->pref_ptr == &h_sgemm_beta_one) cuda_task->pref_ptr=NULL;
#else
     err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                (float*)alpha_plus_p,(float*)larg,(int)lc,(float*)rarg,(int)lc,(float*)beta_p,(float*)darg,(int)ll);
#endif
     break;
    case R8:
#ifdef USE_CUTENSOR
     if(cuda_task->pref_ptr == NULL) cuda_task->pref_ptr=&h_dgemm_beta_one;
     err_cutensor=cutensorInitContractionDescriptor(&(cutensor_handle[gpu_num]),&contr_cudesc,
                                                    &(cuda_task->tens_cudesc[1]),cumod_l,align_l,
                                                    &(cuda_task->tens_cudesc[2]),cumod_r,align_r,
                                                    &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                    &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                    CUTENSOR_R_MIN_64F);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorInitContractionFind(&(cutensor_handle[gpu_num]),&find_cudesc,CUTENSOR_ALGO_DEFAULT);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorInitContractionPlan(&(cutensor_handle[gpu_num]),&plan_cudesc,&contr_cudesc,&find_cudesc,
                                              (uint64_t)(cutensor_worksize[gpu_num]));
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorContraction(&(cutensor_handle[gpu_num]),&plan_cudesc,
                                      cuda_task->pref_ptr,larg,rarg,
                                      (const void*)&h_dgemm_beta_one,darg,darg,
                                      cutensor_workspace[gpu_num],(uint64_t)(cutensor_worksize[gpu_num]),
                                      *cuda_stream);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     if(cuda_task->pref_ptr == &h_dgemm_beta_one) cuda_task->pref_ptr=NULL;
#else
     err_cublas=cublasDgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                (double*)alpha_plus_p,(double*)larg,(int)lc,(double*)rarg,(int)lc,(double*)beta_p,(double*)darg,(int)ll);
#endif
     break;
    case C4:
     if(fast_math == YEP){
      if(conj_r){
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_plus_p,&(((float*)larg)[0]),(int)lc,&(((float*)rarg)[0]),(int)lr,(float*)beta_p,
                  &(((float*)darg)[0]),(int)ll);
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_minus_p,&(((float*)larg)[vol_l]),(int)lc,&(((float*)rarg)[vol_r]),(int)lr,(float*)beta_one_p,
                  &(((float*)darg)[0]),(int)ll);
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_plus_p,&(((float*)larg)[vol_l]),(int)lc,&(((float*)rarg)[0]),(int)lr,(float*)beta_p,
                  &(((float*)darg)[vol_d]),(int)ll);
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_plus_p,&(((float*)larg)[0]),(int)lc,&(((float*)rarg)[vol_r]),(int)lr,(float*)beta_one_p,
                  &(((float*)darg)[vol_d]),(int)ll);
      }else{
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_plus_p,&(((float*)larg)[0]),(int)lc,&(((float*)rarg)[0]),(int)lc,(float*)beta_p,
                  &(((float*)darg)[0]),(int)ll);
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_minus_p,&(((float*)larg)[vol_l]),(int)lc,&(((float*)rarg)[vol_r]),(int)lc,(float*)beta_one_p,
                  &(((float*)darg)[0]),(int)ll);
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_plus_p,&(((float*)larg)[vol_l]),(int)lc,&(((float*)rarg)[0]),(int)lc,(float*)beta_p,
                  &(((float*)darg)[vol_d]),(int)ll);
       err_cublas=cublasSgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (float*)alpha_plus_p,&(((float*)larg)[0]),(int)lc,&(((float*)rarg)[vol_r]),(int)lc,(float*)beta_one_p,
                  &(((float*)darg)[vol_d]),(int)ll);
      }
     }else{
#ifdef USE_CUTENSOR
      if(cuda_task->pref_ptr == NULL) cuda_task->pref_ptr=&h_cgemm_beta_one;
      err_cutensor=cutensorInitContractionDescriptor(&(cutensor_handle[gpu_num]),&contr_cudesc,
                                                     &(cuda_task->tens_cudesc[1]),cumod_l,align_l,
                                                     &(cuda_task->tens_cudesc[2]),cumod_r,align_r,
                                                     &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                     &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                     CUTENSOR_C_MIN_32F);
      if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
      err_cutensor=cutensorInitContractionFind(&(cutensor_handle[gpu_num]),&find_cudesc,CUTENSOR_ALGO_DEFAULT);
      if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
      err_cutensor=cutensorInitContractionPlan(&(cutensor_handle[gpu_num]),&plan_cudesc,&contr_cudesc,&find_cudesc,
                                               (uint64_t)(cutensor_worksize[gpu_num]));
      if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
      err_cutensor=cutensorContraction(&(cutensor_handle[gpu_num]),&plan_cudesc, //`Missing complex conjugation for arguments
                                       cuda_task->pref_ptr,larg,rarg,
                                       (const void*)&h_cgemm_beta_one,darg,darg,
                                       cutensor_workspace[gpu_num],(uint64_t)(cutensor_worksize[gpu_num]),
                                       *cuda_stream);
      if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
      if(cuda_task->pref_ptr == &h_cgemm_beta_one) cuda_task->pref_ptr=NULL;
#else
      if(conj_r){
       err_cublas=cublasCgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (talshComplex4*)alpha_plus_p,(talshComplex4*)larg,(int)lc,(talshComplex4*)rarg,(int)lr,(talshComplex4*)beta_p,
                  (talshComplex4*)darg,(int)ll);
      }else{
       err_cublas=cublasCgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                  (talshComplex4*)alpha_plus_p,(talshComplex4*)larg,(int)lc,(talshComplex4*)rarg,(int)lc,(talshComplex4*)beta_p,
                  (talshComplex4*)darg,(int)ll);
      }
#endif
     }
     break;
    case C8:
#ifdef USE_CUTENSOR
     if(cuda_task->pref_ptr == NULL) cuda_task->pref_ptr=&h_zgemm_beta_one;
     err_cutensor=cutensorInitContractionDescriptor(&(cutensor_handle[gpu_num]),&contr_cudesc,
                                                    &(cuda_task->tens_cudesc[1]),cumod_l,align_l,
                                                    &(cuda_task->tens_cudesc[2]),cumod_r,align_r,
                                                    &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                    &(cuda_task->tens_cudesc[0]),cumod_d,align_d,
                                                    CUTENSOR_C_MIN_64F);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorInitContractionFind(&(cutensor_handle[gpu_num]),&find_cudesc,CUTENSOR_ALGO_DEFAULT);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorInitContractionPlan(&(cutensor_handle[gpu_num]),&plan_cudesc,&contr_cudesc,&find_cudesc,
                                              (uint64_t)(cutensor_worksize[gpu_num]));
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     err_cutensor=cutensorContraction(&(cutensor_handle[gpu_num]),&plan_cudesc, //`Missing complex conjugation for arguments
                                      cuda_task->pref_ptr,larg,rarg,
                                      (const void*)&h_zgemm_beta_one,darg,darg,
                                      cutensor_workspace[gpu_num],(uint64_t)(cutensor_worksize[gpu_num]),
                                      *cuda_stream);
     if(err_cutensor != CUTENSOR_STATUS_SUCCESS) break;
     if(cuda_task->pref_ptr == &h_zgemm_beta_one) cuda_task->pref_ptr=NULL;
#else
     if(conj_r){
      err_cublas=cublasZgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                 (talshComplex8*)alpha_plus_p,(talshComplex8*)larg,(int)lc,(talshComplex8*)rarg,(int)lr,(talshComplex8*)beta_p,
                 (talshComplex8*)darg,(int)ll);
     }else{
      err_cublas=cublasZgemm(cublas_handle[gpu_num],left_conj,right_conj,(int)ll,(int)lr,(int)lc,
                 (talshComplex8*)alpha_plus_p,(talshComplex8*)larg,(int)lc,(talshComplex8*)rarg,(int)lc,(talshComplex8*)beta_p,
                 (talshComplex8*)darg,(int)ll);
     }
#endif
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,73); errc=gpu_activate(cur_gpu); return 73;
   }
#ifdef USE_CUTENSOR
   if(err_cutensor != CUTENSOR_STATUS_SUCCESS){
    if(VERBOSE){
     err_msg=cutensorGetErrorString(err_cutensor);
     if(err_msg != NULL) printf("#ERROR(gpu_tensor_block_contract_dlf): cuTensor error: %s\n",err_msg);
    }
    errc=cuda_task_record(cuda_task,coh_ctrl,74); errc=gpu_activate(cur_gpu); return 74;
   }
#else
   if(err_cublas != CUBLAS_STATUS_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,74); errc=gpu_activate(cur_gpu); return 74;}
#endif
  }else{ //BLAS is disabled
#endif /*NO_BLAS*/
   bx=1+(vol_l-1)/MAT_MULT_TILE_DIMX; by=1+(vol_r-1)/MAT_MULT_TILE_DIMY; limit_cuda_blocks2d(MAX_CUDA_BLOCKS,&bx,&by);
   //if(DEBUG) printf("\n#DEBUG(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): CUDA exec conf: %d %d %d %d\n",bx,by,MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY); //debug
   dim3 blcks(bx,by); dim3 thrds(MAT_MULT_TILE_DIMX,MAT_MULT_TILE_DIMY);
   switch(dtens->data_kind){
    case R4:
     gpu_matrix_multiply_tn__<<<blcks,thrds,0,*cuda_stream>>>(ll,lr,lc,(float*)larg,(float*)rarg,(float*)darg,(float)scale_real);
     break;
    case R8:
     gpu_matrix_multiply_tn__<<<blcks,thrds,0,*cuda_stream>>>(ll,lr,lc,(double*)larg,(double*)rarg,(double*)darg,scale_real);
     break;
    default: //`Add complex cases with and without conjugation
     errc=cuda_task_record(cuda_task,coh_ctrl,75); errc=gpu_activate(cur_gpu); return 75;
   }
#ifndef NO_BLAS
  }
#endif
 }
#ifdef GPU_FINE_TIMING
// Record a CUDA event:
 err=cudaEventRecord(*cuda_mmend,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the mmend event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,76); errc=gpu_activate(cur_gpu); return 76;
 }
#endif
 switch(dtens->data_kind){
  case R4:
   gpu_stats[gpu_num].flops+=2.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
  case R8:
   gpu_stats[gpu_num].flops+=2.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
  case C4:
   gpu_stats[gpu_num].flops+=8.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
  case C8:
   gpu_stats[gpu_num].flops+=8.0*((double)(lc))*((double)(ll))*((double)(lr));
   break;
 }
//Schedule the inverse tensor transpose for the destination tensor:
 if(perm_d == YEP){
  if(TRANS_SHMEM == EFF_TRN_ON){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     if(fast_math == YEP){
      gpu_tensor_block_copy_cmplx_split_in_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
        (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     }else{
      gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
       (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
        (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     }
     break;
    case C8:
     gpu_tensor_block_copy_dlf__<<<bx,THRDS_TENSOR_COPY,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,77); errc=gpu_activate(cur_gpu); return 77;
   }
  }else if(TRANS_SHMEM == EFF_TRN_ON_CUTT){
#ifdef USE_CUTT
   errc=prmn_convert(drank,cuda_task->tens_args[0].prmn_p,dprm); for(i=0;i<drank;++i) --(dprm[i]);
   for(i=0;i<drank;++i) rprm[i]=(dtens->shape).dims[drank-i-1]; //inversed dimension order
   cutt_err=cuttPlan(&cutt_d,drank,rprm,dprm,((size_t)tds_d),*cuda_stream);
   if(cutt_err == CUTT_SUCCESS){
    cutt_err=cuttExecute(cutt_d,dtens->tmp_rsc->gmem_p,dtens->dst_rsc->gmem_p);
    if(cutt_err != CUTT_SUCCESS){errc=cuda_task_record(cuda_task,coh_ctrl,78); errc=gpu_activate(cur_gpu); return 78;};
   }else{
    errc=cuda_task_record(cuda_task,coh_ctrl,79); errc=gpu_activate(cur_gpu); return 79;
   }
#else
   errc=cuda_task_record(cuda_task,coh_ctrl,80); errc=gpu_activate(cur_gpu); return 80;
#endif
  }else if(TRANS_SHMEM == EFF_TRN_OFF){
   bx=1+(vol_d-1)/THRDS_TENSOR_COPY_SCAT; if(bx > MAX_CUDA_BLOCKS) bx=MAX_CUDA_BLOCKS;
   switch(dtens->data_kind){
    case R4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(float*)(dtens->tmp_rsc->gmem_p),(float*)(dtens->dst_rsc->gmem_p));
     break;
    case R8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,(double*)(dtens->tmp_rsc->gmem_p),(double*)(dtens->dst_rsc->gmem_p));
     break;
    case C4:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex4*)(dtens->tmp_rsc->gmem_p),(talshComplex4*)(dtens->dst_rsc->gmem_p));
     break;
    case C8:
     gpu_tensor_block_copy_scatter_dlf__<<<bx,THRDS_TENSOR_COPY_SCAT,0,*cuda_stream>>>
      (1,0,drank,cuda_task->tens_args[0].const_mem_entry,
       (talshComplex8*)(dtens->tmp_rsc->gmem_p),(talshComplex8*)(dtens->dst_rsc->gmem_p));
     break;
    default:
     errc=cuda_task_record(cuda_task,coh_ctrl,81); errc=gpu_activate(cur_gpu); return 81;
   }
  }else{
   errc=cuda_task_record(cuda_task,coh_ctrl,82); errc=gpu_activate(cur_gpu); return 82;
  }
 }
//Record a CUDA event (output ready on GPU):
 err=cudaEventRecord(*cuda_output,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the output event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,83); errc=gpu_activate(cur_gpu); return 83;
 }
//Transfer back the updated destination tensor if needed ("T","K" coherence control):
 coh=(coh_ctrl>>4)&(TWO_BITS_SET); //select bits 4,5 (destination tensor coherence)
 if(gpu_d != gpu_num && coh >= 2){ //data is not on the computing GPU and coherence control = 2("T") or (3)"K":
  err=cudaMemcpyAsync(dtens->src_rsc->gmem_p,dtens->dst_rsc->gmem_p,dsize,cudaMemcpyDefault,*cuda_stream);
  if(err != cudaSuccess){
   err_msg=cudaGetErrorString(err);
   if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Destination tensor body back copy failed: %s\n",err_msg);
   errc=cuda_task_record(cuda_task,coh_ctrl,84); errc=gpu_activate(cur_gpu); return 84;
  }
  gpu_stats[gpu_num].traffic_out+=dsize;
 }
//Record a CUDA event (task finished):
 err=cudaEventRecord(*cuda_finish,*cuda_stream);
 if(err != cudaSuccess){
  err_msg=cudaGetErrorString(err);
  if(VERBOSE) printf("\n#ERROR(tensor_algebra_gpu_nvidia:gpu_tensor_block_contract_dlf): Unable to record the finish event: %s\n",err_msg);
  errc=cuda_task_record(cuda_task,coh_ctrl,85); errc=gpu_activate(cur_gpu); return 85;
 }
//Record the successfully scheduled CUDA task and update the Last Task:
 errc=cuda_task_record(cuda_task,coh_ctrl,0);
 LastTask[gpu_num]=cuda_task;
 if(gpu_num != cur_gpu) errc=gpu_activate(cur_gpu);
 return stat; //either 0 (success) or NOT_CLEAN (warning)
}

__host__ int gpu_tensor_block_decompose_svd(const char absorb, tensBlck_t *dtens, tensBlck_t *ltens, tensBlck_t *rtens, tensBlck_t *stens, int gpu_id)
{
 //`Finish
 return -1;
}

#endif /*NO_GPU*/
