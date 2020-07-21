#pragma once

#include "device_algebra.h"
#include "talsh_complex.h"

int prmn_convert(int n, const int* o2n, int* n2o);
int non_trivial_prmn(int n, const int* prm);
#ifndef NO_GPU
int cuda_stream_get(int gpu_num, int* cuda_stream_handle);
int cuda_stream_release(int gpu_num, int cuda_stream_handle);
cudaStream_t* cuda_stream_ptr(int gpu_num, int cuda_stream_handle);
int cuda_event_get(int gpu_num, int* cuda_event_handle);
int cuda_event_release(int gpu_num, int cuda_event_handle);
cudaEvent_t* cuda_event_ptr(int gpu_num, int cuda_event_handle);
void limit_cuda_blocks2d(int max_blocks, int* bx, int* by);
int tens_op_best_gpu(const tensBlck_t* tens0 = NULL, const tensBlck_t* tens1 = NULL, const tensBlck_t* tens2 = NULL);
int cuda_task_set_arg(cudaTask_t* cuda_task, unsigned int arg_num, tensBlck_t* tens_p);
int cuda_task_set_prefactor(cudaTask_t* cuda_task, talshComplex4 prefactor);
int cuda_task_set_prefactor(cudaTask_t* cuda_task, talshComplex8 prefactor);
int cuda_task_record(cudaTask_t* cuda_task, unsigned int coh_ctrl, unsigned int err_code = 0);
int cuda_task_finalize(cudaTask_t* cuda_task);
#endif /*NO_GPU*/
