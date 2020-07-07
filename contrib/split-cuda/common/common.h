/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#ifndef COMMON_H
#define COMMON_H

#include <link.h>
#include <stdio.h>
#include <string.h>

#include <asm/prctl.h>
#include <linux/limits.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <cusolverDn.h>

#include "lower_half_cuda_if.h"
typedef char* VA;  /* VA = virtual address */

// Based on the entries in /proc/<pid>/stat as described in `man 5 proc`
enum Procstat_t
{
  PID = 1,
  COMM,   // 2
  STATE,  // 3
  PPID,   // 4
  NUM_THREADS = 19,
  STARTSTACK = 27,
};

#define PAGE_SIZE 0x1000LL

// FIXME: 0x1000 is one page; Use sysconf(PAGESIZE) instead.
#define ROUND_DOWN(x) ((unsigned long long)(x) \
                      & ~(unsigned long long)(PAGE_SIZE - 1))
#define ROUND_UP(x)  (((unsigned long long)(x) + PAGE_SIZE - 1) & \
                      ~(PAGE_SIZE - 1))
#define PAGE_OFFSET(x)  ((x) & (PAGE_SIZE - 1))

// TODO: This is very x86-64 specific; support other architectures??
#define eax rax
#define ebx rbx
#define ecx rcx
#define edx rax
#define ebp rbp
#define esi rsi
#define edi rdi
#define esp rsp
#define CLEAN_FOR_64_BIT_HELPER(args ...) # args
#define CLEAN_FOR_64_BIT(args ...)        CLEAN_FOR_64_BIT_HELPER(args)

typedef struct __LowerHalfInfo
{
  void *lhSbrk;
  void *lhMmap;
  void *lhMunmap;
  void *lhDlsym;
  unsigned long lhFsAddr;
  void *lhMmapListFptr;
  void *uhEndofHeapFptr;
  void *lhGetDeviceHeapFptr;
  void *lhCopyToCudaPtrFptr;
  void *lhDeviceHeap;
  void *getFatCubinHandle;
} LowerHalfInfo_t;

typedef struct __UpperHalfInfo
{
  void *uhEndofHeap;
  void *lhPagesRegion;
  void *cudaLogVectorFptr;
} UpperHalfInfo_t;

typedef struct __MmapInfo
{
  void *addr;
  size_t len;
} MmapInfo_t;

typedef struct __CudaCallLog{
  char *fncargs;
  size_t size;
  char *results;
  size_t res_size;
} CudaCallLog_t;

extern LowerHalfInfo_t lhInfo;
extern UpperHalfInfo_t uhInfo;

#ifdef __cplusplus
extern "C" {
#endif
void* lhDlsym(Cuda_Fncs_t type);
void** fatHandle();
#ifdef __cplusplus
}
#endif
typedef void* (*LhDlsym_t)(Cuda_Fncs_t type);
//getter function returning new_fatCubinHandle
// from the replay code
typedef void** (*fatHandle_t)();


//global_fatCubinHandle defined in cuda-plugin.cpp
extern void ** global_fatCubinHandle;
#endif // ifndef COMMON_H
