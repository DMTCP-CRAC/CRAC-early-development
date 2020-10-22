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

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include "cudart_apis.h"

#include "jassert.h"
#include "log_and_replay.h"


using namespace std;
static bool disable_logging = false;
// struct CudaCallLog_t {
//   void *fncargs;
//   size_t size;
//   void *results;
// };

// enum pages_t {
//   CUDA_MALLOC_PAGE = 0,
//   CUDA_UVM_PAGE,
//   ....
// };

// typedef struct Lhckpt_pages_t {
//   pages_t mem_type;
//   size_t mem_len;
// }lhckpt_pages_t;

map<void *, lhckpt_pages_t> lh_pages_map;

map<void *, lhckpt_pages_t> &
getLhPageMaps()
{
  return lh_pages_map;
}

std::vector<CudaCallLog_t> cudaCallsLog;

std::vector<CudaCallLog_t> &
getCudaCallsLog() {
  return cudaCallsLog;
}

void display_map()
{
  for (auto lh_page : lh_pages_map) {
    printf("\n Address = %p with size = %lu", lh_page.first, lh_page.second.mem_len);
  }
}

void
enableLogging()
{
  disable_logging = false;
}

void
disableLogging()
{
  disable_logging = true;
}

bool
isLoggingDisabled()
{
  return disable_logging;
}

// This function does in-memory logging of CUDA calls that are specified
// using the @LogReplay decorator.
void
logAPI(Cuda_Fncs_t cuda_fnc_op, ...)
{
  if (isLoggingDisabled())
  {
    return;
  }
  // printf("In logAPI\n");
  fflush(stdout);
  va_list arglist;
  va_start (arglist, cuda_fnc_op);
  CudaCallLog_t log;
  char buf[4096];
  size_t chars_wrote = 0;
  // fill the cuda function op fisrtmem_typeto the buf
  memcpy(buf + chars_wrote, &cuda_fnc_op, sizeof cuda_fnc_op);
  chars_wrote += sizeof cuda_fnc_op;
  printf("In logAPI %s:start\n", cuda_Fnc_to_str[cuda_fnc_op]);
  fflush(stdout);

  switch(cuda_fnc_op) {
    case GENERATE_ENUM(cudaMalloc):
    {
      // args
      void **pointer = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, pointer, sizeof (void *));
      chars_wrote += sizeof (void *);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof (size);
      // update the map
      lhckpt_pages_t page = {CUDA_MALLOC_PAGE, *pointer, size};
      lh_pages_map[*pointer] = page;
      // display_map();
      break;
    }
    case GENERATE_ENUM(cuMemAlloc_v2):
    {
      // args
      CUdeviceptr *pointer = va_arg(arglist, CUdeviceptr *);
      memcpy(buf + chars_wrote, &pointer, sizeof (CUdeviceptr *));
      chars_wrote += sizeof (CUdeviceptr *);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof (size);
      // update the map
      lhckpt_pages_t page = {CUMEM_ALLOC_PAGE, pointer, size};
      lh_pages_map[pointer] = page;
      // display_map();
      break;
    }
    case GENERATE_ENUM(cudaFree):
    {
      // args
      void *pointer = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &pointer, sizeof (void *));
      chars_wrote += sizeof (void *);
      // remove from maps
      lh_pages_map.erase(pointer);
      break;
    }
    case GENERATE_ENUM(cuMemFree_v2):
    {
      // args
      CUdeviceptr pointer = va_arg(arglist, CUdeviceptr);
      memcpy(buf + chars_wrote, &pointer, sizeof (CUdeviceptr));
      chars_wrote += sizeof (CUdeviceptr);
      // remove from maps
      lh_pages_map.erase(&pointer);
      break;
    }
    case GENERATE_ENUM(__cudaInitModule):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, fatCubinHandle, sizeof (void *));
      chars_wrote += sizeof (void *);

      break;
    }
    case GENERATE_ENUM(__cudaPushCallConfiguration):
    {
      // args
      dim3 gridDim = va_arg(arglist, dim3);
      memcpy(buf + chars_wrote, &gridDim, sizeof (gridDim));
      chars_wrote += sizeof (gridDim);

      dim3 blockDim = va_arg(arglist, dim3);
      memcpy(buf + chars_wrote, &blockDim, sizeof (blockDim));
      chars_wrote += sizeof (blockDim);

      size_t sharedMem = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &sharedMem, sizeof (sharedMem));
      chars_wrote += sizeof (sharedMem);

      void *stream = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &stream, sizeof (void *));
      chars_wrote += sizeof (void *);

      log.res_size = sizeof(unsigned int);
      log.results = (char *)JALLOC_MALLOC(log.res_size + 1);

      unsigned int res = va_arg(arglist, unsigned int);
      memcpy(log.results, &res, sizeof (unsigned int));

      break;
    }
    case GENERATE_ENUM(__cudaPopCallConfiguration):
    {
      // args
      dim3 *gridDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &gridDim, sizeof (*gridDim));
      chars_wrote += sizeof (*gridDim);

      dim3 *blockDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &blockDim, sizeof (*blockDim));
      chars_wrote += sizeof (*blockDim);

      size_t *sharedMem = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, &sharedMem, sizeof (*sharedMem));
      chars_wrote += sizeof (*sharedMem);

      void *stream = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &stream, sizeof (void *));
      chars_wrote += sizeof (void *);

      break;
    }
    case GENERATE_ENUM(__cudaRegisterFatBinary):
    {
      // args
      void *fatCubin = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubin, sizeof (void *));
      chars_wrote += sizeof (void *);
      // result
      void **res = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &res, sizeof (void *));
      chars_wrote += sizeof (void *);

      log.res_size = sizeof(*res);
      log.results = (char *)JALLOC_MALLOC(log.res_size + 1);
      memcpy(log.results, res, sizeof (*res));
      break;
    }
    // new
    case GENERATE_ENUM(__cudaRegisterFatBinaryEnd):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (void *));
      chars_wrote += sizeof (void *);

      break;
    }

    case GENERATE_ENUM(__cudaUnregisterFatBinary):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      break;
    }
    case GENERATE_ENUM(__cudaRegisterFunction):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      char *hostFun = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &hostFun, sizeof(char *));
      chars_wrote += sizeof(char *);


      char *deviceFun = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceFun, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(char *));
      chars_wrote += sizeof(char *);

      int thread_limit = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &thread_limit, sizeof thread_limit);
      chars_wrote += sizeof thread_limit;

      uint3 *tid = va_arg(arglist, uint3 *);
      memcpy(buf + chars_wrote, &tid, sizeof (uint3 *));
      chars_wrote += sizeof (uint3 *);

      uint3 *bid = va_arg(arglist, uint3 *);
      memcpy(buf + chars_wrote, &bid, sizeof (uint3 *));
      chars_wrote += sizeof (uint3 *);

      dim3 *bDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &bDim, sizeof (dim3 *));
      chars_wrote += sizeof (dim3 *);

      dim3 *gDim = va_arg(arglist, dim3 *);
      memcpy(buf + chars_wrote, &gDim, sizeof (dim3 *));
      chars_wrote += sizeof (dim3 *);

      int *wSize = va_arg(arglist, int*);
      memcpy(buf + chars_wrote, &wSize, sizeof (int *));
      chars_wrote += sizeof (int *);
      break;
    }
    case GENERATE_ENUM(__cudaRegisterVar):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      char *hostVar = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &hostVar, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceAddress = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(char *));
      chars_wrote += sizeof(char *);

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof (size));
      chars_wrote += sizeof (size);

      int constant = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &constant, sizeof constant);
      chars_wrote += sizeof constant;

      int global = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &global, sizeof global);
      chars_wrote += sizeof global;
      break;
    }
    case GENERATE_ENUM(__cudaRegisterManagedVar):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      void **hostVarPtrAddress = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &hostVarPtrAddress, sizeof (*hostVarPtrAddress));
      chars_wrote += sizeof (*hostVarPtrAddress);

      char *deviceAddress = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(char *));
      chars_wrote += sizeof(char *);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(char *));
      chars_wrote += sizeof(char *);

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof (size));
      chars_wrote += sizeof (size);

      int constant = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &constant, sizeof constant);
      chars_wrote += sizeof constant;

      int global = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &global, sizeof global);
      chars_wrote += sizeof global;
      break;
    }
    case GENERATE_ENUM(__cudaRegisterTexture):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      struct textureReference *hostVar = va_arg(arglist,
                                                struct textureReference *);
      memcpy(buf + chars_wrote, hostVar, sizeof(*hostVar));
      chars_wrote += sizeof(*hostVar);

      void **deviceAddress = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, deviceAddress, sizeof(*deviceAddress));
      chars_wrote += sizeof(*deviceAddress);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, deviceName, strlen(deviceName));
      chars_wrote += strlen(deviceName);

      int dim = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &dim, sizeof dim);
      chars_wrote += sizeof dim;

      int norm = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &norm, sizeof norm);
      chars_wrote += sizeof norm;

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;
      break;
    }
    case GENERATE_ENUM(__cudaRegisterSurface):
    {
      // args
      void **fatCubinHandle = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, fatCubinHandle, sizeof (*fatCubinHandle));
      chars_wrote += sizeof (*fatCubinHandle);

      struct surfaceReference *hostVar = va_arg(arglist,
                                                struct surfaceReference *);
      memcpy(buf + chars_wrote, &hostVar, sizeof(hostVar));
      chars_wrote += sizeof(hostVar);

      void **deviceAddress = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &deviceAddress, sizeof(deviceAddress));
      chars_wrote += sizeof(deviceAddress);

      char *deviceName = va_arg(arglist, char *);
      memcpy(buf + chars_wrote, &deviceName, sizeof(deviceName));
      chars_wrote += sizeof(deviceName);

      int dim = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &dim, sizeof dim);
      chars_wrote += sizeof dim;

      int ext = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &ext, sizeof ext);
      chars_wrote += sizeof ext;
      break;
    }
    case GENERATE_ENUM(cudaCreateTextureObject):
    {
      // args
      cudaTextureObject_t * pTexObject = va_arg(arglist, cudaTextureObject_t *);
      memcpy(buf + chars_wrote, &pTexObject, sizeof(pTexObject));
      chars_wrote += sizeof(pTexObject);

      struct cudaResourceDesc * pResDesc = va_arg(arglist,
                                                  struct cudaResourceDesc *);
      memcpy(buf + chars_wrote, &pResDesc, sizeof(pResDesc));
      chars_wrote += sizeof(pResDesc);

      struct cudaTextureDesc * pTexDesc = va_arg(arglist,
                                                 struct cudaTextureDesc *);
      memcpy(buf + chars_wrote, &pTexDesc, sizeof(pTexDesc));
      chars_wrote += sizeof(pTexDesc);

      struct cudaResourceViewDesc * pResViewDesc = va_arg(arglist,
                                                 struct cudaResourceViewDesc *);
      memcpy(buf + chars_wrote, &pResViewDesc, sizeof(pResViewDesc));
      chars_wrote += sizeof(pResViewDesc);
      break;
    }
    case GENERATE_ENUM(cudaDestroyTextureObject):
    {
      // args
      cudaTextureObject_t texObject = va_arg(arglist, cudaTextureObject_t);
      memcpy(buf + chars_wrote, &texObject, sizeof(texObject));
      chars_wrote += sizeof(texObject);
      break;
    }
    case GENERATE_ENUM(cudaBindTextureToArray):
    {
      textureReference* texref = va_arg(arglist, textureReference *);
      memcpy(buf + chars_wrote, &texref, sizeof(texref));
      chars_wrote += sizeof(texref);

      cudaArray_const_t array = va_arg(arglist, cudaArray_const_t);
      memcpy(buf + chars_wrote, &array, sizeof(array));
      chars_wrote += sizeof(array);

      cudaChannelFormatDesc * desc = va_arg(arglist, cudaChannelFormatDesc *);
      memcpy(buf + chars_wrote, &desc, sizeof(desc));
      chars_wrote += sizeof(desc);
      break;
    }
    case GENERATE_ENUM(cudaUnbindTexture):
    {
      struct textureReference * texref = va_arg(arglist, textureReference *);
      memcpy(buf + chars_wrote, &texref, sizeof(texref));
      chars_wrote += sizeof(texref);
      break;
    }
    case GENERATE_ENUM(cudaCreateChannelDesc):
    {
      // args
      int x = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &x, sizeof(x));
      chars_wrote += sizeof(x);

      int y = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &y, sizeof(y));
      chars_wrote += sizeof(y);

      int z = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &z, sizeof(z));
      chars_wrote += sizeof(z);

      int w = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &w, sizeof(w));
      chars_wrote += sizeof(w);

      int f = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &f, sizeof(f));
      chars_wrote += sizeof(f);

      // result
      cudaChannelFormatDesc res = va_arg(arglist, cudaChannelFormatDesc);
      log.res_size = sizeof(res);
      log.results = (char *)JALLOC_MALLOC(log.res_size + 1);
      memcpy(log.results, &res, sizeof (res));
      break;
    }
    case GENERATE_ENUM(cudaMallocArray):
    {
      // args
      cudaArray_t ** array = va_arg(arglist, cudaArray_t **);
      memcpy(buf + chars_wrote, &array, sizeof(array));
      chars_wrote += sizeof(array);

      struct cudaChannelFormatDesc * desc = va_arg(arglist,
                                              struct cudaChannelFormatDesc *);
      memcpy(buf + chars_wrote, &desc, sizeof(desc));
      chars_wrote += sizeof(desc);

      size_t width = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &width, sizeof(width));
      chars_wrote += sizeof(width);

      size_t height = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &height, sizeof(height));
      chars_wrote += sizeof(height);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaFreeArray):
    {
      // args
      cudaArray_t array = va_arg(arglist, cudaArray_t);
      memcpy(buf + chars_wrote, &array, sizeof(array));
      chars_wrote += sizeof(array);
      break;
    }
    case GENERATE_ENUM(cudaMallocHost):
    {
      //args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(void *));
      chars_wrote += sizeof(void *);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;
      break;
    }
    case GENERATE_ENUM(cuMemAllocHost_v2):
    {
      //args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(void *));
      chars_wrote += sizeof(void *);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;
      break;
    }
    case GENERATE_ENUM(cudaFreeHost):
    {
      //args
      void * ptr = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &ptr, sizeof(void *));
      chars_wrote += sizeof(void *);
      break;
    }
    case GENERATE_ENUM(cuMemFreeHost):
    {
      //args
      void * ptr = va_arg(arglist, void *);
      memcpy(buf + chars_wrote, &ptr, sizeof(void *));
      chars_wrote += sizeof(void *);
      break;
    }
    case GENERATE_ENUM(cudaHostAlloc):
    {
      //args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(ptr));
      chars_wrote += sizeof(ptr);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuMemHostAlloc):
    {
      //args
      void ** ptr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, &ptr, sizeof(ptr));
      chars_wrote += sizeof(ptr);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaDeviceReset):
    {
      break;
    }
    case GENERATE_ENUM(cudaMallocPitch):
    {
      void ** devPtr = va_arg(arglist, void **);
      memcpy(buf + chars_wrote, *devPtr, sizeof(devPtr));
      chars_wrote += sizeof(devPtr);

      size_t * pitch = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, &pitch, sizeof pitch);
      chars_wrote += sizeof(pitch);

      size_t width = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &width, sizeof(width));
      chars_wrote += sizeof(width);

      size_t height = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &height, sizeof(height));
      chars_wrote += sizeof(height);
      break;
    }
    case GENERATE_ENUM(cuMemAllocPitch_v2):
    {
      CUdeviceptr* devPtr = va_arg(arglist, CUdeviceptr*);
      memcpy(buf + chars_wrote, &devPtr, sizeof(devPtr));
      chars_wrote += sizeof(devPtr);

      size_t * pitch = va_arg(arglist, size_t *);
      memcpy(buf + chars_wrote, &pitch, sizeof pitch);
      chars_wrote += sizeof(pitch);

      size_t width = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &width, sizeof(width));
      chars_wrote += sizeof(width);

      size_t height = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &height, sizeof(height));
      chars_wrote += sizeof(height);

      unsigned int ElementSizeBytes = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &ElementSizeBytes, sizeof(ElementSizeBytes));
      chars_wrote += sizeof(ElementSizeBytes);
      break;
    }
    case GENERATE_ENUM(cudaDeviceSynchronize):
    {
      break;
    }
    case GENERATE_ENUM(cudaMallocManaged):
    {
      //args
      void ** devPtr = va_arg(arglist, void **);
      // memcpy(buf + chars_wrote, *devPtr, sizeof(devPtr));
      memcpy(buf + chars_wrote, *devPtr, sizeof(devPtr));
      chars_wrote += sizeof(devPtr);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      // update the map
      lhckpt_pages_t page = {CUDA_UVM_PAGE, *devPtr, size};
      lh_pages_map[*devPtr] = page;
      display_map();
      break;
    }
    case GENERATE_ENUM(cuMemAllocManaged):
    {
      //args
      CUdeviceptr * devPtr = va_arg(arglist, CUdeviceptr*);
      // memcpy(buf + chars_wrote, *devPtr, sizeof(devPtr));
      memcpy(buf + chars_wrote, &devPtr, sizeof(devPtr));
      chars_wrote += sizeof(devPtr);

      size_t size = va_arg(arglist, size_t);
      memcpy(buf + chars_wrote, &size, sizeof size);
      chars_wrote += sizeof size;

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      // update the map
      lhckpt_pages_t page = {CUDA_UVM_PAGE, devPtr, size};
      lh_pages_map[devPtr] = page;
      display_map();
      break;
    }
    case GENERATE_ENUM(cudaStreamCreate):
    {
      //args
      cudaStream_t *pStream = va_arg(arglist, cudaStream_t *);
      memcpy(buf + chars_wrote, &pStream, sizeof(pStream));
      chars_wrote += sizeof(pStream);
      break;
    }
    case GENERATE_ENUM(cuStreamCreate):
    {
      //args
      CUstream *phStream = va_arg(arglist, CUstream *);
      memcpy(buf + chars_wrote, &phStream, sizeof(phStream));
      chars_wrote += sizeof(phStream);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaStreamCreateWithFlags):
    {
      //args
      cudaStream_t *pStream = va_arg(arglist, cudaStream_t *);
      memcpy(buf + chars_wrote, &pStream, sizeof(pStream));
      chars_wrote += sizeof(pStream);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaStreamCreateWithPriority):
    {
      //args
      cudaStream_t *pStream = va_arg(arglist, cudaStream_t *);
      memcpy(buf + chars_wrote, &pStream, sizeof(pStream));
      chars_wrote += sizeof(pStream);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);

      int priority = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &priority, sizeof(priority));
      chars_wrote += sizeof(priority);
      break;
    }
    case GENERATE_ENUM(cuStreamCreateWithPriority):
    {
      //args
      CUstream *phStream = va_arg(arglist, CUstream *);
      memcpy(buf + chars_wrote, &phStream, sizeof(phStream));
      chars_wrote += sizeof(phStream);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);

      int priority = va_arg(arglist, int);
      memcpy(buf + chars_wrote, &priority, sizeof(priority));
      chars_wrote += sizeof(priority);
      break;
    }
    case GENERATE_ENUM(cudaStreamDestroy):
    {
      //args
      cudaStream_t pStream = va_arg(arglist, cudaStream_t);
      memcpy(buf + chars_wrote, &pStream, sizeof(pStream));
      chars_wrote += sizeof(pStream);
      break;
    }
    case GENERATE_ENUM(cuStreamDestroy_v2):
    {
      //args
      CUstream hStream = va_arg(arglist, CUstream);
      memcpy(buf + chars_wrote, &hStream, sizeof(hStream));
      chars_wrote += sizeof(hStream);
      break;
    }
    case GENERATE_ENUM(cudaEventCreate):
    {
      cudaEvent_t * event = va_arg(arglist, cudaEvent_t *);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      break;
    }
    case GENERATE_ENUM(cuEventCreate):
    {
      CUevent * event = va_arg(arglist, CUevent *);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cudaEventDestroy):
    {
      cudaEvent_t event = va_arg(arglist, cudaEvent_t );
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      break;
    }
    case GENERATE_ENUM(cuEventDestroy_v2):
    {
      CUevent event = va_arg(arglist, CUevent);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);
      break;
    }
    case GENERATE_ENUM(cudaEventCreateWithFlags):
    {
      cudaEvent_t * event = va_arg(arglist, cudaEvent_t *);
      memcpy(buf + chars_wrote, &event, sizeof(event));
      chars_wrote += sizeof(event);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuDestroyExternalMemory):
    {
      CUexternalMemory extMem = va_arg(arglist, CUexternalMemory);
      memcpy(buf + chars_wrote, &extMem, sizeof(extMem));
      chars_wrote += sizeof(extMem);
      break;
    }
    case GENERATE_ENUM(cuDestroyExternalSemaphore):
    {
      CUexternalSemaphore extSem = va_arg(arglist, CUexternalSemaphore);
      memcpy(buf + chars_wrote, &extSem, sizeof(extSem));
      chars_wrote += sizeof(extSem);
      break;
    }
    case GENERATE_ENUM(cuGraphCreate):
    {
      CUgraph * phGraph = va_arg(arglist, CUgraph *);
      memcpy(buf + chars_wrote, &phGraph, sizeof(phGraph));
      chars_wrote += sizeof(phGraph);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);
      break;
    }
    case GENERATE_ENUM(cuGraphDestroy):
    {
      CUgraph hGraph = va_arg(arglist, CUgraph );
      memcpy(buf + chars_wrote, &hGraph, sizeof(hGraph));
      chars_wrote += sizeof(hGraph);
      break;
    }
    case GENERATE_ENUM(cuGraphDestroyNode):
    {
      CUgraphNode hNode = va_arg(arglist, CUgraphNode );
      memcpy(buf + chars_wrote, &hNode, sizeof(hNode));
      chars_wrote += sizeof(hNode);
      break;
    }
    case GENERATE_ENUM(cuGraphExecDestroy):
    {
      CUgraphExec hGraphExec = va_arg(arglist, CUgraphExec );
      memcpy(buf + chars_wrote, &hGraphExec, sizeof(hGraphExec));
      chars_wrote += sizeof(hGraphExec);
      break;
    }
    case GENERATE_ENUM(cuTexRefCreate):
    {
      CUtexref *pTexRef = va_arg(arglist, CUtexref *);
      memcpy(buf + chars_wrote, &pTexRef, sizeof(pTexRef));
      chars_wrote += sizeof(pTexRef);
      break;
    }
    case GENERATE_ENUM(cuTexRefDestroy):
    {
      CUtexref hTexRef = va_arg(arglist, CUtexref);
      memcpy(buf + chars_wrote, &hTexRef, sizeof(hTexRef));
      chars_wrote += sizeof(hTexRef);
      break;
    }
    case GENERATE_ENUM(cuTexObjectCreate):
    {
      CUtexObject *pTexObject = va_arg(arglist, CUtexObject *);
      memcpy(buf + chars_wrote, &pTexObject, sizeof(pTexObject));
      chars_wrote += sizeof(pTexObject);

      CUDA_RESOURCE_DESC* pResDesc = va_arg(arglist, CUDA_RESOURCE_DESC *);
      memcpy(buf + chars_wrote, &pResDesc, sizeof(pResDesc));
      chars_wrote += sizeof(pResDesc);

      CUDA_TEXTURE_DESC* pTexDesc = va_arg(arglist, CUDA_TEXTURE_DESC *);
      memcpy(buf + chars_wrote, &pTexDesc, sizeof(pTexDesc));
      chars_wrote += sizeof(pTexDesc);

      CUDA_RESOURCE_VIEW_DESC* pResViewDesc = va_arg(arglist, CUDA_RESOURCE_VIEW_DESC *);
      memcpy(buf + chars_wrote, &pResViewDesc, sizeof(pResViewDesc));
      chars_wrote += sizeof(pResViewDesc);
      break;
    }
    case GENERATE_ENUM(cuTexObjectDestroy):
    {
      CUtexObject pTexObject = va_arg(arglist, CUtexObject );
      memcpy(buf + chars_wrote, &pTexObject, sizeof(pTexObject));
      chars_wrote += sizeof(pTexObject);
      break;
    }
    case GENERATE_ENUM(cuSurfObjectCreate):
    {
      CUsurfObject *pSurfObject = va_arg(arglist, CUsurfObject *);
      memcpy(buf + chars_wrote, &pSurfObject, sizeof(pSurfObject));
      chars_wrote += sizeof(pSurfObject);

      CUDA_RESOURCE_DESC *pResDesc = va_arg(arglist, CUDA_RESOURCE_DESC *);
      memcpy(buf + chars_wrote, &pResDesc, sizeof(pResDesc));
      chars_wrote += sizeof(pResDesc);
      break;
    }
    case GENERATE_ENUM(cuSurfObjectDestroy):
    {
      CUsurfObject pSurfObject = va_arg(arglist, CUsurfObject);
      memcpy(buf + chars_wrote, &pSurfObject, sizeof(pSurfObject));
      chars_wrote += sizeof(pSurfObject);
      break;
    }
    case GENERATE_ENUM(cublasCreate_v2):
    {
      cublasHandle_t *handle = va_arg(arglist, cublasHandle_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cublasDestroy_v2):
    {
      cublasHandle_t handle = va_arg(arglist, cublasHandle_t);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreate):
    {
      cusparseHandle_t *handle = va_arg(arglist, cusparseHandle_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroy):
    {
      cusparseHandle_t handle = va_arg(arglist, cusparseHandle_t);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreateMatDescr):
    {
      cusparseMatDescr_t *handle = va_arg(arglist, cusparseMatDescr_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroyMatDescr):
    {
      cusparseMatDescr_t handle = va_arg(arglist, cusparseMatDescr_t );
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreateHybMat):
    {
      cusparseHybMat_t *handle = va_arg(arglist, cusparseHybMat_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroyHybMat):
    {
      cusparseHybMat_t handle = va_arg(arglist, cusparseHybMat_t );
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreateSolveAnalysisInfo):
    {
      cusparseSolveAnalysisInfo_t *info = va_arg(arglist,
                                                 cusparseSolveAnalysisInfo_t *);
      memcpy(buf + chars_wrote, &info, sizeof(info));
      chars_wrote += sizeof(info);
      break;
    }
    case GENERATE_ENUM(cusparseDestroySolveAnalysisInfo):
    {
      cusparseSolveAnalysisInfo_t info = va_arg(arglist,
                                                 cusparseSolveAnalysisInfo_t);
      memcpy(buf + chars_wrote, &info, sizeof(info));
      chars_wrote += sizeof(info);
      break;
    }
    case GENERATE_ENUM(cusolverDnCreate):
    {
      cusolverDnHandle_t *handle = va_arg(arglist, cusolverDnHandle_t *);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cusolverDnDestroy):
    {
      cusolverDnHandle_t handle = va_arg(arglist, cusolverDnHandle_t);
      memcpy(buf + chars_wrote, &handle, sizeof(handle));
      chars_wrote += sizeof(handle);
      break;
    }
    case GENERATE_ENUM(cuCtxCreate_v2):
    {
      CUcontext *pctx = va_arg(arglist, CUcontext *);
      memcpy(buf + chars_wrote, &pctx, sizeof(pctx));
      chars_wrote += sizeof(pctx);

      unsigned int flags = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &flags, sizeof(flags));
      chars_wrote += sizeof(flags);

      CUdevice dev = va_arg(arglist, CUdevice);
      memcpy(buf + chars_wrote, &dev, sizeof(dev));
      chars_wrote += sizeof(dev);
      break;
    }
    case GENERATE_ENUM(cuCtxDestroy_v2):
    {
      CUcontext pctx = va_arg(arglist, CUcontext );
      memcpy(buf + chars_wrote, &pctx, sizeof(pctx));
      chars_wrote += sizeof(pctx);
      break;
    }
    case GENERATE_ENUM(cuLinkCreate_v2):
    {
      unsigned int numOptions = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &numOptions, sizeof(numOptions));
      chars_wrote += sizeof(numOptions);

      CUjit_option * options = va_arg(arglist, CUjit_option *);
      memcpy(buf + chars_wrote, &options, sizeof(options));
      chars_wrote += sizeof(options);

      void** optionValues = va_arg(arglist, void**);
      memcpy(buf + chars_wrote, &optionValues, sizeof(optionValues));
      chars_wrote += sizeof(optionValues);
      // out parameter
      CUlinkState * stateOut = va_arg(arglist, CUlinkState *);
      memcpy(buf + chars_wrote, &stateOut, sizeof(stateOut));
      chars_wrote += sizeof(stateOut);
      break;
    }
    case GENERATE_ENUM(cuLinkDestroy):
    {
      CUlinkState  stateOut = va_arg(arglist, CUlinkState);
      memcpy(buf + chars_wrote, &stateOut, sizeof(stateOut));
      chars_wrote += sizeof(stateOut);
      break;
    }
    case GENERATE_ENUM(cuArray3DCreate_v2):
    {
      CUarray* pHandle = va_arg(arglist, CUarray*);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);

      CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray = \
            va_arg(arglist, CUDA_ARRAY3D_DESCRIPTOR *);
      memcpy(buf + chars_wrote, &pAllocateArray, sizeof(pAllocateArray));
      chars_wrote += sizeof(pAllocateArray);
      break;
    }
    case GENERATE_ENUM(cuArrayCreate_v2):
    {
      CUarray* pHandle = va_arg(arglist, CUarray*);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);

      CUDA_ARRAY_DESCRIPTOR* pAllocateArray = \
            va_arg(arglist, CUDA_ARRAY_DESCRIPTOR *);
      memcpy(buf + chars_wrote, &pAllocateArray, sizeof(pAllocateArray));
      chars_wrote += sizeof(pAllocateArray);
      break;
    }
    case GENERATE_ENUM(cuArrayDestroy):
    {
      CUarray pHandle = va_arg(arglist, CUarray);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);
      break;
    }
    case GENERATE_ENUM(cuMipmappedArrayCreate):
    {
      CUmipmappedArray* pHandle = va_arg(arglist, CUmipmappedArray*);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);

      CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc = \
            va_arg(arglist, CUDA_ARRAY3D_DESCRIPTOR *);
      memcpy(buf + chars_wrote, &pMipmappedArrayDesc, sizeof(pMipmappedArrayDesc));
      chars_wrote += sizeof(pMipmappedArrayDesc);

      unsigned int numMipmapLevels = va_arg(arglist, unsigned int);
      memcpy(buf + chars_wrote, &numMipmapLevels, sizeof(numMipmapLevels));
      chars_wrote += sizeof(numMipmapLevels);
      break;
    }
    case GENERATE_ENUM(cuMipmappedArrayDestroy):
    {
      CUmipmappedArray pHandle = va_arg(arglist, CUmipmappedArray);
      memcpy(buf + chars_wrote, &pHandle, sizeof(pHandle));
      chars_wrote += sizeof(pHandle);
      break;
    }
    default:
    {
      JNOTE("log API not implemented") (cuda_fnc_op);
      break;
    }
  }
  // common for every API
  log.fncargs = (char *)JALLOC_MALLOC(chars_wrote);
  memcpy(log.fncargs, buf, chars_wrote);
  log.size = chars_wrote;
  //push_back fails/segfaults when a lot of cuda Calls are made
  //To avoid the segfault we can resize cudaCallsLog
  cudaCallsLog.resize(log.size);
  cudaCallsLog.push_back(log);
  va_end(arglist);
}
