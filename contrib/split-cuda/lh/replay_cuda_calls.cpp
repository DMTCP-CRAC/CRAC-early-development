/****************************************************************************
 *   Copyright (C) 2019-2020 by Twinkle Jain, and Gene Cooperman            *
 *   jain.t@husky.neu.edu, gene@ccs.neu.edu                                 *
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

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cusparse_v2.h>
#include <cublas.h>
#include <assert.h>
#include <stdio.h>

#include "getmmap.h"
#include "common.h"

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
// #define __CUDACC__
#endif
#include "crt/host_runtime.h"
#include "crt/device_functions.h"
#include "log_and_replay.h"

void **new_fatCubinHandle = NULL;
void replayAPI(CudaCallLog_t *l)
{
  Cuda_Fncs_t op;
  memcpy(&op, l->fncargs, sizeof op);
  size_t chars_read = sizeof op;
  switch(op) {
    case GENERATE_ENUM(cudaMalloc):
    {
      void *oldDevPtr;
      memcpy(&oldDevPtr, l->fncargs + chars_read, sizeof oldDevPtr);
      chars_read += sizeof oldDevPtr;
      size_t len;

      memcpy(&len, l->fncargs + chars_read, sizeof len);
      void *newDevPtr = NULL;
      cudaError_t ret = cudaMalloc(&newDevPtr, len);
      assert(ret == cudaSuccess);

      // JASSERT(ret == cudaSuccess) ("cudaMalloc replay failed!");
      // JASSERT(newDevPtr == oldDevPtr) (oldDevPtr) (newDevPtr)
      //   .Text("new device pointer is different than old one!");
      break;
    }
    case GENERATE_ENUM(cuMemAlloc_v2):
    {
      CUdeviceptr *oldDevPtr;
      memcpy(&oldDevPtr, l->fncargs + chars_read, sizeof oldDevPtr);
      chars_read += sizeof oldDevPtr;
      size_t len;
      memcpy(&len, l->fncargs + chars_read, sizeof len);
      CUdeviceptr *newDevPtr = NULL;
      CUresult ret = cuMemAlloc_v2(newDevPtr, len);
      assert(ret == CUDA_SUCCESS);
      // JASSERT(ret == cudaSuccess) ("cudaMalloc replay failed!");
      // JASSERT(newDevPtr == oldDevPtr) (oldDevPtr) (newDevPtr)
      //   .Text("new device pointer is different than old one!");
      break;
    }
    case GENERATE_ENUM(cudaMallocManaged):
    {
      void *oldDevPtr;
      memcpy(&oldDevPtr, l->fncargs + chars_read, sizeof oldDevPtr);
      chars_read += sizeof oldDevPtr;

      size_t len;
      memcpy(&len, l->fncargs + chars_read, sizeof len);
      chars_read += sizeof len;

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);

      void *newDevPtr = NULL;
      cudaError_t ret = cudaMallocManaged(&newDevPtr, len, flags);
      assert(ret == cudaSuccess);
      // JASSERT(ret == cudaSuccess) ("cudaMalloc replay failed!");
      // JASSERT(newDevPtr == oldDevPtr) (oldDevPtr) (newDevPtr)
      //   .Text("new device pointer is different than old one!");
      break;
    }
    case GENERATE_ENUM(cuMemAllocManaged):
    {
      CUdeviceptr * oldDevPtr;
      memcpy(&oldDevPtr, l->fncargs + chars_read, sizeof oldDevPtr);
      chars_read += sizeof oldDevPtr;

      size_t len;
      memcpy(&len, l->fncargs + chars_read, sizeof len);
      chars_read += sizeof len;

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);

      CUdeviceptr * newDevPtr = NULL;
      CUresult ret = cuMemAllocManaged(newDevPtr, len, flags);
      assert(ret == CUDA_SUCCESS);
      // JASSERT(ret == cudaSuccess) ("cudaMalloc replay failed!");
      // JASSERT(newDevPtr == oldDevPtr) (oldDevPtr) (newDevPtr)
      //   .Text("new device pointer is different than old one!");
      break;
    }
    case GENERATE_ENUM(cudaFree):
    {
      void *devPtr;
      memcpy(&devPtr, l->fncargs + chars_read, sizeof devPtr);
      // cudaError_t ret = cudaFree(devPtr);
      cudaFree(devPtr);
      // JASSERT(ret == cudaSuccess) ("cudaFree replay failed!");
      break;
    }
    case GENERATE_ENUM(cuMemFree_v2):
    {
      // args
      CUdeviceptr devPtr;
      memcpy(&devPtr, l->fncargs + chars_read, sizeof devPtr);
      // cudaError_t ret = cudaFree(devPtr);
      cuMemFree_v2(devPtr);
      // JASSERT(ret == cudaSuccess) ("cudaFree replay failed!");
      break;
    }
    case GENERATE_ENUM(__cudaInitModule):
    {
      void  *fatCubinHandle;
      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);
      char oldRes;
      memcpy(&oldRes, l->fncargs + chars_read, sizeof oldRes);
      // replay
      // char newRes;
      // newRes =
      // __cudaInitModule(&fatCubinHandle);
      __cudaInitModule(new_fatCubinHandle);
      // JASSERT(memcmp(&oldRes, &newRes, sizeof(oldRes))!= 0)
      //   .Text("old and new desc are not same!");
      break;
    }
    case GENERATE_ENUM(__cudaPopCallConfiguration):
    {
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      void *stream;
      memcpy(&gridDim, l->fncargs + chars_read, sizeof gridDim);
      chars_read += sizeof gridDim;
      memcpy(&blockDim, l->fncargs + chars_read, sizeof blockDim);
      chars_read += sizeof blockDim;
      memcpy(&sharedMem, l->fncargs + chars_read, sizeof sharedMem);
      chars_read += sizeof sharedMem;
      memcpy(&stream, l->fncargs + chars_read, sizeof (void *));
      // replay
      // cudaError_t ret =
      __cudaPopCallConfiguration(&gridDim, &blockDim,
                          &sharedMem, stream);
      // JASSERT(ret == cudaSuccess)
      //   .Text("__cudaPopCallConfiguration replay failed");
      break;
    }
    case GENERATE_ENUM(__cudaPushCallConfiguration):
    {
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      void *stream;
      memcpy(&gridDim, l->fncargs + chars_read, sizeof gridDim);
      chars_read += sizeof gridDim;
      memcpy(&blockDim, l->fncargs + chars_read, sizeof blockDim);
      chars_read += sizeof blockDim;
      memcpy(&sharedMem, l->fncargs + chars_read, sizeof sharedMem);
      chars_read += sizeof sharedMem;
      memcpy(&stream, l->fncargs + chars_read, sizeof (void *));
      // replay
      typedef unsigned int (*pushFptr_t)(dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream);
      Cuda_Fncs_t fnc = Cuda_Fnc___cudaPushCallConfiguration;
      pushFptr_t func = (pushFptr_t)lhDlsym(fnc);
      func(gridDim, blockDim,sharedMem, stream);
      break;
    }
    case GENERATE_ENUM(__cudaRegisterFatBinary):
    {
      void * fatCubin;
      memcpy(&fatCubin, l->fncargs + chars_read, sizeof(void *));
      chars_read += sizeof (void *);
      void *oldRes;
      memcpy(&oldRes, l->fncargs + chars_read, sizeof(void *));
      // replay
      void  **newRes = __cudaRegisterFatBinary(fatCubin);
      printf("\n old fatcubinhandle = %p\n", oldRes);
      printf("fatcubinhandle = %p\n", newRes);
      new_fatCubinHandle = newRes;
      // JASSERT(memcmp(&oldRes, *newRes, sizeof(*newRes))!= 0)
      //   .Text("old and new results are not same!");
      break;
    }
  case GENERATE_ENUM(__cudaRegisterFatBinaryEnd):
    {
      // replay
      // This call was introduced in CUDA 10.2
      // CUDA 10.2 will fail without this call
      __cudaRegisterFatBinaryEnd(new_fatCubinHandle);
      break;
    }

    case GENERATE_ENUM(__cudaUnregisterFatBinary):
    {
      void *fatCubinHandle;
      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof(void *));
      // replay
      // __cudaUnregisterFatBinary(&fatCubinHandle);
      __cudaUnregisterFatBinary(new_fatCubinHandle);
      // JTRACE(" __cudaUnregisterFatBinary replayed");
      break;
    }
    case GENERATE_ENUM(__cudaRegisterFunction):
    {
      void **fatCubinHandle;
      // int hostFunLen;
      // int deviceFunLen;
      // int deviceNameLen;
      int thread_limit;
      uint3 tid;
      uint3 bid;
      dim3 bDim;
      dim3 gDim;
      int wSize;

      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      // memcpy(&hostFunLen, l->fncargs + chars_read, sizeof (hostFunLen));
      // chars_read += sizeof (hostFunLen);
      // char *hostFun = (char *)malloc(hostFunLen);
      // memcpy(hostFun, l->fncargs + chars_read, hostFunLen);
      // chars_read += hostFunLen;
      char *hostFun;
      memcpy(&hostFun, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);


      // memcpy(&deviceFunLen, l->fncargs + chars_read, sizeof (deviceFunLen));
      // chars_read += sizeof (deviceFunLen);
      // char *deviceFun = (char *)malloc(deviceFunLen);
      // memcpy(deviceFun, l->fncargs + chars_read, deviceFunLen);
      // chars_read += deviceFunLen;
      char *deviceFun;
      memcpy(&deviceFun, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);


      // memcpy(&deviceNameLen, l->fncargs + chars_read, sizeof (deviceNameLen));
      // chars_read += sizeof (deviceNameLen);
      // char *deviceName = (char *)malloc(deviceNameLen);
      // memcpy(deviceName, l->fncargs + chars_read, deviceNameLen);
      // chars_read += deviceNameLen;
      char *deviceName;
      memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);


      memcpy(&thread_limit, l->fncargs + chars_read, sizeof (thread_limit));
      chars_read += sizeof (thread_limit);

      memcpy(&tid, l->fncargs + chars_read, sizeof tid);
      chars_read += sizeof (tid);

      memcpy(&bid, l->fncargs + chars_read, sizeof bid);
      chars_read += sizeof (bid);

      memcpy(&bDim, l->fncargs + chars_read, sizeof bDim);
      chars_read += sizeof (bDim);

      memcpy(&gDim, l->fncargs + chars_read, sizeof gDim);
      chars_read += sizeof (gDim);

      memcpy(&wSize, l->fncargs + chars_read, sizeof wSize);

      // replay
      // __cudaRegisterFunction(&fatCubinHandle, hostFun, deviceFun,
      //   deviceName, thread_limit, &tid, &bid, &bDim, &gDim, &wSize);
      __cudaRegisterFunction(new_fatCubinHandle, hostFun, deviceFun,
        deviceName, thread_limit, &tid, &bid, &bDim, &gDim, &wSize);
      // JTRACE("__cudaRegisterFunction replayed");
      break;
    }
    case GENERATE_ENUM(__cudaRegisterVar):
    {
      void **fatCubinHandle;
      char *hostVar;
      int ext;
      size_t size;
      int constant;
      int global;

      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      memcpy(&hostVar, l->fncargs + chars_read, sizeof (char *));
      chars_read += sizeof (char *);

      char *deviceAddress;
      memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);

      char *deviceName;
      memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);

      memcpy(&ext, l->fncargs + chars_read, sizeof ext);
      chars_read += sizeof (ext);

      memcpy(&size, l->fncargs + chars_read, sizeof size);
      chars_read += sizeof (size);

      memcpy(&constant, l->fncargs + chars_read, sizeof constant);
      chars_read += sizeof (constant);

      memcpy(&global, l->fncargs + chars_read, sizeof global);

      // replay
      __cudaRegisterVar(new_fatCubinHandle, hostVar,
                                deviceAddress, deviceName,
                                ext, size, constant, global);
      // JTRACE("__cudaRegisterVar replayed");
      break;
    }
    case GENERATE_ENUM(__cudaRegisterManagedVar):
    {
      void **fatCubinHandle;
      void **hostVarPtrAddress;
      int ext;
      size_t size;
      int constant;
      int global;

      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      memcpy(&hostVarPtrAddress, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      char *deviceAddress;
      memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);

      char *deviceName;
      memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);

      memcpy(&ext, l->fncargs + chars_read, sizeof ext);
      chars_read += sizeof (ext);

      memcpy(&size, l->fncargs + chars_read, sizeof size);
      chars_read += sizeof (size);

      memcpy(&constant, l->fncargs + chars_read, sizeof constant);
      chars_read += sizeof (constant);

      memcpy(&global, l->fncargs + chars_read, sizeof global);

      // replay
      __cudaRegisterManagedVar(new_fatCubinHandle, hostVarPtrAddress,
                                deviceAddress, deviceName,
                                ext, size, constant, global);
      // JTRACE("__cudaRegisterVar replayed");
      break;
    }
    case GENERATE_ENUM(__cudaRegisterTexture):
    {
      void  **fatCubinHandle;
      struct textureReference *hostVar;
      const void **deviceAddress;
      int dim;
      int norm;
      int ext;
      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      memcpy(&hostVar, l->fncargs + chars_read, sizeof (hostVar));
      chars_read += sizeof (hostVar);

      memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(void *));
      chars_read += sizeof (void *);

      char *deviceName;
      memcpy(&deviceName, l->fncargs + chars_read, sizeof(char *));
      chars_read += sizeof(char *);

      memcpy(&dim, l->fncargs + chars_read, sizeof dim);
      chars_read += sizeof (dim);

      memcpy(&norm, l->fncargs + chars_read, sizeof norm);
      chars_read += sizeof (norm);

      memcpy(&ext, l->fncargs + chars_read, sizeof ext);

      // replay
      __cudaRegisterTexture(new_fatCubinHandle, hostVar, deviceAddress,
                            deviceName, dim, norm, ext);
      // JTRACE("__cudaRegisterTexture replayed");
      break;
    }
    case GENERATE_ENUM(__cudaRegisterSurface):
    {
      void  **fatCubinHandle;
      struct surfaceReference *hostVar;
      const void **deviceAddress;
      int dim;
      int ext;
      memcpy(&fatCubinHandle, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      memcpy(&hostVar, l->fncargs + chars_read, sizeof (hostVar));
      chars_read += sizeof (hostVar);

      memcpy(&deviceAddress, l->fncargs + chars_read, sizeof(void *));
      chars_read += sizeof (void *);

      char *deviceName;
      memcpy(&deviceName, l->fncargs + chars_read, sizeof(deviceName));
      chars_read += sizeof(deviceName);

      memcpy(&dim, l->fncargs + chars_read, sizeof dim);
      chars_read += sizeof (dim);

      memcpy(&ext, l->fncargs + chars_read, sizeof ext);

      // replay
      __cudaRegisterSurface(new_fatCubinHandle, hostVar, deviceAddress,
                            deviceName, dim, ext);
      // JTRACE("__cudaRegisterSurface replayed");
      break;
    }
    case GENERATE_ENUM(cudaCreateTextureObject):
    {
      // args
      cudaTextureObject_t * pTexObject;
      memcpy(&pTexObject, l->fncargs + chars_read, sizeof(pTexObject));
      chars_read += sizeof(pTexObject);

      struct cudaResourceDesc * pResDesc;
      memcpy(&pResDesc, l->fncargs + chars_read, sizeof(pResDesc));
      chars_read += sizeof(pResDesc);

      struct cudaTextureDesc * pTexDesc;
      memcpy(&pTexDesc, l->fncargs + chars_read, sizeof(pTexDesc));
      chars_read += sizeof(&pTexDesc);

      struct cudaResourceViewDesc * pResViewDesc;
      memcpy(&pResViewDesc, l->fncargs + chars_read, sizeof(pResViewDesc));
      cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
      break;
    }
    case GENERATE_ENUM(cudaDestroyTextureObject):
    {
      // args
      cudaTextureObject_t texObject;
      memcpy(&texObject, l->fncargs + chars_read, sizeof(texObject));

      cudaDestroyTextureObject(texObject);
      break;
    }
    case GENERATE_ENUM(cudaBindTextureToArray):
    {
      textureReference* texref;
      memcpy(&texref, l->fncargs + chars_read, sizeof(texref));
      chars_read += sizeof(texref);

      cudaArray_const_t array;
      memcpy(&array, l->fncargs + chars_read, sizeof(array));
      chars_read += sizeof(array);

      cudaChannelFormatDesc * desc;
      memcpy(&desc, l->fncargs + chars_read, sizeof(desc));
      chars_read += sizeof(desc);
      cudaBindTextureToArray(texref, array, desc);
      break;
    }
    case GENERATE_ENUM(cudaUnbindTexture):
    {
      struct textureReference * texref;
      memcpy(&texref, l->fncargs + chars_read, sizeof(texref));
      chars_read += sizeof(texref);
      cudaUnbindTexture(texref);
      break;
    }
    case GENERATE_ENUM(cudaCreateChannelDesc):
    {
      int x,y,z,w;
      memcpy(&x, l->fncargs + chars_read, sizeof x);
      chars_read += sizeof x;
      memcpy(&y, l->fncargs + chars_read, sizeof y);
      chars_read += sizeof y;
      memcpy(&z, l->fncargs + chars_read, sizeof z);
      chars_read += sizeof z;
      memcpy(&w, l->fncargs + chars_read, sizeof w);
      chars_read += sizeof w;
      cudaChannelFormatKind f;
      memcpy(&f, l->fncargs + chars_read, sizeof f);
      cudaChannelFormatDesc oldDesc;
      memcpy(&oldDesc, l->results, sizeof oldDesc);
      // replay
      // cudaChannelFormatDesc newDesc =
      cudaCreateChannelDesc(x, y, z, w, f);
      // JASSERT(memcmp(&oldDesc, &newDesc, sizeof(oldDesc))!= 0)
      //   .Text("old and new desc are not same!");
      break;
    }
    case GENERATE_ENUM(cudaMallocArray):
    {
      // TODO : check the cudaMallocArray log and replay
      // args
      cudaArray_t * array;
      struct cudaChannelFormatDesc * desc;
      size_t width;
      size_t height;
      unsigned int flags;

      memcpy(&array, l->fncargs + chars_read, sizeof array);
      chars_read += sizeof (array);

      memcpy(&desc, l->fncargs + chars_read, sizeof desc);
      chars_read += sizeof (desc);

      memcpy(&width, l->fncargs + chars_read, sizeof width);
      chars_read += sizeof (width);

      memcpy(&height, l->fncargs + chars_read, sizeof height);
      chars_read += sizeof (height);

      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      cudaMallocArray(array, desc, width, height, flags);
      break;
    }
    case GENERATE_ENUM(cudaFreeArray):
    {
      // args
      cudaArray_t array;
      memcpy(&array, l->fncargs + chars_read, sizeof(array));

      cudaFreeArray(array);
      break;
    }
    case GENERATE_ENUM(cudaMallocHost):
    {
      //args
      void ** ptr;
      size_t size;
      memcpy(&ptr, l->fncargs + chars_read, sizeof(void *));
      chars_read += sizeof(void *);

      memcpy(&size, l->fncargs + chars_read, sizeof size);
      cudaMallocHost(ptr, size);
      break;
    }
    case GENERATE_ENUM(cuMemAllocHost_v2):
    {
      //args
      void ** ptr;
      size_t size;
      memcpy(&ptr, l->fncargs + chars_read, sizeof(void *));
      chars_read += sizeof(void *);

      memcpy(&size, l->fncargs + chars_read, sizeof size);
      cuMemAllocHost_v2(ptr, size);
      break;
    }
    case GENERATE_ENUM(cudaFreeHost):
    {
      void *ptr;
      memcpy(&ptr, l->fncargs + chars_read, sizeof(void *));
      cudaFreeHost(ptr);
      break;
    }
    case GENERATE_ENUM(cuMemFreeHost):
    {
      void *ptr;
      memcpy(&ptr, l->fncargs + chars_read, sizeof(void *));
      cuMemFreeHost(ptr);
      break;
    }
    case GENERATE_ENUM(cudaHostAlloc):
    {
      // before replaying the host alloc
      void **ptr;
      size_t size;
      unsigned int flags;
      memcpy(&ptr, l->fncargs + chars_read, sizeof (ptr));
      chars_read += sizeof (ptr);

      memcpy(&size, l->fncargs + chars_read, sizeof size);
      chars_read += sizeof (size);

      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      cudaHostAlloc(ptr, size , flags);
      break;
    }
    case GENERATE_ENUM(cuMemHostAlloc):
    {
      // before replaying the host alloc
      void **ptr;
      size_t size;
      unsigned int flags;
      memcpy(&ptr, l->fncargs + chars_read, sizeof (ptr));
      chars_read += sizeof (ptr);

      memcpy(&size, l->fncargs + chars_read, sizeof size);
      chars_read += sizeof (size);

      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      cuMemHostAlloc(ptr, size , flags);
      break;
    }
    case GENERATE_ENUM(cudaDeviceReset):
    {
      // no arguments to read from buffer
      cudaDeviceReset();
      break;
    }
    case GENERATE_ENUM(cudaMallocPitch):
    {
      void ** devPtr;
      memcpy(&devPtr, l->fncargs + chars_read, sizeof(devPtr));
      chars_read += sizeof(devPtr);

      size_t * pitch;
      memcpy(&pitch, l->fncargs + chars_read, sizeof pitch);
      chars_read += sizeof(pitch);

      size_t width;
      memcpy(&width, l->fncargs + chars_read, sizeof(width));
      chars_read += sizeof(width);

      size_t height;
      memcpy(&height, l->fncargs + chars_read, sizeof(height));
      chars_read += sizeof(height);
      cudaMallocPitch(devPtr, pitch, width, height);
      break;
    }
    case GENERATE_ENUM(cuMemAllocPitch_v2):
    {
      CUdeviceptr* devPtr;
      memcpy(&devPtr, l->fncargs + chars_read, sizeof(devPtr));
      chars_read += sizeof(devPtr);

      size_t * pitch;
      memcpy(&pitch, l->fncargs + chars_read, sizeof pitch);
      chars_read += sizeof(pitch);

      size_t width;
      memcpy(&width, l->fncargs + chars_read, sizeof(width));
      chars_read += sizeof(width);

      size_t height;
      memcpy(&height, l->fncargs + chars_read, sizeof(height));
      chars_read += sizeof(height);

      unsigned int ElementSizeBytes;
      memcpy(&ElementSizeBytes, l->fncargs + chars_read, sizeof(ElementSizeBytes));
      chars_read += sizeof(ElementSizeBytes);
      cuMemAllocPitch_v2(devPtr, pitch, width, height, ElementSizeBytes);
      break;
    }
    case GENERATE_ENUM(cudaDeviceSynchronize):
    {
      // no arguments to read from buffer
      cudaDeviceSynchronize();
      break;
    }
    //sth about cudaStreamCreate
    //if it is used in a program it will affect
    //the deterministism of cudaMalloc
    //mrCUDA
    case GENERATE_ENUM(cudaStreamCreate):
    {
      cudaStream_t *pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof pStream);
      cudaStreamCreate(pStream);
      break;
    }
    case GENERATE_ENUM(cuStreamCreate):
    {
      CUstream *pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof pStream);
      chars_read += sizeof(pStream);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      cuStreamCreate(pStream, flags);
      break;
    }
    case GENERATE_ENUM(cudaStreamCreateWithFlags):
    {
      cudaStream_t *pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof pStream);
      chars_read += sizeof(pStream);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      cudaStreamCreateWithFlags(pStream, flags);
      break;
    }
    case GENERATE_ENUM(cudaStreamCreateWithPriority):
    {
      cudaStream_t *pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof pStream);
      chars_read += sizeof(pStream);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      chars_read += sizeof(flags);

      int priority;
      memcpy(&priority, l->fncargs + chars_read, sizeof priority);
      chars_read += sizeof(priority);
      cudaStreamCreateWithPriority(pStream, flags, priority);
      break;
    }
    case GENERATE_ENUM(cuStreamCreateWithPriority):
    {
      CUstream *pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof pStream);
      chars_read += sizeof(pStream);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      chars_read += sizeof(flags);

      int priority;
      memcpy(&priority, l->fncargs + chars_read, sizeof priority);
      chars_read += sizeof(priority);
      cuStreamCreateWithPriority(pStream, flags, priority);
      break;
    }
    case GENERATE_ENUM(cudaStreamDestroy):
    {
      cudaStream_t pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof(pStream));
      cudaStreamDestroy(pStream);
      break;
    }
    case GENERATE_ENUM(cuStreamDestroy_v2):
    {
      CUstream pStream;
      memcpy(&pStream, l->fncargs + chars_read, sizeof(pStream));
      cuStreamDestroy_v2(pStream);
      break;
    }
    case GENERATE_ENUM(cudaEventCreate):
    {
      cudaEvent_t * event;
      memcpy(&event, l->fncargs + chars_read, sizeof(event));
      chars_read += sizeof(event);
      cudaEventCreate(event);
      break;
    }
    case GENERATE_ENUM(cuEventCreate):
    {
      CUevent * event;
      memcpy(&event, l->fncargs + chars_read, sizeof(event));
      chars_read += sizeof(event);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      chars_read += sizeof(flags);
      cuEventCreate(event, flags);
      break;
    }
    case GENERATE_ENUM(cudaEventDestroy):
    {
      cudaEvent_t event;
      memcpy(&event, l->fncargs + chars_read, sizeof(event));
      chars_read += sizeof(event);
      cudaEventDestroy(event);
      break;
    }
    case GENERATE_ENUM(cuEventDestroy_v2):
    {
      CUevent event;
      memcpy(&event, l->fncargs + chars_read, sizeof(event));
      chars_read += sizeof(event);
      cuEventDestroy_v2(event);
      break;
    }
    case GENERATE_ENUM(cudaEventCreateWithFlags):
    {
      cudaEvent_t * event;
      memcpy(&event, l->fncargs + chars_read, sizeof(event));
      chars_read += sizeof(event);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      chars_read += sizeof(flags);
      cudaEventCreateWithFlags(event, flags);
      break;
    }
    case GENERATE_ENUM(cuDestroyExternalMemory):
    {
      CUexternalMemory extMem;
      memcpy(&extMem, l->fncargs + chars_read, sizeof(extMem));
      chars_read += sizeof(extMem);
      cuDestroyExternalMemory(extMem);
      break;
    }
    case GENERATE_ENUM(cuDestroyExternalSemaphore):
    {
      CUexternalSemaphore extSem;
      memcpy(&extSem, l->fncargs + chars_read, sizeof(extSem));
      chars_read += sizeof(extSem);
      cuDestroyExternalSemaphore(extSem);
      break;
    }
    case GENERATE_ENUM(cuGraphCreate):
    {
      CUgraph * phGraph;
      memcpy(&phGraph, l->fncargs + chars_read, sizeof(phGraph));
      chars_read += sizeof(phGraph);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof flags);
      chars_read += sizeof(flags);
      cuGraphCreate(phGraph, flags);
      break;
    }
    case GENERATE_ENUM(cuGraphDestroy):
    {
      CUgraph hGraph;
      memcpy(&hGraph, l->fncargs + chars_read, sizeof(hGraph));
      chars_read += sizeof(hGraph);
      cuGraphDestroy(hGraph);
      break;
    }
    case GENERATE_ENUM(cuGraphDestroyNode):
    {
      CUgraphNode hNode;
      memcpy(&hNode, l->fncargs + chars_read, sizeof(hNode));
      chars_read += sizeof(hNode);
      cuGraphDestroyNode(hNode);
      break;
    }
    case GENERATE_ENUM(cuGraphExecDestroy):
    {
      CUgraphExec hGraphExec;
      memcpy(&hGraphExec, l->fncargs + chars_read, sizeof(hGraphExec));
      chars_read += sizeof(hGraphExec);
      cuGraphExecDestroy(hGraphExec);
      break;
    }
    case GENERATE_ENUM(cuTexRefCreate):
    {
      CUtexref *pTexRef;
      memcpy(&pTexRef, l->fncargs + chars_read, sizeof(pTexRef));
      chars_read += sizeof(pTexRef);
      cuTexRefCreate(pTexRef);
      break;
    }
    case GENERATE_ENUM(cuTexRefDestroy):
    {
      CUtexref hTexRef;
      memcpy(&hTexRef, l->fncargs + chars_read, sizeof(hTexRef));
      chars_read += sizeof(hTexRef);
      cuTexRefDestroy(hTexRef);
      break;
    }
    case GENERATE_ENUM(cuTexObjectCreate):
    {
      CUtexObject *pTexObject;
      memcpy(&pTexObject, l->fncargs + chars_read, sizeof(pTexObject));
      chars_read += sizeof(pTexObject);

      CUDA_RESOURCE_DESC* pResDesc;
      memcpy(&pResDesc, l->fncargs + chars_read, sizeof(pResDesc));
      chars_read += sizeof(pResDesc);

      CUDA_TEXTURE_DESC* pTexDesc;
      memcpy(&pTexDesc, l->fncargs + chars_read, sizeof(pTexDesc));
      chars_read += sizeof(pTexDesc);

      CUDA_RESOURCE_VIEW_DESC* pResViewDesc;
      memcpy(&pResViewDesc, l->fncargs + chars_read, sizeof(pResViewDesc));
      chars_read += sizeof(pResViewDesc);
      cuTexObjectCreate(pTexObject, pResDesc, \
                        (const CUDA_TEXTURE_DESC*) pTexDesc, \
                        (const CUDA_RESOURCE_VIEW_DESC*)pResViewDesc);
      break;
    }
    case GENERATE_ENUM(cuTexObjectDestroy):
    {
      CUtexObject pTexObject;
      memcpy(&pTexObject, l->fncargs + chars_read,sizeof(pTexObject));
      chars_read += sizeof(pTexObject);
      cuTexObjectDestroy(pTexObject);
      break;
    }
    case GENERATE_ENUM(cuSurfObjectCreate):
    {
      CUsurfObject *pSurfObject;
      memcpy(&pSurfObject, l->fncargs + chars_read, sizeof(pSurfObject));
      chars_read += sizeof(pSurfObject);

      CUDA_RESOURCE_DESC *pResDesc;
      memcpy(&pResDesc, l->fncargs + chars_read, sizeof(pResDesc));
      chars_read += sizeof(pResDesc);
      cuSurfObjectCreate(pSurfObject, pResDesc);
      break;
    }
    case GENERATE_ENUM(cuSurfObjectDestroy):
    {
      CUsurfObject pSurfObject;
      memcpy(&pSurfObject, l->fncargs + chars_read, sizeof(pSurfObject));
      chars_read += sizeof(pSurfObject);
      cuSurfObjectDestroy(pSurfObject);
      break;
    }
    case GENERATE_ENUM(cublasCreate_v2):
    {
      cublasHandle_t *handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cublasCreate_v2(handle);
      break;
    }
    case GENERATE_ENUM(cublasDestroy_v2):
    {
      cublasHandle_t handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cublasDestroy_v2(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreate):
    {
      cusparseHandle_t *handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusparseCreate(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroy):
    {
      cusparseHandle_t handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusparseDestroy(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreateMatDescr):
    {
      cusparseMatDescr_t *handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusparseCreateMatDescr(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroyMatDescr):
    {
      cusparseMatDescr_t handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusparseDestroyMatDescr(handle);
      break;
    }
#if CUDA_VERSION <= 10020
    case GENERATE_ENUM(cusparseCreateHybMat):
    {
      cusparseHybMat_t *handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusparseCreateHybMat(handle);
      break;
    }
    case GENERATE_ENUM(cusparseDestroyHybMat):
    {
      cusparseHybMat_t handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusparseDestroyHybMat(handle);
      break;
    }
    case GENERATE_ENUM(cusparseCreateSolveAnalysisInfo):
    {
      cusparseSolveAnalysisInfo_t *info;
      memcpy(&info, l->fncargs + chars_read, sizeof(info));
      chars_read += sizeof(info);
      cusparseCreateSolveAnalysisInfo(info);
      break;
    }
    case GENERATE_ENUM(cusparseDestroySolveAnalysisInfo):
    {
      cusparseSolveAnalysisInfo_t info;
      memcpy(&info, l->fncargs + chars_read, sizeof(info));
      chars_read += sizeof(info);
      cusparseDestroySolveAnalysisInfo(info);
      break;
    }
#endif
    case GENERATE_ENUM(cusolverDnCreate):
    {
      cusolverDnHandle_t *handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusolverDnCreate(handle);
      break;
    }
    case GENERATE_ENUM(cusolverDnDestroy):
    {
      cusolverDnHandle_t handle;
      memcpy(&handle, l->fncargs + chars_read, sizeof(handle));
      chars_read += sizeof(handle);
      cusolverDnDestroy(handle);
      break;
    }
    case GENERATE_ENUM(cuCtxCreate_v2):
    {
      CUcontext *pctx;
      memcpy(&pctx, l->fncargs + chars_read, sizeof(pctx));
      chars_read += sizeof(pctx);

      unsigned int flags;
      memcpy(&flags, l->fncargs + chars_read, sizeof(flags));
      chars_read += sizeof(flags);

      CUdevice dev;
      memcpy(&dev, l->fncargs + chars_read, sizeof(dev));
      chars_read += sizeof(dev);
      cuCtxCreate_v2(pctx, flags, dev);
      break;
    }
    case GENERATE_ENUM(cuCtxDestroy_v2):
    {
      CUcontext pctx;
      memcpy(&pctx, l->fncargs + chars_read, sizeof(pctx));
      chars_read += sizeof(pctx);
      cuCtxDestroy_v2(pctx);
      break;
    }
    case GENERATE_ENUM(cuLinkCreate_v2):
    {
      unsigned int numOptions;
      memcpy(&numOptions, l->fncargs + chars_read, sizeof(numOptions));
      chars_read += sizeof(numOptions);

      CUjit_option * options;
      memcpy(&options, l->fncargs + chars_read, sizeof(options));
      chars_read += sizeof(options);

      void** optionValues;
      memcpy(&optionValues, l->fncargs + chars_read, sizeof(optionValues));
      chars_read += sizeof(optionValues);
      // out parameter
      CUlinkState * stateOut;
      memcpy(&stateOut, l->fncargs + chars_read, sizeof(stateOut));
      chars_read += sizeof(stateOut);
      cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
      break;
    }
    case GENERATE_ENUM(cuLinkDestroy):
    {
      CUlinkState  stateOut;
      memcpy(&stateOut, l->fncargs + chars_read, sizeof(stateOut));
      chars_read += sizeof(stateOut);
      cuLinkDestroy(stateOut);
      break;
    }
    case GENERATE_ENUM(cuArray3DCreate_v2):
    {
      CUarray* pHandle;
      memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
      chars_read += sizeof(pHandle);

      CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray;
      memcpy(&pAllocateArray, l->fncargs + chars_read, sizeof(pAllocateArray));
      chars_read += sizeof(pAllocateArray);
      cuArray3DCreate_v2(pHandle, pAllocateArray);
      break;
    }
    case GENERATE_ENUM(cuArrayCreate_v2):
    {
      CUarray* pHandle;
      memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
      chars_read += sizeof(pHandle);

      CUDA_ARRAY_DESCRIPTOR* pAllocateArray;
      memcpy(&pAllocateArray, l->fncargs + chars_read, sizeof(pAllocateArray));
      chars_read += sizeof(pAllocateArray);
      cuArrayCreate_v2(pHandle, pAllocateArray);
      break;
    }
    case GENERATE_ENUM(cuArrayDestroy):
    {
      CUarray pHandle;
      memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
      chars_read += sizeof(pHandle);
      cuArrayDestroy(pHandle);
      break;
    }
    case GENERATE_ENUM(cuMipmappedArrayCreate):
    {
      CUmipmappedArray* pHandle;
      memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
      chars_read += sizeof(pHandle);

      CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc;
      memcpy(&pMipmappedArrayDesc, l->fncargs + chars_read, sizeof(pMipmappedArrayDesc));
      chars_read += sizeof(pMipmappedArrayDesc);

      unsigned int numMipmapLevels;
      memcpy(&numMipmapLevels, l->fncargs + chars_read, sizeof(numMipmapLevels));
      chars_read += sizeof(numMipmapLevels);
      cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels);
      break;
    }
    case GENERATE_ENUM(cuMipmappedArrayDestroy):
    {
      CUmipmappedArray pHandle;
      memcpy(&pHandle, l->fncargs + chars_read, sizeof(pHandle));
      chars_read += sizeof(pHandle);
      cuMipmappedArrayDestroy(pHandle);
      break;
    }
/*    case GENERATE_ENUM(cudaLaunchKernel):
    {
      void * func_addr;
      void **args;
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      cudaStream_t stream;
      memcpy(&func_addr, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      memcpy(&gridDim, l->fncargs + chars_read, sizeof gridDim);
      chars_read += sizeof gridDim;

      memcpy(&blockDim, l->fncargs + chars_read, sizeof blockDim);
      chars_read += sizeof blockDim;

      memcpy(&args, l->fncargs + chars_read, sizeof (void *));
      chars_read += sizeof (void *);

      memcpy(&sharedMem, l->fncargs + chars_read, sizeof sharedMem);
      chars_read += sizeof sharedMem;

      memcpy(&stream, l->fncargs + chars_read, sizeof (void *));
      // replay
      cudaLaunchKernel(func_addr, gridDim, blockDim, args, sharedMem, stream);
      break;
    } */
    default:
      assert(false);
     break;
      // JASSERT(false)(op).Text("Replaying unknown op code");
  }
}
//getter for fatCubinHandle generated by replayed __cudaRegisterFatBinary
void** fatHandle(){
	return new_fatCubinHandle;
}
// This function iterates over the CUDA calls log and calls the given
// function on each call log object
// void logs_read_and_apply(void (*apply)(CudaCallLog_t *l))
void logs_read_and_apply()
{ 
  GetCudaCallsLogFptr_t fnc = (GetCudaCallsLogFptr_t)uhInfo.cudaLogVectorFptr;
  std::vector<CudaCallLog_t>& cudaCallsLog = fnc();
  for (auto it = cudaCallsLog.begin(); it != cudaCallsLog.end(); it++) {
    replayAPI(&(*it));
  }
}
