#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

extern "C" cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const struct cudaResourceDesc * pResDesc, const struct cudaTextureDesc * pTexDesc, const struct cudaResourceViewDesc * pResViewDesc) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaBindTexture ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaBindTexture2D ( size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, size_t pitch ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaBindTextureToArray ( const textureReference* texref, cudaArray_const_t array, const cudaChannelFormatDesc* desc ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaUnbindTexture(const struct textureReference * texref) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaChannelFormatDesc cudaCreateChannelDesc ( int  x, int  y, int  z, int  w, cudaChannelFormatKind f ) {
  assert(0);
  cudaChannelFormatDesc ret_val ;
  return ret_val;
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t * event) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMalloc(void ** pointer, size_t size) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaFree ( void * pointer ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMallocArray(struct cudaArray ** array, const struct cudaChannelFormatDesc * desc, size_t width, size_t height, unsigned int flags) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaFreeArray(struct cudaArray * array) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMallocHost ( void ** ptr , size_t size ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaFreeHost ( void* ptr ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaHostAlloc ( void ** ptr , size_t size , unsigned int flags ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaGetDevice(int * device) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaGetDeviceCount(int * count) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceReset() {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaDeviceSynchronize() {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemcpyToArray (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemset(void * devPtr, int value, size_t count) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int value, size_t width, size_t height ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemGetInfo(size_t * free, size_t * total) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemAdvise(const void * devPtr, size_t count, enum cudaMemoryAdvise advice, int device) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaStreamCreate(cudaStream_t * pStream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaThreadSynchronize () {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaThreadExit () {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaPointerGetAttributes ( cudaPointerAttributes* attributes, const void* ptr ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" const char* cudaGetErrorString ( cudaError_t error ) {
  assert(0);
  const char* ret_val = NULL;
  return ret_val;
}

extern "C" const char* cudaGetErrorName ( cudaError_t error ) {
  assert(0);
  const char* ret_val = NULL;
  return ret_val;
}

extern "C" cudaError_t cudaGetLastError() {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaPeekAtLastError() {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaFuncSetCacheConfig ( const void* func, cudaFuncCache cacheConfig ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
  assert(0);
  char ret_val = 0;
  return ret_val;
}

extern "C" cudaError_t __cudaPopCallConfiguration( dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" unsigned int __cudaPushCallConfiguration( dim3 gridDim, dim3 blockDim, size_t sharedMem, void * stream ) {
  assert(0);
  unsigned int ret_val = 0;
  return ret_val;
}

extern "C" void** __cudaRegisterFatBinary(void *fatCubin) {
  assert(0);
  void** ret_val = NULL;
  return ret_val;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  assert(0);
}

extern "C" void __cudaRegisterFunction( void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize ) {
  assert(0);
}

extern "C" void __cudaRegisterManagedVar( void **fatCubinHandle, void **hostVarPtrAddress, char  *deviceAddress, const char  *deviceName, int    ext, size_t size, int    constant, int    global ) {
  assert(0);
}

extern "C" void __cudaRegisterTexture( void  **fatCubinHandle, const struct textureReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext ) {
  assert(0);
}

extern "C" void __cudaRegisterSurface( void **fatCubinHandle, const struct surfaceReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext ) {
  assert(0);
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char  *deviceAddress, const char  *deviceName, int ext, size_t size, int constant, int global) {
  assert(0);
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags ) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
  assert(0);
  cudaError_t ret_val = cudaSuccess;
  return ret_val;
}

extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t * handle) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double * x, int incx, const double * y, int incy, double * result) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double * alpha, const double * x, int incx, double * y, int incy) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDgemm_v2 (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDgemv_v2 (cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasDswap_v2 (cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
  assert(0);
  cublasStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDestroyMatDescr (cusparseMatDescr_t descrA) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseCreateHybMat(cusparseHybMat_t *hybA) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDestroyHybMat(cusparseHybMat_t hybA) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDcsr2csc(cusparseHandle_t handle, int m, int n, int nnz, const double  *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, double *cscSortedVal, int *cscSortedRowInd, int *cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const double *alpha, const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *x, const double *beta, double *y) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDcsrsv_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, cusparseSolveAnalysisInfo_t info) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDcsrsv_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, const double *alpha, const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, cusparseSolveAnalysisInfo_t info, const double *f, double *x) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDhyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, double *csrSortedValA, int *csrSortedRowPtrA, int *csrSortedColIndA) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseDhybmv(cusparseHandle_t handle, cusparseOperation_t transA, const double *alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const double *x, const double *beta, double *y) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) {
  assert(0);
  cusparseMatrixType_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA) {
  assert(0);
  cusparseFillMode_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA) {
  assert(0);
  cusparseDiagType_t ret_val ;
  return ret_val;
}

extern "C" cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
  assert(0);
  cusparseIndexBase_t ret_val ;
  return ret_val;
}

extern "C" cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  assert(0);
  cusparseStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork ) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDgetrf( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo ) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo ) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork ) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo ) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" cusolverStatus_t cusolverDnDpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo) {
  assert(0);
  cusolverStatus_t ret_val ;
  return ret_val;
}

extern "C" CUresult cuInit ( unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDriverGetVersion ( int* driverVersion ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGet ( CUdevice* device, int  ordinal ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetCount ( int* count ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetUuid ( CUuuid* uuid, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceComputeCapability ( int* major, int* minor, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetProperties ( CUdevprop* prop, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDevicePrimaryCtxGetState ( CUdevice dev, unsigned int* flags, int* active ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDevicePrimaryCtxRelease ( CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDevicePrimaryCtxReset ( CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDevicePrimaryCtxRetain ( CUcontext* pctx, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDevicePrimaryCtxSetFlags ( CUdevice dev, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxCreate_v2 ( CUcontext* pctx, unsigned int  flags, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxDestroy_v2 ( CUcontext ctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetCurrent ( CUcontext* pctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetDevice ( CUdevice* device ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetFlags ( unsigned int* flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxPopCurrent_v2 ( CUcontext* pctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxPushCurrent_v2 ( CUcontext ctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxSetCacheConfig ( CUfunc_cache config ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxSetCurrent ( CUcontext ctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxSetLimit ( CUlimit limit, size_t value ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxSynchronize () {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxAttach ( CUcontext* pctx, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxDetach ( CUcontext ctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLinkAddData_v2 ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLinkAddFile_v2 ( CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLinkComplete ( CUlinkState state, void** cubinOut, size_t* sizeOut ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLinkCreate_v2 ( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLinkDestroy ( CUlinkState state ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleGetGlobal_v2 ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleLoad ( CUmodule* module, const char* fname ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleLoadData ( CUmodule* module, const void* image ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuModuleUnload ( CUmodule hmod ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuArray3DCreate_v2 ( CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuArray3DGetDescriptor_v2 ( CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuArrayCreate_v2 ( CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuArrayDestroy ( CUarray hArray ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuArrayGetDescriptor_v2 ( CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuIpcOpenMemHandle ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemAlloc_v2 ( CUdeviceptr* dptr, size_t bytesize ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemAllocHost_v2 ( void** pp, size_t bytesize ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemAllocPitch_v2 ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemFree_v2 ( CUdeviceptr dptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemFreeHost ( void* p ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemGetAddressRange_v2 ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemGetInfo_v2 ( size_t* free, size_t* total ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2 ( CUdeviceptr* pdptr, void* p, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemHostRegister_v2 ( void* p, size_t bytesize, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemHostUnregister ( void* p ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy2D_v2 ( const CUDA_MEMCPY2D* pCopy ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy2DAsync_v2 ( const CUDA_MEMCPY2D* pCopy, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy2DUnaligned_v2 ( const CUDA_MEMCPY2D* pCopy ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy3D_v2 ( const CUDA_MEMCPY3D* pCopy ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy3DAsync_v2 ( const CUDA_MEMCPY3D* pCopy, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyAtoA_v2 ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyAtoD_v2 ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyAtoH_v2 ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyAtoHAsync_v2 ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyDtoA_v2 ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyDtoD_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyDtoDAsync_v2 ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyDtoH_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyDtoHAsync_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyHtoA_v2 ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyHtoAAsync_v2 ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyHtoD_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyHtoDAsync_v2 ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD16_v2 ( CUdeviceptr dstDevice, unsigned short us, size_t N ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD2D16_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD2D32_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD2D8_v2 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD32_v2 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD8_v2 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMipmappedArrayGetLevel ( CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int  level ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemAdvise ( CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemPrefetchAsync ( CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemRangeGetAttribute ( void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuMemRangeGetAttributes ( void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamAddCallback ( CUstream hStream, CUstreamCallback callback, void* userData, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamAttachMemAsync ( CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamCreate ( CUstream* phStream, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamCreateWithPriority ( CUstream* phStream, unsigned int  flags, int  priority ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamDestroy_v2 ( CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamEndCapture ( CUstream hStream, CUgraph* phGraph ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamGetCtx ( CUstream hStream, CUcontext* pctx ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamGetFlags ( CUstream hStream, unsigned int* flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamGetPriority ( CUstream hStream, int* priority ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamIsCapturing ( CUstream hStream, CUstreamCaptureStatus* captureStatus ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamQuery ( CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamSynchronize ( CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamWaitEvent ( CUstream hStream, CUevent hEvent, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuEventCreate ( CUevent* phEvent, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuEventDestroy_v2 ( CUevent hEvent ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuEventElapsedTime ( float* pMilliseconds, CUevent hStart, CUevent hEnd ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuEventQuery ( CUevent hEvent ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuEventRecord ( CUevent hEvent, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuEventSynchronize ( CUevent hEvent ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDestroyExternalMemory ( CUexternalMemory extMem ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDestroyExternalSemaphore ( CUexternalSemaphore extSem ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuExternalMemoryGetMappedBuffer ( CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray ( CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuImportExternalMemory ( CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuImportExternalSemaphore ( CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuSignalExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuWaitExternalSemaphoresAsync ( const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int  numExtSems, CUstream stream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamBatchMemOp ( CUstream stream, unsigned int  count, CUstreamBatchMemOpParams* paramArray, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamWaitValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamWaitValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamWriteValue32 ( CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuStreamWriteValue64 ( CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuFuncGetAttribute ( int* pi, CUfunction_attribute attrib, CUfunction hfunc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuFuncSetAttribute ( CUfunction hfunc, CUfunction_attribute attrib, int  value ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuFuncSetSharedMemConfig ( CUfunction hfunc, CUsharedconfig config ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunchCooperativeKernelMultiDevice ( CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int  numDevices, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunchHostFunc ( CUstream hStream, CUhostFn fn, void* userData ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuFuncSetBlockShape ( CUfunction hfunc, int  x, int  y, int  z ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuFuncSetSharedSize ( CUfunction hfunc, unsigned int  bytes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunch ( CUfunction f ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunchGrid ( CUfunction f, int  grid_width, int  grid_height ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuLaunchGridAsync ( CUfunction f, int  grid_width, int  grid_height, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuParamSetSize ( CUfunction hfunc, unsigned int  numbytes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuParamSetTexRef ( CUfunction hfunc, int  texunit, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuParamSetf ( CUfunction hfunc, int  offset, float  value ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuParamSeti ( CUfunction hfunc, int  offset, unsigned int  value ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuParamSetv ( CUfunction hfunc, int  offset, void* ptr, unsigned int  numbytes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphCreate ( CUgraph* phGraph, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphDestroy ( CUgraph hGraph ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphDestroyNode ( CUgraphNode hNode ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphExecDestroy ( CUgraphExec hGraphExec ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphGetEdges ( CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphGetNodes ( CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphGetRootNodes ( CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphHostNodeGetParams ( CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphHostNodeSetParams ( CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphInstantiate ( CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphKernelNodeGetParams ( CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphKernelNodeSetParams ( CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphLaunch ( CUgraphExec hGraphExec, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphMemcpyNodeGetParams ( CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphMemcpyNodeSetParams ( CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphMemsetNodeGetParams ( CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphMemsetNodeSetParams ( CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphNodeFindInClone ( CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphNodeGetDependencies ( CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphNodeGetDependentNodes ( CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphNodeGetType ( CUgraphNode hNode, CUgraphNodeType* type ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefCreate ( CUtexref* pTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefDestroy ( CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetAddress_v2 ( CUdeviceptr* pdptr, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetAddressMode ( CUaddress_mode* pam, CUtexref hTexRef, int  dim ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetArray ( CUarray* phArray, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetBorderColor ( float* pBorderColor, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetFlags ( unsigned int* pFlags, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetFormat ( CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetMaxAnisotropy ( int* pmaxAniso, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetMipmapFilterMode ( CUfilter_mode* pfm, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetMipmapLevelBias ( float* pbias, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetMipmapLevelClamp ( float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefGetMipmappedArray ( CUmipmappedArray* phMipmappedArray, CUtexref hTexRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetAddress_v2 ( size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetAddress2D_v3 ( CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetAddressMode ( CUtexref hTexRef, int  dim, CUaddress_mode am ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetArray ( CUtexref hTexRef, CUarray hArray, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetBorderColor ( CUtexref hTexRef, float* pBorderColor ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetFilterMode ( CUtexref hTexRef, CUfilter_mode fm ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetFlags ( CUtexref hTexRef, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetFormat ( CUtexref hTexRef, CUarray_format fmt, int  NumPackedComponents ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetMaxAnisotropy ( CUtexref hTexRef, unsigned int  maxAniso ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetMipmapFilterMode ( CUtexref hTexRef, CUfilter_mode fm ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetMipmapLevelBias ( CUtexref hTexRef, float  bias ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetMipmapLevelClamp ( CUtexref hTexRef, float  minMipmapLevelClamp, float  maxMipmapLevelClamp ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexRefSetMipmappedArray ( CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuSurfRefGetArray ( CUarray* phArray, CUsurfref hSurfRef ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuSurfRefSetArray ( CUsurfref hSurfRef, CUarray hArray, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexObjectCreate ( CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexObjectDestroy ( CUtexObject texObject ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexObjectGetResourceViewDesc ( CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuTexObjectGetTextureDesc ( CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuSurfObjectCreate ( CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuSurfObjectDestroy ( CUsurfObject surfObject ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuSurfObjectGetResourceDesc ( CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxDisablePeerAccess ( CUcontext peerContext ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuCtxEnablePeerAccess ( CUcontext peerContext, unsigned int  Flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceCanAccessPeer ( int* canAccessPeer, CUdevice dev, CUdevice peerDev ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuDeviceGetP2PAttribute ( int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsMapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsResourceGetMappedMipmappedArray ( CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsResourceGetMappedPointer_v2 ( CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsResourceSetMapFlags_v2 ( CUgraphicsResource resource, unsigned int  flags ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsSubResourceGetMappedArray ( CUarray* pArray, CUgraphicsResource resource, unsigned int  arrayIndex, unsigned int  mipLevel ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsUnmapResources ( unsigned int  count, CUgraphicsResource* resources, CUstream hStream ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" CUresult cuGraphicsUnregisterResource ( CUgraphicsResource resource ) {
  assert(0);
  CUresult ret_val = CUDA_SUCCESS;
  return ret_val;
}

extern "C" void                                                                             __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  assert(0);
}

