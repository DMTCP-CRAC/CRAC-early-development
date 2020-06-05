/*
  Source: https://github.com/forresti/stackoverflow_examples/blob/master/testTexcacheObject_1D/testTexcacheObject.cu
*/
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <unistd.h>
#define N 10

// texture object is a kernel argument
__global__ void printGpu_tex(cudaTextureObject_t tex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        float x = tex1Dfetch<float>(tex, tid);
        printf("tid=%d, tex1Dfetch<float>(tex, %d) = %f \n", tid, tid, x);
    }
}

__global__ void printGpu_vanilla(float* d_buffer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        float x = d_buffer[tid];
        printf("tid=%d, d_buffer[%d] = %f \n", tid, tid, x);
    }
}

int main() {
    // declare and allocate memory
    float* d_buffer;
    cudaMalloc(&d_buffer, N*sizeof(float));

    float h_buffer[10] = {1,2,3,4,5,6,7,8,9,10};
    cudaMemcpy(d_buffer, h_buffer, sizeof(float)*N, cudaMemcpyHostToDevice);

    //CUDA 5 texture objects: https://developer.nvidia.com/content/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = N*sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    // create texture object: we only have to do this once!
    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);


    //float *d_result;
    //cudaMalloc(&d_result, N*sizeof(float));

    int grid = N/16+1;
    int block = 16;
    printGpu_tex<<<grid, block>>>(tex);
    sleep(10);
    printGpu_vanilla<<<grid, block>>>(d_buffer);


    // destroy texture object
    cudaDestroyTextureObject(tex);

    cudaFree(d_buffer);
}
