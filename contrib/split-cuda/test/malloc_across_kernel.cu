#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

// using device-side mallocs that persist across kernel invocations
// based off of http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#allocation-persisting-kernel-launches
#define NUM_BLOCKS 1
__device__ int* tenptr[NUM_BLOCKS];

__global__ void alloc_and_set_ten()
{
    // Only the first thread in the block does the allocation
    // since we want only one allocation per block.
    if (threadIdx.x == 0) {
        tenptr[blockIdx.x] = (int*)malloc(sizeof(int));
        printf("tenptr[blockIdx.x] = %p\n", tenptr[blockIdx.x]);
        printf("*tenptr[blockIdx.x] = %d\n", *tenptr[blockIdx.x]);
        *tenptr[blockIdx.x] = 10;  // set the value
    }
    __syncthreads();
}

__global__ void add(int a, int b, int *c)
{
	*c = a+b+*tenptr[blockIdx.x];
}

__global__ void free_ten()
{
    // Free from the leader thread in each thread block
    if (threadIdx.x == 0)
        free(tenptr[blockIdx.x]);
}

int main(int argc, char **argv)
{
	// test
    int a = 2, b = 3, c;
    uint64_t d;
	int *cuda_c = NULL;

	cudaMalloc(&cuda_c, sizeof(int));
    alloc_and_set_ten<<<NUM_BLOCKS,1>>>();
    // cudaDeviceSynchronize();
    sleep(10);
    void * ptr = (void *)0x7fffe4bd4697;
    cudaMemcpy(&d, ptr, 8, cudaMemcpyDeviceToHost);
	printf("%zx\n", d);

    add<<<NUM_BLOCKS,1>>>(a, b, cuda_c);
    free_ten<<<NUM_BLOCKS,1>>>();
	cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cuda_c);
	
	printf("%d + %d + 10 = %d\n", a, b, c);

	exit(EXIT_SUCCESS);
}
