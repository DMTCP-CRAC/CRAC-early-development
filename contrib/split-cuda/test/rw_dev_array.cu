#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// reading and writing a device variable (with array)
__device__ int incr_idx = 0;
__device__ int incr_arr [] = {10, 11};

__global__ void add(int a, int b, int *c)
{
	*c = a+b+incr_arr[incr_idx++];
	printf("c = %d at %p\n", *c, c);
}

int main(int argc, char **argv)
{
	// test
	int a = 2, b = 3, c;
	int *cuda_c = NULL;

	cudaMalloc(&cuda_c, sizeof(int));
	printf("cuda_c = %p\n", cuda_c);
	add<<<1,1>>>(a, b, cuda_c);
	add<<<1,1>>>(a, b, cuda_c);
	cudaDeviceSynchronize();
    sleep(10);
	cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cuda_c);

	printf("%d + %d = %d\n", a, b, c);

	exit(EXIT_SUCCESS);
}
