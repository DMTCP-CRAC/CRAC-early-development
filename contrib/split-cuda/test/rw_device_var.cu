#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// reading and writing a device variable
__device__ int incr = 10;

__global__ void add(int a, int b, int *c)
{
	*c = *c+a+b+incr++;
}

int main(int argc, char **argv)
{
	// test
	int a = 2, b = 3, c;
	int t = 0;
	int *cuda_c = NULL;

	cudaMalloc(&cuda_c, sizeof(int));
	cudaError_t ret = cudaMemcpy(cuda_c, &t, sizeof(int), cudaMemcpyHostToDevice);

	sleep(10);
	add<<<1,1>>>(a, b, cuda_c);
	add<<<1,1>>>(a, b, cuda_c);
	cudaDeviceSynchronize();
	cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cuda_c);

	printf("%d + %d = %d\n", a, b, c);

	exit(EXIT_SUCCESS);
}
