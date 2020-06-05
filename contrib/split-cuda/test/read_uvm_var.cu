#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// reading a UVM global variable
__managed__ int ten = 10;

__global__ void add(int a, int b, int *c)
{
	*c = a+b+ten;
}

int main(int argc, char **argv)
{
	// test
	int a = 2, b = 3, c;
	int *cuda_c = NULL;
	cudaMalloc(&cuda_c, sizeof(int));
    sleep(10);
	add<<<1,1>>>(a, b, cuda_c);
	cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(cuda_c);

	printf("%d + %d = %d\n", a, b, c);

	exit(EXIT_SUCCESS);
}
