#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void add(int *a, int *b, int *c)
{
	*c = *a + *b;
}

int main(int argc, char **argv)
{
	// test
	int *a = NULL, *b = NULL, *c = NULL;

	cudaMallocManaged(&a, sizeof(int));
	cudaMallocManaged(&b, sizeof(int));
    sleep(10); // Sleep to allow for checkpointing
	cudaMallocManaged(&c, sizeof(int));
	*a = 2;
	*b = 3;
	*c = 0;
	add<<<1,1>>>(a, b, c);
	cudaDeviceSynchronize();
	printf("%d + %d = %d\n", *a, *b, *c);

	exit(EXIT_SUCCESS);
}
