#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
/*
extern "C"
{
 void _ZL24__sti____cudaRegisterAllv(){
 }
} */

__global__ void add(int a, int b, int *c)
{
	*c = a+b;
	printf("Inside %d + %d = %d\n", a, b, *c);
}

int main(int argc, char **argv)
{
	// test
	int a = 3, b = 3, c=0;
	int *cuda_c = NULL;
	printf("before any cuda call\n");
	printf("Hey\n");
	fflush(stdout);
	cudaMallocManaged(&cuda_c, sizeof(int));
	add<<<1,1>>>(a, b, cuda_c);
	cudaError_t ret = cudaMemcpy(&c, cuda_c, sizeof(int), cudaMemcpyDeviceToHost);
	 printf("device 1: %d \n", *cuda_c);

	cudaSetDevice(1);
	int *cuda_c2 = NULL;
	cudaMallocManaged(&cuda_c2, sizeof(int));
	//cudaSetDevice(0);
	printf("device 1: %d \n", *cuda_c);
	ret = cudaMemcpy(cuda_c2, cuda_c, sizeof(int), cudaMemcpyDeviceToDevice);
	printf("error: %s \n", cudaGetErrorString(ret));	
//        printf("device 1: %d \n", *cuda_c);
//        printf("device 2: %d \n", *cuda_c2);

	cudaDeviceSynchronize();
	ret = cudaMemcpy(&c, cuda_c2, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//printf("error: %s \n", cudaGetErrorString(ret));
	printf("device 1: %d \n", *cuda_c);
	cudaDeviceSynchronize();
	printf("device 2: %d \n", *cuda_c2);
	printf("host %d \n", c);
	fflush(stdout);
	exit(EXIT_SUCCESS);
}
