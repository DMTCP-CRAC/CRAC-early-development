#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void add_2_1(int a, int b, int *c)
{
  *c = a + b;
}

int main(int argc, char **argv)
{
  // test
  int a, b, *c = NULL;

  // Test for read faults
  cudaMallocManaged(&c, sizeof(int));
  a = 2;
  b = 3;
  sleep(10); // Allow time to checkpoint
  add_2_1<<<1,1>>>(a, b, c);
  cudaDeviceSynchronize();

  printf("%d + %d = %d\n", a, b, *c);

  exit(EXIT_SUCCESS);
}
