#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
  int myrank;
  int *val_device, *val_host, *new_val_host;

  printf("Before MPI Initialization\n");
  fflush(stdout);
  // sleep(15);
  int ret_val = MPI_Init(&argc, &argv);
  if (ret_val < 0) {
    perror("MPI Init failed");
    exit(-1);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  val_host = (int*)malloc(sizeof(int));
  new_val_host = (int*)malloc(sizeof(int));
  cudaError_t ret = cudaMalloc((void **)&val_device, sizeof(int));
  if (ret < 0) {
    perror("cudaMalloc failed");
    exit(-1);
  }

  *val_host = -1;
  printf("%s %d %s %d\n", "I am rank", myrank, "and my initial value is:",
          *val_host);
  *new_val_host = myrank;
  ret = cudaMemcpy(val_device, new_val_host, sizeof(int), cudaMemcpyHostToDevice);
  if (ret < 0) {
    perror("cudaMemcpy host to device failed");
    exit(-1);
  }
  printf("%s %d %s %d\n", "I am rank", myrank, "and copying to device",
         *new_val_host);
  // fetch it back
  ret = cudaMemcpy(val_host, val_device, sizeof(int), cudaMemcpyDeviceToHost);
  if (ret < 0) {
    perror("cudaMemcpy device to host failed");
    exit(-1);
  }

  printf("%s %d %s %d\n", "I am rank", myrank, "and received value:", *val_host);

  cudaFree(val_device);
  free(val_host);
  free(new_val_host);

  MPI_Finalize();
  return 0;
}
