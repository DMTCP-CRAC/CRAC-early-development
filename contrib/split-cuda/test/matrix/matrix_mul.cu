#include<stdio.h>
#include<unistd.h>
#include<sys/time.h>
#include<stdlib.h>
#include<iostream>

#define BLOCK_SIZE 10

using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------

typedef struct {
	int dimension1;
	int dimension2;	
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory  
float *A, *B, *C, *C_CPU;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error); 
void check_error(cudaError e);

//----------------------------------- CUDA function definitions -----------------------------------------

__global__ void computeGpuMMM(float *A_GPU, float *B_GPU, float *C_GPU, int width);
__global__ void computeGpuMMM_primitive(float *A_GPU, float *B_GPU, float *C_GPU, int width, int B_D2);

//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
void *cuda_ptr1 = NULL;
  void *cuda_ptr2 = NULL;	
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
	int thread_dim_per_block = (argc > 5) ? atoi(argv[5]) : 2;
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;

	printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
	printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
	printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);

	allocateAndInitializeAB();

	//matrix matrix multiplication in the GPU
	dim3 dimGrid(A_MD.dimension1/thread_dim_per_block, B_MD.dimension2/thread_dim_per_block);
	dim3 dimBlock(thread_dim_per_block, thread_dim_per_block);
        copyMatricesToGPU();
	clock_t start = clock();
	computeGpuMMM_primitive<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, A_MD.dimension2, B_MD.dimension2);
	//computeGpuMMM<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, A_MD.dimension2);
	cudaThreadSynchronize();
        clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        printf("Computation time in the GPU: %f seconds\n", elapsed);
        printf("before copying results\n");
	
	copyResultFromGPU();

        printf("after copying results\n");
        // matrix matrix multiplication in the CPU

	sleep(8);
	
	printf("I restarted successefully\n");
	printf("I restarted successefully\n");
	printf("I restarted successefully\n");
       // start = clock();
        computeCpuMMM();
	printf("I I am working just fine\n");
       // end = clock();
       // elapsed = (end - start) / (double) CLOCKS_PER_SEC;
       // printf("Computation time in the CPU: %f seconds\n", elapsed);

	printf("Now I will call cuda malloc and checkpoint here\n");
	cudaError_t rc = cudaMalloc(&cuda_ptr1, 436*sizeof(char));
	printf("cudaMalloc returned: %d, cuda_ptr1: %p\n", (int)rc, cuda_ptr1);
	rc = cudaMalloc(&cuda_ptr2, 43*sizeof(char));
	printf("cudaMalloc returned: %d, cuda_ptr1: %p\n", (int)rc, cuda_ptr1);
        printf("cudaMalloc returned: %d, cuda_ptr2: %p\n", (int)rc, cuda_ptr2);
	cudaFree(cuda_ptr1);
        cudaFree(cuda_ptr2);
	//compareHostAndGpuOutput();	
	return 0;
}

// allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	A = (float*) malloc(sizeofA);
	
	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++) {
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A[index] = (rand() % 1000) * 0.001; 
		}
	}
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.dimension1; i++) {
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B[index] = (rand() % 1000) * 0.001; 
		}
	}
}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));
	
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_CPU = (float*) malloc(sizeofC);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {
	
	// allocate the result matrix for the CPU computation
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C = (float*) malloc(sizeofC);
	
	// compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.dimension1; i++) {
		int a_i = i * A_MD.dimension2;
		int c_i = i * C_MD.dimension2;
		for (int j = 0; j < B_MD.dimension2; j++) {
			int c_index = c_i + j;
			C[c_index] = 0;
			for (int k = 0; k < B_MD.dimension1; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.dimension2 + j;
				C[c_index] += A[a_index] * B[b_index];
			}
		}
	}
}

__global__ void computeGpuMMM(float *A_GPU, float *B_GPU, float *C_GPU, int width){
	
	// getting position index of the thread in device
	int t_x = threadIdx.x, t_y = threadIdx.y;
	int b_x = blockIdx.x,  b_y = blockIdx.y;
	int row = b_y*blockDim.y + t_y;
	int col = b_x*blockDim.x + t_x;
	
	// allocate shared memory in block for threads
	__shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

	float result = 0;

  	//In the unit of Block in device, computing the Blocking that each unit is in charge of by
	//dot product-ing each Blocking in A & B
  	for(int b = 0; b < width/BLOCK_SIZE; b++)
  	{
    		// load Blocking into shared memory, cooperated by all the threads in the Block in device
    		s_a[t_y][t_x] = A_GPU[row*width + (b*BLOCK_SIZE + t_x)];
    		s_b[t_y][t_x] = B_GPU[(b*BLOCK_SIZE + t_y)*width + col];
    		__syncthreads();
    		
		// dot product on current loaded Blocking
    		for(int i = 0; i < BLOCK_SIZE; i++)
      			result += s_a[t_y][i] * s_b[i][t_x];
    		__syncthreads();
  	}
	
	C_GPU[row*width+col] = result;
}

__global__ void computeGpuMMM_primitive(float *A_GPU, float *B_GPU, float *C_GPU, int width, int B_D2){

        // getting position index of the thread in device
        int row = blockIdx.y*blockDim.y + threadIdx.y;
        int col = blockIdx.x*blockDim.x + threadIdx.x;

        float result = 0;

        // dot product
        for(int i = 0; i < width; i++)
                result += A_GPU[row*width+i] * B_GPU[i*B_D2+col];

        C_GPU[row*B_D2+col] = result;
}

// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
// from the CPU
void compareHostAndGpuOutput() {
	int totalElements = C_MD.dimension1 * C_MD.dimension2;
	int missmatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C[i] - C_CPU[i]) > 0.01) {
			missmatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
		else{
			printf("match at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
	}
	if (missmatchCount > 0) {
		printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
	} else {
		printf("Computation is correct: CPU and GPU outputs match\n");
	}
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

