/*
Matrix Multiplication in CUDA
Modified from :
https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
https://www.geeksforgeeks.org/strassens-matrix-multiplication/
*/
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char const *argv[]){
	int N, BLOCK_SIZE;
	printf("please type matrix dimension in N : ");
    scanf("%d",  &N);
	printf("please input block size: ");
    scanf("%d",  &BLOCK_SIZE);
	int nBytes = N*N*sizeof(float);
	float *h_a, *h_b;
	// Memory Allocation in Host
	h_a = (float *)malloc(nBytes);
	h_b = (float *)malloc(nBytes);

	//  Initialization of a and b Matrix
	for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
		  h_a[i * N + j] = 1;
		  h_b[i * N + j] = 2;
      }
	}
	printf("Matrix A \n");
	for (int i = 0; i < N; ++i)
    {
		printf("[ ");
        for (int j = 0; j < N; ++j)
        {            
		printf("%f ",h_a[i*N+ j]);	
        }
		printf("] \n");
    }
    printf("\n\n\n");
	printf("Matrix B \n");
	for (int i = 0; i < N; ++i)
    {
		printf("[ ");
        for (int j = 0; j < N; ++j)
        {            
		printf("%f ",h_b[i*N+ j]);	
        }
		printf("] \n");
    }
	printf("\n");

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    return 0;
}
