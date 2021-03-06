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

__global__ void gpuMatrixMul(float *d_a , float *d_b, float *d_c,int N) {
        // Calculate the row index of the P element and M
        int Row = blockIdx.y*blockDim.y+threadIdx.y;
        // Calculate the column index of P and N
        int Col = blockIdx.x*blockDim.x+threadIdx.x;
        if ((Row < N) && (Col < N)) {
                float Pvalue = 0;
                // each thread computes one element of the block sub-matrix
                for (int k = 0; k < N; ++k) {
                        Pvalue += d_a[Row*N+k]*d_b[k*N+Col];
                        }
                d_c[Row*N+Col] = Pvalue;
                }
}

void cpuMatrixMul(float *h_a, float *h_b, float *h_c, int N) {
    for (int i = 0; i < N; ++i) // Row
    {
        for (int j = 0; j < N; ++j) //Col
        {
            int tmp = 0.0;
            for (int h = 0; h < N; ++h)  // Row
            {
                tmp += h_a[i * N + h] * h_b[h * N + j];
            }
            h_c[i * N + j] = tmp;
        }
    }
}

void verification(float *h_c,float *h_cc,int N,float gpu_elapsed_time_ms,float cpu_elapsed_time_ms){
        int all_ok = 1;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {

            if(h_cc[i*N+ j] != h_c[i*N + j])
            {
                all_ok = 0;
            }
        }
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }
}

void print_matrix(float *h_matrix,int N){
        for (int i = 0; i < N; ++i)
    {
                printf("[ ");
        for (int j = 0; j < N; ++j)
        {
                printf("%f ",h_matrix[i*N+ j]);
        }
                printf("] \n");
    }


}
int main(int argc, char const *argv[]){
    if (argc < 3) {
		printf("Required args: N(dimension),BlockSize(), k\n");
		exit(-1);
	    }

    int N = atoi(argv[1]);
    int BLOCK_SIZE = atoi(argv[2]);  
    int nBytes = N*N*sizeof(float);
    float *h_a, *h_b, *h_c,*h_cc;
        // Memory Allocation in Host
    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_c = (float *)malloc(nBytes);
    h_cc = (float *)malloc(nBytes);
        //  Initialization of a and b Matrix
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        h_a[i * N + j] = 1.0;
        h_b[i * N + j] = 2.0;
      }
        }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*N*N);
    cudaMalloc((void **) &d_b, sizeof(float)*N*N);
    cudaMalloc((void **) &d_c, sizeof(float)*N*N);



    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpuMatrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c,N);
    cudaMemcpy(h_c, d_c, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", N,N,N,N, gpu_elapsed_time_ms);

    // start the CPU version
    cudaEventRecord(start, 0);

    cpuMatrixMul(h_a, h_b, h_cc,N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", N,N,N,N, cpu_elapsed_time_ms);
    
    
    verification(h_c,h_cc,N,gpu_elapsed_time_ms,cpu_elapsed_time_ms);
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
