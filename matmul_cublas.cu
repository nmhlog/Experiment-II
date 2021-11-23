#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main ( void ){
if (argc < 2) {
		printf("Required args: N(dimension), k\n");
		exit(-1);
	}
int N = atoi(argv[2]);
cudaError_t cudaStat ; // cudaMalloc status
cublasStatus_t stat ; // CUBLAS functions status
cublasHandle_t handle ; // CUBLAS context

float *h_a, *h_b, *h_c,*h_cc;
        // Memory Allocation in Host
h_a = (float *)malloc(nBytes);
h_b = (float *)malloc(nBytes);
h_c = (float *)malloc(nBytes);
h_cc = (float *)malloc(nBytes);
// define an mxk matrix a column by column

for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        h_a[IDX2C(i,j,N)] = 1.0;
        h_b[IDX2C(i,j,N)] = 2.0;
        h_c[IDX2C(i,j,N)] = 0.0;
      }
    }

float *d_a, *d_b, *d_c;
cudaMalloc((void **) &d_a, sizeof(float)*N*N);
cudaMalloc((void **) &d_b, sizeof(float)*N*N);
cudaMalloc((void **) &d_c, sizeof(float)*N*N);

stat = cublasCreate (& handle ); // initialize CUBLAS context
// copy matrices from the host to the device
stat = cublasSetMatrix (N,N, sizeof (*h_a),h_a,N,d_a ,N); //a -> d_a
stat = cublasSetMatrix (N,N, sizeof (*h_b),h_b,N,d_b ,N); //b -> d_b
stat = cublasSetMatrix (N,N, sizeof (*h_c),h_c,N,d_c ,N); //c -> d_c
float al =1.0f; // al =1
float bet =1.0f; // bet =1
// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
// al ,bet -scalars
stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,,N,N,N,&al,d_a,N,d_b,N,&bet,d_c,N);
stat = cublasGetMatrix (N,N, sizeof (*d_c),d_c ,N,h_c,N); // cp d_c ->c
printf ("Hasil Matrix Calculation :\n");
for(i=0;i<N;i ++){
for(j=0;j<N;j ++){
printf (" %7.0 f",c[ IDX2C (i,j,N )]); // print c after Sgemm
}
printf ("\n");
}
cudaFree (d_a ); // free device memory
cudaFree (d_b ); // free device memory
cudaFree (d_c ); // free device memory
cublasDestroy ( handle ); // destroy CUBLAS context
free (a); // free host memory
free (b); // free host memory
free (c); // free host memory
return 0 ;
}