#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
# include <cuda_runtime .h>
# include " cublas_v2 .h"
# define IDX2C (i,j,ld) (((j)*( ld ))+( i ))

int main ( void ){
if (argc < 2) {
		printf("Required args: N(dimension), k\n");
		exit(-1);
	}
int N = atoi(argv[2])
cudaError_t cudaStat ; // cudaMalloc status
cublasStatus_t stat ; // CUBLAS functions status
cublasHandle_t handle ; // CUBLAS context
// int i,j; // i- row index ,j- column index
float * h_a; // mxk matrix a on the host
float * h_b; // kxn matrix b on the host
float * h_c; // mxn matrix c on the host
h_a=( float *) malloc (N*N* sizeof ( float )); // host memory for a
h_b=( float *) malloc (N*N* sizeof ( float )); // host memory for b
h_c=( float *) malloc (N*N* sizeof ( float )); 
// define an mxk matrix a column by column

for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        h_a[IDX2C(i,j,N)] = 1.0;
        h_b[IDX2C(i,j,N)] = 2.0;
      }
    }

float * d_a; // d_a - a on the device
float * d_b; // d_b - b on the device
float * d_c; // d_c - c on the device
cudaStat = cudaMalloc (( void **)& d_a ,N*N* sizeof (float )); // device
// memory alloc for a
cudaStat = cudaMalloc (( void **)& d_b ,N*N* sizeof (float )); // device
// memory alloc for b
cudaStat = cudaMalloc (( void **)& d_c ,N*N* sizeof (float )); // device
// memory alloc for c
stat = cublasCreate (& handle ); // initialize CUBLAS context
// copy matrices from the host to the device
stat = cublasSetMatrix (m,k, sizeof (*a),a,m,d_a ,m); //a -> d_a
stat = cublasSetMatrix (k,n, sizeof (*b),b,k,d_b ,k); //b -> d_b
stat = cublasSetMatrix (m,n, sizeof (*c),c,m,d_c ,m); //c -> d_c
float al =1.0 f; // al =1
float bet =0.0 f; // bet =1
// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
// al ,bet -scalars
stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,,N,N,N,&al,d_a,N,d_b,N,&bet,d_c,N);
stat = cublasGetMatrix (m,n, sizeof (*c),d_c ,N,h_c,N); // cp d_c ->c
printf ("Hasil Matrix Calculation :\n");
for(i=0;i<m;i ++){
for(j=0;j<n;j ++){
printf (" %7.0 f",c[ IDX2C (i,j,m )]); // print c after Sgemm
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