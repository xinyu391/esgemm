#include <stdio.h>
#include <stdlib.h>
#include "esgemm.h"
#include <time.h>

static double now_ms(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return 1000.0 * res.tv_sec + (double)res.tv_nsec / 1e6;
}
 void testblas(int m, int n, int k);

 void dumpFloatAry(float *tmp, int h, int w)
{
    for (int i = 0; i < w * h; i++)
    {
        printf("%.0f", tmp[i]);
        if ((i + 1) % w == 0)
            printf("\n");
        else
            printf(" ");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
/*
    int M = 32;
    int N = 32;
    int K = 32;
    // M*K
    float *A = (float *)malloc(sizeof(float) * M * K);

    for (int i = 0; i < M * K; i++)
        A[i] = i;
    // K*N
    float *B = (float *)malloc(sizeof(float) * K * N);
    for (int i = 0; i < N * K; i++)
        B[i] = i % 2;

    float *C = (float *)malloc(sizeof(float) * M * N);
    
    float *out = NULL;
    float alpha = 1.0, beta = 0.0;
    int ret;
    dumpFloatArray(A, M, N);
    dumpFloatArray(B, M, N);
    for (int i = 0; i < 1000; i++){
        ret = sgemm(0,0,M, N, K, A, B, alpha, beta, C);
        printf(" %d -> %d\n", i,ret);
    }
    // dumpFloatArray(C, M, N);
    printf("finish %d\n", ret);
    float c1 = C[1];
    float c5 = C[5];
      unsigned char* tmp = (unsigned char*) &c1;
        printf("C1: %.0f  %d %d %d %d \n",c1,  tmp[0],tmp[1],tmp[2],tmp[3]);
        tmp = (unsigned char*) &c5;
        printf("C5: %.0f  %d %d %d %d \n",c5,  tmp[0],tmp[1],tmp[2],tmp[3]);
    free(A);
    free(B);
    free(C);
    */
     testblas(1,1,1);
     printf("\n");
     double start =now_ms();
     int count = 8344;
   for(int i=256;i<count;i+=1){
        double time =now_ms();
	    // testblas(32,i,6);
        testblas(i,i,i);
        time = now_ms()-time;
        	printf("MNK %d  %.2fms\n", i, time);
            break;
    }
    start =now_ms()-start;
    printf("average   %.2fms\n", start/count);
    // DestroyEGL();
 
    return 0;
}

void testblas(int m, int n, int k){
	 int M = m;
    int N = n;
    int K = k;
    // M*K
    float *A = (float *)malloc(sizeof(float) * M * K);

    for (int i = 0; i < M * K; i++)
        A[i] = i;
    // K*N
    float *B = (float *)malloc(sizeof(float) * K * N);
    for (int i = 0; i < N * K; i++)
        B[i] = i % 2;

    float *C = (float *)malloc(sizeof(float) * M * N);
    
    // dumpFloatAry(A,M,K);
    // dumpFloatAry(B,K,N);

    float *out = NULL;
    float alpha = 1.0, beta = 0.0;

    int lda = K;
    int ldb = N;
    int ldc = N;

    CBLAS_TRANSPOSE TA = CblasNoTrans;
    CBLAS_TRANSPOSE TB = CblasNoTrans;
    cblas_sgemm(CblasRowMajor,
	TA, TB,
	M,  N,  K,
	alpha, A,lda,
	B,  ldb,  beta,
	C, ldc);

    // dumpFloatAry(C,M,N);
    free(A);
    free(B);
    free(C);
}