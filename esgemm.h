// sgemm implemnts by penGL ES .
// need library libEGL.so libGLESv2.so and dl
// only support 


extern "C"
{

    // from cblas.h
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
//end from cblas.h


int sgemm(int TA, int TB, int M, int N, int K, const float*A, const float*B,  float alpha,  float beta, float*C);

// cblas api interface
void cblas_sgemm(const enum CBLAS_ORDER Order,
	const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
	const int M, const int N, const int K,
	const float  alpha, const float *A, const int lda,
	const float *B, const int ldb, const float  beta,
	float *C, const int ldc);
 
void dumpByteArray(unsigned char *tmp, int h, int w);
void dumpFloatArray(float *tmp, int h, int w);

void DestroyEGL();
}