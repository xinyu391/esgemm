#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "GLES3/gl3.h"

#include "EGL/egl.h"
#include "esgemm.h"

#include "glsl_src.h"

//要使用的openGL 版本，2,3
#define GLES_VERSION 3
#include <time.h>

static double now_mss(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return 1000.0 * res.tv_sec + (double)res.tv_nsec / 1e6;
}

#define POSITION_UNIFORM_NAME "pos"
#define TEXTURE_UNIFORM_NAME "tex"

#define TEXTURE_UNIFORM_NAME_0 "A"
#define TEXTURE_UNIFORM_NAME_1 "B_t"
#define TEXTURE_UNIFORM_NAME_2 "C"
#define SHARED_LENGTH_UNIFORM_NAME "K"
#define COLUMN_COUNT_UNIFORM_NAME "N"
#define PAD_UNIFORM_NAME "pad"
#define ALPHA_UNIFORM_NAME "alpha"
#define BETA_UNIFORM_NAME "beta"

#define COMPONENTS_PER_TEXEL 4

 #define ERROR(...) printf(__VA_ARGS__)
 #define DEBUG(...) printf(__VA_ARGS__)
// #define LOG(...) printf(__VA_ARGS__)
#define LOG(...)

typedef struct
{
    GLint vertexShader;
    GLuint program_;
    GLuint program_c;
    GLuint program;
    GLboolean standalone;

    GLuint frameBuffer;
    GLuint renderBuffer;

// EGL environment
    EGLDisplay eglDisplay;
    EGLSurface eglSurface;
    EGLContext eglContext;
// shader position
    GLint position;
    GLint texture;
    GLint K_gl;
    GLint alpha_gl;
    GLint beta_gl;
    GLint N_gl;
    GLint pad_gl;

} GLESContext;

GLESContext sContext;

GLboolean SetupEGL()
{
    if(sContext.eglContext!=NULL){
        return GL_TRUE;
    }
    EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY)
    {
        ERROR("Could not get EGL display\n");
        return GL_FALSE;
    }

    EGLBoolean bsuccess;
    EGLint major = 0;
    EGLint minor = 0;
    bsuccess = eglInitialize(eglDisplay, &major, &minor);

    if (!bsuccess)
    {
        ERROR("Could not initialize EGL display\n");

        return GL_FALSE;
    }
    LOG("egl version %d.%d\n", major,minor);
    EGLint attrs[] = {EGL_DEPTH_SIZE, 0,
                      EGL_STENCIL_SIZE, 0,
                      EGL_BLUE_SIZE, 8,
                      EGL_GREEN_SIZE, 8,
                      EGL_RED_SIZE, 8,
                      EGL_ALPHA_SIZE, 8,
                      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                      EGL_NONE};
    EGLint numConfig = 0;
    EGLConfig eglConfig = 0;
    bsuccess = eglChooseConfig(eglDisplay, attrs, &eglConfig, 1, &numConfig);
    if (!bsuccess)
    {
        ERROR("Could not find valid EGL config\n");
        return GL_FALSE;
    }


    EGLContext eglContext;
    EGLint ctxattrs[] = {EGL_CONTEXT_CLIENT_VERSION, GLES_VERSION, EGL_NONE};
    eglContext = eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, ctxattrs);
    if (eglContext == EGL_NO_CONTEXT)
    {
        ERROR("Could not create EGL context\n");
        return GL_FALSE;
    }

    const EGLint surfaceAttr[] = {
        EGL_WIDTH, 32,
        EGL_HEIGHT, 32,
        EGL_LARGEST_PBUFFER, EGL_TRUE,
        EGL_NONE};

    EGLSurface eglSurface = eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttr);
    if (eglSurface == EGL_NO_SURFACE)
    {
        ERROR("eglCreatePbufferSurface failed %d\n", eglGetError());

        return GL_FALSE;
    }

    bsuccess = eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
    if (!bsuccess)
    {
        ERROR("Could not activate EGL context\n");
        eglDestroyContext(eglDisplay, eglContext);
        eglDestroySurface(eglDisplay, eglSurface);
        return GL_FALSE;
    }
    sContext.eglSurface = eglSurface;
    sContext.eglContext = eglContext;
    sContext.eglDisplay = eglDisplay;

    LOG("SetupEGL success\n");
    return GL_TRUE;
}

void DestroyEGL()
{
    // program
    // Framebuffer
    glDeleteFramebuffers(1, &(sContext.frameBuffer));
    eglMakeCurrent(EGL_NO_DISPLAY, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(sContext.eglDisplay, sContext.eglContext);
    eglDestroySurface(sContext.eglDisplay, sContext.eglSurface);
    eglTerminate(sContext.eglDisplay);
    sContext.eglDisplay = 0;
    sContext.eglSurface = 0;
    sContext.eglContext = 0;
}

void dumpFloatArray(float *tmp, int h, int w)
{
    for (int i = 0; i < w * h; i++)
    {
        printf("%.2f", tmp[i]);
        if ((i + 1) % w == 0)
            printf("\n");
        else
            printf(", ");
    }
    printf("\n");
}
void dumpByteArray(unsigned char *tmp, int h, int w)
{
    for (int i = 0; i < w * h; i++)
    {
        printf("%d", tmp[i]);
        if ((i + 1) % w == 0)
            printf("\n");
        else
            printf(", ");
    }
    printf("\n");
}

void checkError(const char *str)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
        ERROR("ERROR >>>>>> glGetError %s :%d \n", str, err);
}

int getPad(int n)
{
    int rem = n % COMPONENTS_PER_TEXEL;
    int pad = rem == 0 ? 0 : COMPONENTS_PER_TEXEL - rem;
    return pad;
}

GLuint bindOutputTexture(int M, int N, GLuint texture)
{
    glViewport(0, 0,N, M);
    GLuint frameBuffer = sContext.frameBuffer;
    if (frameBuffer == 0)
    {
        glGenFramebuffers(1, &frameBuffer);
        sContext.frameBuffer = frameBuffer;
        glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    }
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, /*Level*/ 0);
    int ret = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (ret!= GL_FRAMEBUFFER_COMPLETE)
    {
        ERROR("bindOutputTexture initImageFBO failed (%d) ! MxN = %d, %d  \n", ret, M, N );    
        return -1;    
    }
    return frameBuffer;

}

GLint compileShader(const GLchar *src, GLenum type)
{
    GLuint shader;
    GLint status;
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        ERROR("Failed to create (%d)  shader.\n", type);

        GLchar *compiler_log = new GLchar[1024];
        GLsizei slen;
        glGetShaderInfoLog(shader, 1024, &slen, compiler_log);
        compiler_log[slen] = 0;
        ERROR("InfoLog:%s\n", compiler_log);

        glDeleteShader(shader);
        return 0;
    }
    return shader;
}
GLuint createProgram(GLESContext *ctx, const GLchar *src)
{

    GLint status;
    GLuint shader = compileShader(src, GL_FRAGMENT_SHADER);

    GLuint pro = glCreateProgram();

    glAttachShader(pro, ctx->vertexShader);
    glAttachShader(pro, shader);

    // link the program
    glLinkProgram(pro);
    glGetProgramiv(pro, GL_LINK_STATUS, &status);
    if (!status)
    {
        ERROR("Failed to link program.\n");
        GLchar *compiler_log = new GLchar[1024];
        GLsizei slen;
        glGetProgramInfoLog(shader, 1024, &slen, compiler_log);
        compiler_log[slen] = 0;
        ERROR("InfoLog:%s\n", compiler_log);
        glDeleteProgram(pro);
        glDeleteShader(shader);
        return 0;
    }
    //glDeleteShader(shader);
    return pro;
}

GLuint createDataTexture(int h, int w, GLfloat *texels)
{
    GLfloat PAD_TEMPLATE[] = {0.0, 0.0, 0.0, 0.0};

    int rem = w % COMPONENTS_PER_TEXEL;
    int pad = rem == 0 ? 0 : (COMPONENTS_PER_TEXEL - rem);
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    if (pad == 0 || texels == NULL)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (w + pad) / COMPONENTS_PER_TEXEL, h, 0, GL_RGBA, GL_FLOAT, texels);
    }
    else
    {
        // must pad each row

        // create empty texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (w + pad) / COMPONENTS_PER_TEXEL, h, 0, GL_RGBA, GL_FLOAT, NULL);
        int full_texel_row_len = w - rem;
        int full_row_texture_width = full_texel_row_len / COMPONENTS_PER_TEXEL;

        int row_start = 0;
        GLfloat *last_texel = PAD_TEMPLATE; //
        GLfloat *row;
        GLfloat *remainder;
        int full_texel_row_end;
        int BYTES_PER_ELEMENT = sizeof(GLfloat);
        // set texture data, one row at a time, padding each row to a multiple
        // of the texel length
        for (int i = 0; i < h; i++)
        {
            row_start = i * w;
            full_texel_row_end = row_start + full_texel_row_len;
            row = texels + row_start ;
            // if (full_texel_row_len > 0)
            // {
                glTexSubImage2D(GL_TEXTURE_2D,
                                0,                      // mip-map level
                                0,                      // x-offset
                                i,                      // y-offset
                                full_row_texture_width, // width
                                1,                      // height
                                GL_RGBA,                // format
                                GL_FLOAT,               // type
                                row                     // data
                                );
            // }

            remainder = texels + full_texel_row_end ;
            // copy remainder to last_texel
            for (int j = 0; j < 4; j++)
            {
                if (j < rem)
                    last_texel[j] = remainder[j];
               
            }
            glTexSubImage2D(GL_TEXTURE_2D,
                            0,                      // mip-map level
                            full_row_texture_width, // x-offset
                            i,                      // y-offset
                            1,                      // width
                            1,                      // height
                            GL_RGBA,                // format
                            GL_FLOAT,               // type
                            last_texel              // data
                            );
        }
    }
    // clamp to edge to support non-power of two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // don't interpolate when getting data from texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}

GLfloat *readData(int M, int N, GLfloat *out)
{   
    //glPixelStorei(GL_PACK_ALIGNMENT, 1);
    GLubyte *prod = (GLubyte *)out;
    glReadPixels(0, 0, N, M, GL_RGBA, GL_UNSIGNED_BYTE, prod);
    return out;
}

void bindVertices(GLuint prog)
{

    GLuint vertexBuffer[3];
    glGenBuffers(3, vertexBuffer);

    // bind vertices
    GLint position = sContext.position;
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer[0]);
    // define a square that covers the screen
    GLfloat vertices[] = {-1.0, -1.0, 0.0, // bottom left
                          1.0, -1.0, 0.0,  // bottom right
                          1.0, 1.0, 0.0,   // top right
                          -1.0, 1.0, 0.0}; // top left
    glEnableVertexAttribArray(position);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // bind texture cords
    GLint texture = sContext.texture;
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer[1]);
    GLfloat textureCoords[] = {0.0, 0.0,
                               1.0, 0.0,
                               1.0, 1.0,
                               0.0, 1.0};
    glBufferData(GL_ARRAY_BUFFER, sizeof(textureCoords), textureCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(texture, 2, GL_FLOAT, GL_FALSE, 0, 0);
    // glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, textureCoords);
    glEnableVertexAttribArray(texture);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /*	// index to vertices
     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,vertexBuffer[2]);
     // tesselate square into triangles
	// indeces into vertex array creating triangles, with counter-clockwise winding
    GLuint vertexIndices[] = {0, 1, 2,	// bottom right triangle
						 0, 2, 3};	// top left triangle
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(vertexIndices), vertexIndices, GL_STATIC_DRAW);                         
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);*/
}
void bindInputTexture(GLESContext *ctx, GLuint texture, GLenum textureUnit, const GLchar *name)
{
    glActiveTexture(textureUnit);
    glBindTexture(GL_TEXTURE_2D, texture);
    GLint sampler = glGetUniformLocation(ctx->program, name);
    glUniform1i(sampler, textureUnit - GL_TEXTURE0);
}

//绑定变量数据
void bindUniforms(GLESContext *ctx, GLint N, GLint K, GLint pad, GLfloat alpha, GLfloat beta)
{
    GLint K_gl =sContext.K_gl;
    GLint alpha_gl = sContext.alpha_gl;
    GLint beta_gl =sContext.beta_gl;
    GLint N_gl = sContext.N_gl;
    GLint pad_gl = sContext.pad_gl;

    //glUniform1f(beta_gl, beta);
    glUniform1i(N_gl, N);
    glUniform1i(pad_gl, pad);

    // bind length of shared dimension
    glUniform1i(K_gl, K);
    // bind alpha
    glUniform1f(alpha_gl, alpha);
}

void unbindInputTexture(GLenum textureUnit)
{
    glActiveTexture(textureUnit);
    glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint createOutputTexture(int h, int w)
{
    int pad = getPad(w);
    // create and bind texture to render to
    GLuint destTexture;
    glGenTextures(1, &destTexture);
    //glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, destTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w + pad, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // clamp to edge to support non-power of two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // don't interpolate when getting data from texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    return destTexture;
}

void calculate(int M, int N, int K, GLfloat alpha, GLuint A, GLuint B, GLfloat beta, GLuint C, GLuint out)
{
    if (C != 0)
        sContext.program = sContext.program_c;
    else
        sContext.program = sContext.program_;
    // set calculator program to current shader program
    glUseProgram(sContext.program);

    // 顶点坐标，纹理坐标
    bindVertices(sContext.program);

    GLint kPad = getPad(K);
    GLint nPad = getPad(N);

    //  N, K, alpha 赋值
    bindUniforms(&sContext, N, K + kPad, nPad, alpha, beta);

    //  bind our input textures containing matrix data
    bindInputTexture(&sContext, A, GL_TEXTURE0, TEXTURE_UNIFORM_NAME_0);
    bindInputTexture(&sContext, B, GL_TEXTURE1, TEXTURE_UNIFORM_NAME_1);

    if (C != 0)
    {
        bindInputTexture(&sContext, C, GL_TEXTURE2, TEXTURE_UNIFORM_NAME_2);
    }

    // 创建目标texture
    if (sContext.standalone)
    {
        bindOutputTexture(M,N+nPad, out);
    }
    else
    {
         bindOutputTexture(M,(N+nPad)/4, out);
    }

    LOG("draw Elements  view port(%d, %d)\n", (N + nPad), M);
    //  glDrawArrays(GL_TRIANGLES, 0, 3);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    // glDrawElements(GL_TRIANGLES, /*num items*/4, GL_UNSIGNED_SHORT, 0);
    checkError("glDrawElements");

    unbindInputTexture(GL_TEXTURE0);
    unbindInputTexture(GL_TEXTURE1);
    unbindInputTexture(GL_TEXTURE2);
}

int loadProgram(GLboolean standalone)
{
    if(sContext.vertexShader!=0){
        return 0;
    }
    const GLchar *src = sl_pass_through; // ("../glsl/pass_through.glsl");
    GLuint shader = compileShader(src, GL_VERTEX_SHADER);
    sContext.vertexShader = shader;

    if (standalone == GL_TRUE)
    {
        sContext.standalone = GL_TRUE;
        sContext.program_ = createProgram(&sContext, sl_standalone);    //"../glsl/sgemm/standalone.glsl"
        sContext.program_c = createProgram(&sContext, sl_standalone_c); //"../glsl/sgemm/standalone_c.glsl"
    }
    else
    {
        sContext.standalone = GL_FALSE;
        sContext.program_ = createProgram(&sContext, sl_pipeline);    //"../glsl/sgemm/pipeline.glsl"
        sContext.program_c = createProgram(&sContext, sl_pipeline_c); //"../glsl/sgemm/pipeline_c.glsl"
    }

    sContext.position =  glGetAttribLocation(sContext.program_, POSITION_UNIFORM_NAME);
    sContext.texture  = glGetAttribLocation(sContext.program_, TEXTURE_UNIFORM_NAME);

    sContext.K_gl = glGetUniformLocation(sContext.program_, SHARED_LENGTH_UNIFORM_NAME);
    sContext.alpha_gl = glGetUniformLocation(sContext.program_, ALPHA_UNIFORM_NAME);
    sContext.beta_gl = glGetUniformLocation(sContext.program_, BETA_UNIFORM_NAME);
    sContext.N_gl = glGetUniformLocation(sContext.program_, COLUMN_COUNT_UNIFORM_NAME);
    sContext.pad_gl = glGetUniformLocation(sContext.program_, PAD_UNIFORM_NAME);
    LOG("loadProgram success\n");
    return 0;
}
GLfloat *transpose(int h, int w, const float *B)
{
    GLfloat *a = (GLfloat *)malloc(h * w * sizeof(GLfloat));
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            a[c * h + r] = B[r * w + c];
    return a;
}
#define  DMSG(str) do{\
    double t = now_mss();\
      DEBUG("DEBUG %.3f \t%s\n",t-time, str);\
    time = t;\
}while(0);

int sgemm(int TA, int TB, int M, int N, int K, const float *A, const float *B, float alpha, float beta, float *C)
{
   double time = now_mss();
   DMSG("sgemm");
    if(SetupEGL()==GL_FALSE){
        return -1;
    }
    DMSG("init");
    loadProgram(GL_TRUE); // createProgram
    DMSG("loadProgram");
    if (A == NULL || B == NULL)
        return -1;
    GLfloat *rA = (GLfloat *)A;
    if(TA==1){// transpose A
        rA = transpose(M, K, A);        
    }
    DMSG("transpose A");
    GLfloat *rB = (GLfloat *)B;
    if(TB==0){//因为B本来就要转置，再转置（TB=1），据相当于不转置
        rB = transpose(K, N, B);
    }
    
DMSG("transpose B");
    GLuint tA = createDataTexture(M, K, rA);
    GLuint tB = createDataTexture(N, K, rB);
DMSG("input text");
    GLuint tC = 0;
    if (C != NULL)
    {// not support now( 只支持 1×N的C)
        tC = 0;//createDataTexture(1, N, C);
    }
    GLuint tOut =  createOutputTexture(M,N);
DMSG("output text");
    calculate(M, N, K, alpha, tA, tB, beta, tC, tOut);
    glFinish();
    DMSG("calculate");
    readData(M, N, C);
      DMSG("readData");
    // dumpFloatArray(C, M, N);
 
    glDeleteTextures(1, &tA);
    glDeleteTextures(1, &tB);
    if (tC != 0)
        glDeleteTextures(1, &tC);
    glDeleteTextures(1, &tOut);     
      DMSG("glDeleteTextures");
    if(TA==1)
        free(rA);
    if(TB==0)
        free(rB);
      DMSG(" free");
    return 0;
}

void cblas_sgemm(const enum CBLAS_ORDER Order,
	const enum CBLAS_TRANSPOSE TA, const enum CBLAS_TRANSPOSE TB,
	const int M, const int N, const int K,
	const float  alpha, const float *A, const int lda,
	const float *B, const int ldb, const float  beta,
	float *C, const int ldc){
    //    printf("cblas_sgemm M=%d, N=%d, K=%d a=%.2f, b=%.2f, lda=%d, ldb=%d,ldc=%d order=%d, Ta=%d TB=%d\n", M, N, K,alpha, beta,lda, ldb,ldc, Order, TA, TB);
        if(lda!=K||ldb!=N||ldc!=N){
           //return ;
        }
        if(TA==CblasConjTrans||TB==CblasConjTrans){
           ERROR("current only support CblasNoTrans and CblasTrans\n");
           return ;
        }
         if(Order!=CblasRowMajor){
           ERROR("current only support CblasRowMajor\n");
          //  return ;
          // 数据转置下？
        }
    
        int ta = TA==CblasNoTrans?0:1;
        int tb = TB==CblasNoTrans?0:1;
        sgemm(ta,tb, M,  N,  K, A, B,  alpha,  beta, C);
    }