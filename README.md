# gles-sgemm
移植于[weblas](https://github.com/waylonflinn/weblas)

项目需要OpenGL ES3.0 ,或者支持OES_texture_float(浮点纹理)扩展的ES2.0

编译时，需要链接如下库：libEGL.so libGLESv2.so  libdl.so

+ GLES_SDK_v31/ 为PC上的OpenGL ES模拟驱动库

+ libs/ 为编译的arm64,和x86-64-linux的sgemm静态库

在TextLineEngine程序测试(连续识别40张卡号，日期，持卡人的文本行图片)中，PC上比openBlas(8线程)稍差一点。
但在Android(S6)上，比openBlas(8线程)慢15倍,测试结果对比如下：

pc(i7-6700 CPU @ 3.40GHz × 8 , HD Graphics 530):
> openBlas	52.91ms

> esgemm	67.7ms

> CL sgemm	68.88ms


S6(Exynos7420 2.1GHzx4,1.5GHzx4, Mali T760)：
> openBlas	62.96ms

> esgemm	1044.42ms

> CL sgemm	1054.55ms

* 上面CL sgemm是移植的[openCL版本](https://github.com/clMathLibraries/clBLAS)的sgemm函数,测试性能和OpenGLES版本基本一致。