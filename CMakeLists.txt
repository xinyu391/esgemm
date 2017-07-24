project(egl)

set(SRC test.cpp sgemm.cpp)


OPTION(B_ANDROID "build for android platform" OFF)

if(B_ANDROID)
message("build for Android")
# build for Android 
# Android 5.0 以上需要在此处设置 PIE
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIE -pie")
# 配置使用 NDK Standalone Toolchain 编译
set(NDK_STANDALONE_TOOLCHAIN /home/jesse/work/blas/arm64-android-toolchain)
set(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-gcc)
set(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-g++)
set(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
# 使用 NDK 提供的头文件
add_definitions("--sysroot=${NDK_STANDALONE_TOOLCHAIN}/sysroot")
endif()

if(B_ANDROID)

else()
#include_directories(GLES_SDK_v31/include)
#link_directories(GLES_SDK_v31/x86-64/)

add_executable(eglo test.cpp)
target_link_libraries(eglo  openblas)

endif()

# 编译静态库
add_library(esgemm sgemm.cpp)
#target_link_libraries(esgemm  libEGL.so libGLESv2.so)
#target_link_libraries(esgemm  dl)

#编译测试程序
add_executable(egl test.cpp)
target_link_libraries(egl  esgemm)
target_link_libraries(egl  libEGL.so libGLESv2.so)
target_link_libraries(egl  dl)





