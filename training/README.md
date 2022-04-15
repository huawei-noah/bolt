# Raul - C++ client for federated learning  

[Raul API Documentation (doxygen)](http://aienabling.rnd-gitlab-msc.huawei.com/raul/)

## Build

### Reqirements

Compilers:

- Clang 11.0.0+
- GCC 9.2.1+
- Visual Studio 16 (2019)+
- Android NDK r21+

cmake 3.10+

### Steps

```sh
mkdir build  
cd build  
```

### Win32 (MinGW)

```sh
cmake -G"MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release ../  
```

### Linux

```sh
cmake -DCMAKE_BUILD_TYPE=Release ../  
```

### Android (cross compile from windows using NDK and MinGW):  

Download [NDK](https://developer.android.com/ndk/downloads) (unpack into **ndk_path**)  

#### Shared STL

```sh
cmake -G"MinGW Makefiles" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_TOOLCHAIN_FILE="**ndk_path**/build/cmake/android.toolchain.cmake" -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_STL=c++_shared ../  
```

#### Static STL

```sh
cmake -G"MinGW Makefiles" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_TOOLCHAIN_FILE="**ndk_path**/build/cmake/android.toolchain.cmake" -DANDROID_ABI=armeabi-v7a -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_STL=c++_static ../  
```

Additional cmake build options available:  
-DRAUL_USE_LIB_OPENMP=true  
-DRAUL_USE_LIB_BLAS=None|Atlas|Open|MKL  
Choose one option of BLAS library (None - naive implementation, Open - openBLAS)  

## 3rd Party

### Sources

- [OpenBlas](https://github.com/xianyi/OpenBLAS)  
- [GTest](https://github.com/google/googletest)  
- [Yato](https://bitbucket.org/alexey_gruzdev/yato/)  
- [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)  
- [glm](https://glm.g-truc.net)  

### OpenBLAS

OpenBLAS prebuilt binary for Android already included into repository (/src/thirdParty/openblas/lib/android/libopenblas.so - no openmp)  
OpenBLAS prebuilt for win32:  
https://sourceforge.net/projects/openblas/files/develop/20150903/  
libgfortran-3.dll, libgcc_s_sjlj-1.dll (mingw32_dll):  
https://sourceforge.net/projects/openblas/files/v0.2.12/  
https://sourceforge.net/projects/openblas/files/v0.2.14/  

## Datasets

To pass functional tests download datasets (MNIST, CIFAR-10, PennTreebank).  
Set cmake parameter `RAUL_USE_DATASET_LOADER=true` to download datasets automatically.  
See [testAssets/README.md](testAssets/README.md) for details.

## Docs

The projects supports Doxygen. Use the following command to generate
documentation.

```sh
cmake . -DRAUL_USE_DOCS=ON
make docs
```

Make sure you have installed `doxygen` and `dot`. You can also instgall `mscgen` and `dia` optionally.

Also, there is a class diagram of the source code which is presented in docs/Raul.graphml (use [yED](https://www.yworks.com/products/yed) to open).

## Format

We use [Mozilla coding style](https://firefox-source-docs.mozilla.org/code-quality/coding-style/coding_style_cpp.html). The sources can be automatically foramated with the following command.

```sh
cmake . -DRAUL_USE_CLANG_FORMAT=ON
make format
```

**Note:** there is a CI job which checks and reformat the sources with help of `clang-format`.
