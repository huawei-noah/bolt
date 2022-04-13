# Build

[[_TOC_]]


## Reqirements

- Build system
    - cmake 3.11+
- Compilers
    - Clang 11.0.0+
    - GCC 9.2.1+
    - Visual Studio 16 (2019)+
    - Android NDK r22+ (r21 without assets related tests)

### Dependencies

Raul can be built with the following libraries. 
- [OpenBlas](https://github.com/xianyi/OpenBLAS)  
- [Yato](https://github.com/agruzdev/Yato)  

Build system downloads dependencies automatically using `cmake fetchcontent`.

Tests 
- [GTest](https://github.com/google/googletest)  
- [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) 


#### OpenBLAS

**Android**

OpenBLAS prebuilt binary for Android already included into repository `/src/thirdParty/openblas/lib/android/libopenblas.so` (no openmp). 

**Windows**

OpenBLAS prebuilt for win32: [download](https://sourceforge.net/projects/openblas/files/develop/20150903/). Required libgfortran-3.dll, libgcc_s_sjlj-1.dll (mingw32_dll) can be download from here: [v0.2.12](https://sourceforge.net/projects/openblas/files/v0.2.12/) or [v0.2.14](https://sourceforge.net/projects/openblas/files/v0.2.14/).

## Options 

### Build options

#### Build tests (`RAUL_BUILD_TESTS`)

**Default: OFF**

#### Build experiments (`RAUL_BUILD_EXPERIMENTS`)

**Default: OFF**

#### Build C API (`RAUL_BUILD_C_API`)

**Default: OFF**

#### Tests build options 

**Note**: it is available if `RAUL_BUILD_TESTS` enabled

**Default: ON** (all)

It is possible to switch off unnecessary tests to speed up the build.

- `RAUL_TESTS_BUILD_CORE`: Build core library unit tests
- `RAUL_TESTS_BUILD_ACTIVATIONS`: Build activation functions unit tests
- `RAUL_TESTS_BUILD_INITIALIZERS`: Build initializers unit tests
- `RAUL_TESTS_BUILD_LAYERS`: Build layers unit tests
- `RAUL_TESTS_BUILD_LOSS`: Build loss functions unit tests
- `RAUL_TESTS_BUILD_META`: Build meta layers unit tests
- `RAUL_TESTS_BUILD_OPTIMIZERS`: Build optimizers unit tests
- `RAUL_TESTS_BUILD_TOPOLOGIES`: Build topologies unit tests
- `RAUL_TESTS_BUILD_POSTPROCESSING`: Build postprocessing unit tests

### Config options

#### OpenMP

**Default: OFF**

```sh
cmake -B build -S raul -DRAUL_CONFIG_ENABLE_OPENMP=ON 
cmake --build build --target RaulLib --parallel
```
It uses `FindOpenMP` and can be customized, see cmake docs: https://cmake.org/cmake/help/latest/module/FindOpenMP.html


#### Pedantic mode (`RAUL_CONFIG_ENABLE_PEDANTIC`)

**Default: ON**

Enable more warnings and interpret them as errors.


#### BLAS configuration (`RAUL_CONFIG_BLAS_VENDOR`)

**Default: None**

Configures BLAS in raul library.

| Options             | Description                                                                 | Technical details                                                                                        |
| --------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| None                | Without BLAS                                                                | `RAUL_USE_BLAS=OFF`                                                                                      |
| OpenBLAS            | Search for OpenBLAS using `FindBLAS`                                        | `RAUL_USE_BLAS=ON`, `BLA_VENDOR=OpenBLAS`                                                                |
| Huawei              | Search for BLAS Enhance in repository *(android only)*                      | `RAUL_USE_BLAS=ON`, `BLA_VENDOR=Huawei` , `CMAKE_LIBRARY_PATH=<in repo>`, `CMAKE_INCLUDE_PATH=<in repo>` |
| Custom              | Manual set (provide variables)                                              | `RAUL_USE_BLAS=ON`, `BLAS_LIBRARIES=<ask>`, `BLAS_INCLUDE_DIR=<ask>`                                     |
| OpenBLAS (Internal) | Search for internal prebuilt OpenBLAS *(android, win32/win64, linux, migw)* | `RAUL_USE_BLAS=ON`, `BLAS_LIBRARIES=<in repo>`, `BLAS_INCLUDE_DIR=<in repo>`                             |
| Auto                | Search for any BLAS using `FindBLAS`                                        | `RAUL_USE_BLAS=ON`                                                                                       |


##### Custom BLAS: Example
Using the specific version of OpenBLAS.

1. Download, build and install custom openblas
```sh
wget https://github.com/xianyi/OpenBLAS/releases/download/v0.3.15/OpenBLAS-0.3.15.zip
unzip OpenBLAS-0.3.15.zip
make -C OpenBLAS-0.3.15/ -j72
sudo make -C OpenBLAS-0.3.15/ PREFIX=/opt/openblas install
```

2. Build Raul with manually provided openblas
```sh
cmake -B build -S raul -DRAUL_CONFIG_BLAS_VENDOR=Custom -DBLAS_LIBRARIES=/opt/openblas/lib/libopenblas.so -DBLAS_INCLUDE_DIR=/opt/openblas/include
cmake --build build --target RaulLib --parallel
```

#### OpenCL

**Default: ON**

#### 16 bit floating point (`RAUL_CONFIG_ENABLE_FP16`)

**Default: ON**

#### Parallize build mode (`RAUL_CONFIG_ENABLE_PARALLEL_BUILD`)

**Default: ON**

Parallel build

#### Cppcheck (`RAUL_CONFIG_DEV_ENABLE_CPPCHECK`)

**Default: OFF**

Search for linter `cppcheck`.

#### Clang tidy (`RAUL_CONFIG_DEV_ENABLE_CLANG_TIDY`)

**Default: OFF**

Search for linter `clang-tidy`.

#### Tests config options

**Note**: they are available if `RAUL_BUILD_TESTS` enabled

##### LibJPEG

**Default: OFF**

```sh
cmake -B build -S raul -DRAUL_BUILD_TESTS=ON -DRAUL_TESTS_CONFIG_ENABLE_LIBJPG=ON 
cmake --build build --target RaulLib --parallel
```

##### Verbose tests (`RAUL_TESTS_CONFIG_ENABLE_VERBOSE`)

**Default: OFF**

Enable stdout print for tests.

### Install options

#### Subdirectories (`RAUL_INSTALL_ENABLE_SUBDIRS`)

**Default: OFF**

#### Install tests (`RAUL_INSTALL_TESTS`)

**Note**: it is available if `RAUL_BUILD_TESTS` enabled

**Default: OFF**

## Tools

### Format

If `clang-format` is available in the system then `cmake` exposes format target

```sh
cmake -B build -S raul
cmake --build build --target format --parallel
```

We use [Mozilla coding style](https://firefox-source-docs.mozilla.org/code-quality/coding-style/coding_style_cpp.html). The sources can be automatically foramated with the following command.

**Note:** there is a CI job which checks and reformat the sources with help of `clang-format`.

### Docs

If `doxygen` is available in the system then `cmake` exposes format target

```sh
cmake -B build -S raul
cmake --build build --target docs --parallel
```

Make sure you have installed `doxygen` and `dot`. You can also instgall `mscgen` and `dia` optionally.

Also, there is a class diagram of the source code which is presented in docs/Raul.graphml (use [yED](https://www.yworks.com/products/yed) to open).

## Scenarios

### Cross-compilation for Android

If Android NDK is installed `cmake` exposes `build-android-tests` target (does not work with Microsoft Visual Studio).

```sh
cmake -B build -S raul
cmake --build build --target build-android-tests --parallel
```

This target includes:

- NDK toolchain (clang)
- Release config
- ABI arm64-v8a
- Android API 19
- C++ static std library
- OpenMP
- Huawei BLAS (BLAS Enhance, from repository)

#### NDK_PATH

`cmake` finds NDK by looking in system drive and program files on Windows and /opt directory on Linux. A custom directory can be provided throth `NDK_PATH` environment variable.

```sh
NDK_PATH=/mnt/android-ndk-r22/ cmake -B build -S raul
```

### CTest targets (`RAUL_TESTS_CONFIG_ENABLE_SCENARIOUS`)

**Note**: it is available if `RAUL_BUILD_TESTS` enabled

**Default: OFF**

It is possible to use a powerful cmake-bundled tool `ctest`. Several predefined testing scenarios have been added to the development environment for ctest.

```sh
cmake -B build -S raul -DRAUL_BUILD_TESTS=ON -DRAUL_TESTS_CONFIG_ENABLE_SCENARIOUS=ON
cmake --build build --target test-unit-optimizers --parallel
```
## Tests

### Datasets

Required for functional tests data can be found in `assets` subdirectory. See [assets/README.md](assets/README.md) for details.
