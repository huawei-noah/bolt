# Contents
&nbsp;&nbsp;&nbsp;&nbsp;[Prerequisites](#prerequisites)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Linux System Compilation Tools](#linux-system-compilation-tools)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Windows System Compilation Tools](#windows-system-compilation-tools)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[MacOS System Compilation Tools](#macos-system-compilation-tools)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Android Cross-Compilation Tools](#android-cross-compilation-toolsoptional)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Linux-AArch64 Cross-Compilation Tools](#linux-aarch64-cross-compilation-toolsoptional)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[iOS Cross-Compilation Tools](#ios-cross-compilation-toolsoptional)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Tools](#tools)  
&nbsp;&nbsp;&nbsp;&nbsp;[Download and Build Bolt](#download-and-build-bolt)  
&nbsp;&nbsp;&nbsp;&nbsp;[Common install problem](#common-install-problem)  


# Prerequisites

## Linux System Compilation Tools

- ### CMake

   Download and install Cmake from <https://cmake.org/download/>. Set shell environment variable **PATH**.

- ### GNU make

   Download and install make from <https://ftp.gnu.org/gnu/make/>. Set shell environment variable **PATH**.

- ### Wget

   Download and install Wget from <https://www.gnu.org/software/wget/>. Set shell environment variable **PATH**.

## Windows System Compilation Tools

- ### Git Shell

   Download and install Git Shell from <https://gitforwindows.org/>. Set shell environment variable **PATH**.

- ### CMake

   Download and install Cmake from <https://cmake.org/download/>. Set shell environment variable **PATH**.

- ### Wget

   Download and install Wget from <https://eternallybored.org/misc/wget/>. Set shell environment variable **PATH**.

- ### MinGW toolchains(mingw32-make, gcc, g++)

   Download and install Mingw32-w64 from <https://udomain.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/6.4.0/threads-posix/seh/x86_64-6.4.0-release-posix-seh-rt_v5-rev0.7z>.

## MacOS System Compilation Tools

- ### Xcode

   Download and install Xcode.

## Android Cross-Compilation Tools(optional)

- ### Android NDK

    Refer to the [NDK installation example](https://askubuntu.com/questions/837847/how-to-install-android-ndk) to install [android-ndk-r20](https://developer.android.google.cn/ndk/downloads) and set shell environment variable **ANDROID_NDK_ROOT**.

    ```
    export ANDROID_NDK_ROOT=/data/opt/android-ndk-r20
    ```

## Linux-AArch64 Cross-Compilation Tools(optional)

- ### Cross compiler

    Install [gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu](https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz?revision=2e88a73f-d233-4f96-b1f4-d8b36e9bb0b9&la=en&hash=167687FADA00B73D20EED2A67D0939A197504ACD) and set shell environment **PATH**.

## iOS Cross-Compilation Tools(optional)

- ### Cross compiler for Linux

    [Here is an tutorial for building toolchains](IOS_USAGE.md).

- ### Cross compiler for MacOS
  
    You can use MacOS's clang to build. Only need to set shell environment **IOS_SDK_ROOT** to iPhoneOS.sdk.

    ```
    export IOS_SDK_ROOT=/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk
    ```

## Tools

- ### Android adb

  Refer to the [ADB installation example](https://unix.stackexchange.com/questions/378041/how-to-install-adb-on-ubuntu-from-download) to install [ADB](https://developer.android.com/studio/command-line/adb.html) tool helping you transfer the executables to android mobile phones.

  ```
  export PATH=/data/opt/platform-tools:$PATH
  ```

- ### Android dx

  If you want to directly run *jar* file on Android device, you can use [Android dx tool](https://developer.android.com/studio/releases/build-tools). Install Android *v28.0.3* build tools and set shell environment **PATH**.


- ### JDK

  If you want to use Java API without Android NDK, you need to install JDK.
  
  Download and install [OpenJDK](http://openjdk.java.net/install/) and set shell environment **PATH** and **JNI_ROOT**.
  
  ```
  export JNI_ROOT=/data/opt/openjdk-16_windows-x64_bin
  export PATH=${JNI_ROOT}/bin:$PATH
  ```

# Download and Build Bolt

A simple shell script [install.sh](../install.sh) is provided to build and install the Bolt library, and you should modify it according to your scenario and environment. Use help message to find more useful information. 

We will install Bolt to *install_[target]* directory. These subdirectories will be found in it:

- include
    - [C API](../inference/engine/api/c) header file
    - [Java API](../inference/engine/api/java) class file
- lib
    - libBoltModel.so: build for Java application
    - libbolt.so: build for C/C++ application
    - libflow.so: flow sub project library, when using --flow option
    - libinference.so: inference sub project library
    - libtensor.so: tensor computing sub project library
    - libimage.so: image sub project library
    - libblas_enhance.so: blas_enhance sub project library
    - libmodel_tools.so: model_tools sub project library
    - libuni.so: uni sub project library
- tools
    - *X2bolt* for generally converting deep learning(caffe/onnx/tflite) model to bolt model
    - *tensorflow2caffe* for converting tensorflow model to caffe model
    - *pytorch2caffe* for converting pytorch model to caffe model
    - *tensor_computing_library_search* for performance tuning of the operator library
- tests
    - operator unit test
- examples
    - *benchmark* for measuring inference performance of any model (.bolt)
    These examples will be build when using "--test" install option.
    - *classification* for imagenet classification task
    - *tinybert* for intention identification
    - *nmt* for machine translation
    - *asr_rnnt* for automatic speech recognition task (RNNT model)
    - *asr_convolution_transformer* for automatic speech recognition task (Convolution+Transformer model)
    - *tts* for text to speech
- docs
    - API/html: doxygen html document for C/Java/Flow API   

# Common install problem

- ### wget error

  Use wget to download file. If you use proxy to access the network, you may be reminded to add *--no-check-certificate* flag when using wget.
  
- ### Download is limited by network proxy or time consuming.

  You can download these files and save to a specified directory, Bolt will automatically use it.
  
  1. save Linux protoc <https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-linux-x86_64.zip> 
      or Windows protoc <https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-win64.zip> 
      or MacOS protoc <https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-osx-x86_64.zip> to *third_party/sources/* directory.
  2. save <https://github.com/protocolbuffers/protobuf/archive/v3.14.0.tar.gz> to *third_party/sources/protobuf-3.14.0.tar.gz*.
  3. save <https://github.com/google/flatbuffers/tree/master/include> to *third_party/sources/flatbuffers/include*.
  4. save <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_generated.h> *third_party/sources/tflite/include/tensorflow/lite/schema/schema_generated.h*.
  5. save <https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.4.zip> to *third_party/sources/jsoncpp-1.9.4.zip*.
  6. optional. save <https://github.com/KhronosGroup/OpenCL-Headers/tree/master/CL> to *third_party/sources/opencl/include/CL** when using ARM MALI GPU.
  7. optional. use *ADB* to pull android phone's </vendor/lib64/libOpenCL.so> and </vendor/lib64/egl/libGLES_mali.so> to *third_party/sources/opencl/lib64* when using ARM MALI GPU.
  8. optional. save <http://www.ijg.org/files/jpegsrc.v9c.tar.gz> to *third_party/sources/jpegsrc.v9c.tar.gz* when using example.
  9. optional. save <https://codeload.github.com/anthonix/ffts/zip/master> to *third_party/sources/ffts-master.zip* when using Flow.

- ### MinGW version error

  Third party library protobuf use some POSIX standard system library. If you don't use POSIX version mingw, you may encounter these errors.
  You can download this link to download POSIX version mingw. <https://udomain.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/6.4.0/threads-posix/seh/x86_64-6.4.0-release-posix-seh-rt_v5-rev0.7z>

  ```
  error: 'mutex' in namespace 'std' does not name a type
   std::mutex mu_;
  error: 'once_flag' in namespace 'std' does not name a type
   using once_flag = std::once_flag;
  error: 'call_once' is not a member of 'std'
   std::call_once(std::forward<Args>(args)...);
  error: 'strtoll' was not declared in this scope
   return strtoll(nptr, endptr, base);
  error: 'thread' is not a member of 'std'
   static std::atomic<std::thread::id> runner;
  ```

- ### Don't want to use third party library or model conversion tools.

  Third party library are used in model conversion tools. If you don't want to use it, you can close it by using *--converter=OFF* option. This will not build third party library.

- ### Only want to use partial model conversion tools.

  You can implement it by changing install.sh. for example, there are some cmake options, such as *-DUSE_CAFFE=ON*.

- ### Only want to use partial inference precision.

  You can implement it by changing install.sh. for example, there are some cmake options, such as *-DUSE_INT8=ON*.

- ### Can not build success with special compiler(such as MinGW)

  You may encounter various compilation problem, this maybe caused by compiler or others. Here is an example.

  ```
  mingw64\bin\ar.exe: unable to rename 'CMakeFiles\test_softmax.dir/objects.a'; reason: File exists
  ```

  You can enter build directory *build_[target]* and continuously run *make install*. This may complete all compilation step by step.
