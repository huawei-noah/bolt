* **Why configuring bolt.cmake does not take effect?**

    The [install.sh](../install.sh) serves as an example of compilation setup, and it overwrites some settings in [bolt.cmake](../common/cmakes/bolt.cmake). Please check install.sh first.

* **More details about dependency libraries for cross-compilation?**

    The major dependency is Protobuf for model conversion. Protoc should agree with your building platform but protbuf should be the ARM version. If you don't want to use model conversion, you can use *--converter=off* to close it.

* **How to use 1-bit BNN?**

    Bolt converts 0/1, -1/1 float weight to 1-bit bolt model. If you want to run 1-bit network, you can set model conversion precision to "BNN".

* **Restrictions for 1-bit BNN?**

    Only arm-v8.2+ CPU supports 1-bit inference. 

    For BNN convolution layers, the number of input channels must be divisible by 32, and the output channels must be divisible by 16, weight data must be 0/1 or -1/1.

* **How to use quantization (int8)?**

    Please refer to [QUANTIZATION.md](QUANTIZATION.md).  

* **Restrictions on quantization (int8)?**

    For the time being, Bolt only supports post-training int8 quantization. The quantization method is symmetrical for both activation and weight. We have added a calibration tool for image CNN pipelines. Please feel free to report cases of usage failure.

* **Requirements for float16 and int8?**

    Only arm-v8.2+ CPU supports float16 and int8 dotprod instructions. 

* **Restrictions for ARM Mali GPU?**

    Only *android-aarch64* target supports ARM Mali computing.

* **Exception in thread "main" java.lang.UnsatisfiedLinkError.**

    Because JNI's interface is wrong or not be compiled. If you are runing Java on X86 device, please set *JNI_ROOT* before compiling. *export JNI_ROOT=/usr/lib/jvm/java-8-openjdk-amd64*

    You can use *strings* command to see library symbol.

* **Restrictions for RNN/LSTM/PLSTM/GRU/GRU_LBR?**

    Only supports hidden states number mod 32 = 0 case. Bolt will change hidden state size to number that mod 32 == 0. If you don't want to change, you can set shell environment variable *BOLT_PADDING* to *OFF* before model conversion.

* **Some complex ONNX models can not be inferenced.**

    Maybe you can use [onnx-simplifiler](https://github.com/daquexian/onnx-simplifier) to simplify models. If you encounter problem when simplifing model, you can try to use *--skip-shape-inference* option.

* **How to use Java API apart from Android NDK?**

    Because JNI header files are not a part of compiler apart from Android NDK. You need to install openJDK to get JNI header files.

    Set *JNI_ROOT* shell environment variable, jni.h and jni_md.h are under <JNI_ROOT>/include or <JNI_ROOT>/include/linux directory.

    Run Bolt build and install script. You will see *-D_USE_API_JAVA* in cmake C/C++ compiler flags output information.

* **Time clock error on MacOS.**

    You may encounter this error when runing demos on MacOS. This is because file time is not valid.

    ```
    [0/1] Re-running CMake...
    -- Configuring done
    -- Generating done
    -- Build files have been written to: ...
    
    ninja: error: manifest 'build.ninja' still dirty after 100 tries
    ```
    You can solve it by runing this command under demo directory.
    ```
    find . -name "*" | xargs touch
    ```

* **Can not run android-aarch64 demo on ARMv8 phone**

    This is because some ARMv8 phone not supports armv8.2 feature(float16, int8 instruction), such as Huawei P20. This may tell you **illegal instruction** error or nothing.
    You can solve it by using *--fp16=off --int8=off* to close fp16 and int8 feature when building Bolt.

* **Can not run program on Android phone by using Android ADB**

    You can run C/C++ program in */data/local/tmp* directory by using ADB tool. If you don't have phone's root priviledge. you can only run test in this directory.

    Maybe you will encounter this problem.
    ```
    can't execute: Permission denied
    ```

    This is because program doesn't have execute priviledge. you can give it priviledge by using *chmod +x ./XXX_program*.

* **fatal: unable to access 'https://xxxxxxxxx': SSL certificate problem: unable to get local issuer certificate**

    You can disable ssl check by using command.
    ```
    git config --global http.sslVerify false
    ```

* **error while loading shared libraries: libOpenCL.dll: cannot open shared object file: No such file or directory**

    You can use windows system library /c/Windows/System32/OpenCL.dll, copy id to current directory and rename it to libOpenCL.dll.
    ```
    cp /c/Windows/System32/OpenCL.dll ./libOpenCL.dll
    export LD_LIBRARY_PATH=./
    ```
