# How to reduce gpu inference overhead
---
Bolt supports GPU float32 or float16 inference with OpenCL. 
But building OpenCL kernel from source code and selecting optimal algorithm takes up a lot of time. They can be optimized by preparing the OpenCL binary function library and algorithm file in advance. Inference can directly use prepared files.

- ## Build OpenCL binary function library

  Bolt provides offline tool [preprocess_ocl.sh](../inference/engine/tools/preprocess_ocl/preprocess_ocl.sh) to reduce GPU prepare time. 
  We have test mobilenet_v1 model on ARM MALI G76 GPU. Prepare time can be reduced from 2-3s to 60ms after building OpenCL binary kernel and algorithm file. 
  Here we give an example:

  - ### Android aarch64 platform

    <1> Connect target device by using Andriod *ADB*;
    
    <2> Convert your models to xxx.bolt by using *X2bolt*;
    
    <3> Create a directory on target device, copy all your needed xxx.bolt models into it, E.g:
    
    ```
    adb shell "mkdir /data/local/tmp/preprocess_bolt_models"
    adb shell "cp ${boltModelDir}/*.bolt /data/local/tmp/preprocess_bolt_models"
    ```
    
    <4> Set essential command line arguments for shell script [preprocess_ocl.sh](../inference/engine/tools/preprocess_ocl/preprocess_ocl.sh), and run it on host.
    
      - dNum: Device serial number, which can be aquired by using command
      
      ```
      adb devices
      ```
      
      - device_bolt_models: which is created in step <3>;

    for example:
        
      ```
      ./preprocess_ocl.sh --device 435bc850 --target android-aarch64 -d /data/local/tmp/preprocess_bolt_models
      ```
    After running *preprocess_ocl.sh* successfully, OpenCL binary function library *libxxx_map.so* will be produced. All needed kernels for your models has been compiled from sources to bins, and packaged into *libxxx_map.so*.

     <5> Use OpenCL binary kernel library to reduce gpu prepare time for your model

     <5.1> Reduce benchmark prepare time

      ```
       adb push install_android-aarch64/examples/benchmark /data/local/tmp/preprocess_bolt_models
       adb push libMali_G76p_map.so /data/local/tmp/preprocess_bolt_models
       adb shell "cd /data/local/tmp/preprocess_bolt_models && export LD_LIBRARY_PATH=./ && ./benchmark -m ./mobilenet_v1_f16.bolt -a GPU -l 8 -w 8"
      ```

     <5.2> Reduce C project prepare time
  
    Package kernel binary function library into your project, and put it in *libbolt.so* directory.

  - ### Windows platform
    
    <1> Convert your models to xxx.bolt by using *X2bolt*;
    
    <2> Create a directory on target device, copy all your needed xxx.bolt models into it, E.g:
    
    ```
    mkdir ./preprocess_bolt_models
    cp ${boltModelDir}/*.bolt ./preprocess_bolt_models
    ```
    
    <3> Set essential command line arguments for shell script [preprocess_ocl.sh](../inference/engine/tools/preprocess_ocl/preprocess_ocl.sh), and run it on host.
    
      for example:
        
      ```
      ./preprocess_ocl.sh --target=windows-x86_64_avx2 -d ./preprocess_bolt_models
      ```

      After running *preprocess_ocl.sh* successfully, OpenCL binary function library *libxxx_map.dll* and *kernel.bin* will be produced. All needed kernels for your models has been compiled from sources to bins, and packaged into *kernel.bin*.

     <4> Use OpenCL binary kernel library to reduce gpu prepare time for your model

     <4.1> Reduce benchmark prepare time

      ```
       cp libIntel_R__UHD_Graphics_630_map.dll ./
       cp kernel.bin ./
       export LD_LIBRARY_PATH=./ && ./install_windows-x86_64_avx2/examples/benchmark -m ./mobilenet_v1_f16.bolt -a GPU -l 8 -w 8"
      ```

     <4.2> Reduce C project prepare time

    Package kernel binary function library into your project, and put it in *libbolt.dll* directory.
  
    Note: Intel GPU may have longer time on new desktop computer, even if the GPU is same, but the GPU dirver is not same. If you encounter perormance problem on new desktop computer, you can use *update_ocl.exe* to generate new *kernel.bin* file.
    ```
    rm kernel.bin
    ls libIntel_R__UHD_Graphics_630_map.dll update_ocl.exe
    ls ./preprocess_bolt_models
    ./update_ocl.exe ./preprocess_bolt_models
    ```
   
- ### Note
  - OpenCL kernel functions are stored in the shared library in binary form. 
    Shared library is binding with specific GPU type and bolt models.
    Bolt will use system function *dlopen* to open shared library, please save it in same directory.
  - Please run prepare program under */data/local/tmp* directory for android devices to ensure the program has write permission.
  - Argument *algoPath* of C API *ModelHandle CreateModel(const char \*modelPath, AFFINITY_TYPE affinity, const char \*algoPath)* is abandoned in latest version, 
    algorithm file has been packaged into shared library, please set it to *NULL*.
