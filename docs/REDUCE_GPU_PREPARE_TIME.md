# How to reduce gpu initial time

Bolt support ARM Mali GPU, large addtitional prepare time is cost due to algorithm selecting and building kernel from source code. 

- ## Build extra resources for reducing prepare time on GPU

  Bolt provides offline tools [preprocess_ocl](../inference/engine/tools/preprocess_ocl/build_preprocess_ocl.sh) to reduce GPU prepare time. We have test mobilenet_v1 on MALI G76 GPU. Prepare time can be reduced from 2-3s to 60ms after build algorithm file and OpenCL kernel binary. Here we give an exaple:

  - ### Step By Step

    <1> Connect target device by Andriod ADB;
    
    <2> Convert your models to .bolt with X2bolt;
    
    <3> Make a write/read able folder on target device, copy all your needed .bolt models into it, E.g:
    
    ```
    adb shell "mkdir /data/local/tmp/preprocess_bolt_models"
    adb shell "cp ${boltModelDir}/*.bolt /data/local/tmp/preprocess_bolt_models"
    ```
    
    <4> Set essential variables for sh */inference/engine/tools/preproces_ocl/build_preprocess_ocl.sh*:
    
      - dNum: Device serial number, which can be aquired by using command
      
      ```
      adb devices
      ```
      
      - device_bolt_models: which is created in step <3>;

    for example:
      ```
      ./build_preprocess_ocl.sh --device dNum --target android-aarch64 -d device_bolt_models
      ```
        
    <5> Run *build_preprocess_ocl.sh* on host;

    After running build_preprocess_ocl.sh successfully, these extra xxxlib.so will be produced:
    
    - OpenCL kernel bin dynamic library: All needed kernels for your model has been compiled from sources to bins, and package into .so, such as: *${BOLT_ROOT}/inference/engine/tools/preprocess_ocl/lib/libMali_G76p_map.so*

- ## Use algorithm file and kernel binary dynamic library to reduce gpu prepare time for your model

  - ### Reduce Imagenet classification prepare time
  ```
    adb shell "mkdir /data/local/tmp/kits"
    adb push install_arm_llvm/kits/classification /data/local/tmp/kits
    adb push tools/preprocess_ocl/lib/libMali_G76p_map.so /data/local/tmp/kits
    adb shell "cd /data/local/tmp/kits && export LD_LIBRARY_PATH=./ && ./classification -m ./mobilenet_v1_f16.bolt -a GPU"
  ```

  - ### Reduce C project prepare time
  
    - Argument *algoPath* of C API *ModelHandle CreateModel(const char *modelPath, AFFINITY_TYPE affinity, const char *algoPath)* is used to set your algofile;
    - Argument *algoFileStread* of C API *ModelHandle CreateModelWithFileStream( const char *modelFileStream, AFFINITY_TYPE affinity, const char *algoFileStream)* is used to set your algofile filestream;
    - Package kernel binary dynamic library into your project;

- ## Note
  - Kernel binary dynamic library are binding with specific GPU type and your bolt models;
  - Please run it under file path "/data/local/tmp" for android devices to ensure the program get full authorities;
