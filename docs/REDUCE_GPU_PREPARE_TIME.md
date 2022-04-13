# How to reduce gpu inference overhead
---
Bolt supports ARM GPU inference with OpenCL. 
But building OpenCL kernel function from source code and selecting optimal algorithm takes up a lot of time.
They can be optimized by preparing the OpenCL binary function library and algorithm file in advance.
Inference can directly use prepared files.

- ### Build OpenCL binary kernel library

  Bolt provides offline tool [preprocess_ocl](../inference/engine/tools/preprocess_ocl/build_preprocess_ocl.sh) to reduce GPU prepare time. 
  We have test mobilenet_v1 model on ARM MALI G76 GPU. Prepare time can be reduced from 2-3s to 60ms after building OpenCL binary kernel and algorithm file. 
  Here we give an example:

  - #### Step By Step

    <1> Connect target device by using Andriod *ADB*;
    
    <2> Convert your models to xxx.bolt by using *X2bolt*;
    
    <3> Create a directory on target device, copy all your needed xxx.bolt models into it, E.g:
    
    ```
    adb shell "mkdir /data/local/tmp/preprocess_bolt_models"
    adb shell "cp ${boltModelDir}/*.bolt /data/local/tmp/preprocess_bolt_models"
    ```
    
    <4> Set essential command line arguments for shell script [preprocess_ocl](../inference/engine/tools/preprocess_ocl/build_preprocess_ocl.sh):
    
      - dNum: Device serial number, which can be aquired by using command
      
      ```
      adb devices
      ```
      
      - device_bolt_models: which is created in step <3>;

    for example:
        
      ```
      ./build_preprocess_ocl.sh --device 435bc850 --target android-aarch64 -d /data/local/tmp/preprocess_bolt_models
      ```
        
    <5> Run *build_preprocess_ocl.sh* on host;

    After running build_preprocess_ocl.sh successfully, OpenCL binary kernel shared library libxxx_map.so will be produced.
    All needed kernels for your models has been compiled from sources to bins, 
    and packaged into libxxx_map.so, such as *${BOLT_ROOT}/inference/engine/tools/preprocess_ocl/lib/libMali_G76p_map.so*

- ### Use OpenCL binary kernel library to reduce gpu prepare time for your model

  - #### Reduce Imagenet classification prepare time

      ```
       adb shell "mkdir /data/local/tmp/kits"
       adb push install_arm_llvm/kits/classification /data/local/tmp/kits
       adb push tools/preprocess_ocl/lib/libMali_G76p_map.so /data/local/tmp/kits
       adb shell "cd /data/local/tmp/kits && export LD_LIBRARY_PATH=./ && ./classification -m ./mobilenet_v1_f16.bolt -a GPU"
      ```

  - #### Reduce C project prepare time
  
    Package kernel binary dynamic library into your project, and put it in *libbolt.so* directory.

- ### Note
  - OpenCL kernel functions are stored in the shared library libxxx_map.so in binary form. 
    Shared library libxxx_map.so is binding with specific GPU type and bolt models.
    Bolt will use C system function *dlopen* to open shared library libxxx_map.so, please save it in same directory.
  - Please run prepare program under */data/local/tmp* directory for android devices to ensure the program has write permission.
  - Argument *algoPath* of C API *ModelHandle CreateModel(const char \*modelPath, AFFINITY_TYPE affinity, const char \*algoPath)* is abandoned in latest version, 
    algorithm file has been packaged into libxxx_map.so, please set it to *NULL*.