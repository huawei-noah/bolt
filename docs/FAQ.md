# FAQ on Bolt

1. Why configuring bolt.cmake does not take effect?

   The [install.sh](install.sh) serves as an example of compilation setup, and it overwrites some settings in [bolt.cmake](common/cmakes/bolt.cmake). Please check install.sh first.

2. More details about dependency libraries for cross-compilation?

   The major dependency is Protobuf. Protoc should agree with your building platform but protbuf should be the ARM version.

3. Restrictions for 1-bit BNN?

   For BNN convolution layers, the number of input channels must be divisible by 32, and the output channels must be divisible by 16.

4. Restrictions on quantization (int8)?

   For the time being, Bolt only supports post-training int8 quantization. The quantization method is symmetrical for both activation and weight. We have added a calibration tool for image CNN pipelines. Please feel free to report cases of usage failure.

5. Requirements for float16 and int8?

   Only arm-v8.2 CPU supports float16 and int8 dotprod instructions. 

6. Restrictions for ARM Mali GPU?

   Only *arm_llvm* compilation supports ARM Mali computing.