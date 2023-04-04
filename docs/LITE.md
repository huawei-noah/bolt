# Lite version 

Bolt support a lite version library(~1MB) for ARM cortex-M and cortex-A7 processor.

Currently only support CNN, MLP model.

## How to convert model?

Lite version model is not same with non-lite version model, you can build Linux(x86) lite model conversion toolchains and convert model on Linux(x86).

```
./install.sh --target=linux-x86_64 --lite --serial=on --neon=off --int8=off --fp32=on
```

If you want to verify your model on x86 platform, you can build c api benchmark [c_image_classification](../inference/examples/c_api/c_image_classification.c), *c_image_classification* can inference any model.

```
cd inference/examples/c_api
./compile.sh linux-x86_64
./c_image_classification_static -m ./xxx_f32.bolt -l 10
```

* Note: If your target platform is not support file system, you can use *-B* option of *X2bolt* to get binary array model. and then define model array in your C/C++ code, Then you can pass it to *CreateModelWithFileStream* to create Bolt inference engine.

## How to lite library?

You need to download compiler and set *ARM_TOOLCHAIN_ROOT* to find compiler.
### Cortex-M by using GNU Arm Embedded Toolchain

```
export CC=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-gcc
export CXX=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-g++
export AR=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-ar
export READELF=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-readelf
export STRIP=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-strip
export CFLAGS="-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -fdata-sections -mthumb -mthumb-interwork --specs=nosys.specs"
export CXXFLAGS="${CFLAGS}"
./install.sh --target=generic-armv7_blank --lite --converter=off --serial=on --neon=off --int8=off --fp32=on
```

### Cortex-A armv7 by using GNU Arm Embedded Toolchain

```
export CC=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-gcc
export CXX=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-g++
export AR=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-ar
export READELF=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-readelf
export STRIP=${ARM_TOOLCHAIN_ROOT}/bin/arm-none-eabi-strip
export CFLAGS="-mfpu=neon-vfpv4 -mfloat-abi=hard -mcpu=cortex-a7 -mlittle-endian --specs=nosys.specs"
export CXXFLAGS="${CFLAGS}"
./install.sh --target=generic-armv7_blank --lite --converter=off --serial=off --neon=on --int8=off --fp32=on

```

### Cortex-A armv7 by using arm510 Toolchain

```
export CC=${ARM_TOOLCHAIN_ROOT}/bin/arm-mix510-linux-gcc
export CXX=${ARM_TOOLCHAIN_ROOT}/bin/arm-mix510-linux-g++
export AR=${ARM_TOOLCHAIN_ROOT}/bin/arm-mix510-linux-ar
export READELF=${ARM_TOOLCHAIN_ROOT}/bin/arm-mix510-linux-readelf
export STRIP=${ARM_TOOLCHAIN_ROOT}/bin/arm-mix510-linux-strip
export CFLAGS="-mfpu=neon-vfpv4 -mfloat-abi=hard -mcpu=cortex-a7"
export CXXFLAGS="${CFLAGS}"
./install.sh --target=generic-armv7_blank --lite --converter=off --serial=off --neon=on --int8=off --fp32=on

```

## Note

* If you want to use int8 inference, you can add *--int8=on* to build bolt. The model conversion flow is similar with non-lite version.

* Bolt will transform weight to specified format. This will use double RAM space when you use *CreateModelWithFileStream*. You can close the weight transform when you use serial inference. This can be implemented by skipping *transform_filter* function in [inference/engine/include/cpu/convolution_cpu.hpp](../inference/engine/include/cpu/convolution_cpu.hpp).
