#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
BOLT_ROOT=${script_dir}/..
current_dir=${PWD}

build_dir=$1
use_mali=$2

srcs=""
searchFiles() {
    for line in `find $1 -name "*.o"`
    do
        orginalStr=${line}
        subStr0="static"
        subStr1="model-tools"
        subStr2="tests"
        subStr3="data_loader"
        #if [[ ${line} =~ ${subStr0}   \
        if [[ !(${line} =~ ${subStr0})   \
          && !(${line} =~ ${subStr1}) \
          && !(${line} =~ ${subStr2}) \
          && !(${line} =~ ${subStr3}) ]];
        then
            srcs="${srcs} ${line}"
        fi
    done
}

searchFiles ${build_dir}

srcs="${srcs} ${build_dir}/model-tools/src/CMakeFiles/model-tools.dir/model_serialize_deserialize.cpp.o \
${build_dir}/model-tools/src/CMakeFiles/model-tools.dir/model_tools.cpp.o"

if [ -f "${BOLT_ROOT}/third_party/llvm/opencl/lib64/libOpenCL.so" ] && [ $use_mali == "ON" ];
then
    cp ${BOLT_ROOT}/third_party/llvm/opencl/lib64/libOpenCL.so ${build_dir}
    aarch64-linux-android-strip ${build_dir}/libOpenCL.so || exit 1
    #aarch64-linux-android-readelf -dW ${build_dir}/libOpenCL.so
fi

if [ -f "${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib/libkernelbin.so" ] && [ $use_mali == "ON" ];
then
    aarch64-linux-android-strip ${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib/libkernelbin.so || exit 1
    #cp ${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib/libkernelbin.so ${build_dir}
    aarch64-linux-android21-clang++ -shared -o ${build_dir}/libBoltModel.so ${srcs} \
        -L${BOLT_ROOT}/third_party/llvm/opencl/lib64 -lOpenCL \
        -L${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib -lkernelbin || exit 1
else
    aarch64-linux-android21-clang++ -shared -o ${build_dir}/libBoltModel.so ${srcs} || exit 1
fi
aarch64-linux-android-strip ${build_dir}/libBoltModel.so || exit 1
#aarch64-linux-android-readelf -dW ${build_dir}/libBoltModel.so
