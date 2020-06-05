#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
BOLT_ROOT=${script_dir}/..

build_dir=$1
use_mali=$2
use_debug=$3
use_android=$4
CXX=$5
AR=$6
STRIP=$7

allSrcs=""
skip_list=()
srcs=""
searchFiles() {
    srcs=""
    for line in ${allSrcs}
    do
        skip=false
        for str in "${skip_list[@]}"
        do
            if [[ ${line} =~ ${str} ]];
            then
                skip=true
                break
            fi
        done
        if [[ ${skip} == false ]]
        then
            srcs="${srcs} ${line}"
        fi
    done
}
allSrcs=`find ${build_dir} -name "*.o"`
skip_list=("static" "model-tools" "tests" "tools" "kits" "data_loader")
searchFiles
jniLibrarySrcs="${srcs} ${build_dir}/model-tools/src/CMakeFiles/model-tools.dir/model_deserialize.cpp.o \
${build_dir}/model-tools/src/CMakeFiles/model-tools.dir/model_tools.cpp.o"

allSrcs=`find ${build_dir} -name "*.o" | grep "static.dir"`
skip_list=("tests" "tools" "kits" "BoltModel_Jni")
searchFiles
staticLibrarySrcs="${srcs} ${build_dir}/model-tools/src/CMakeFiles/model-tools_static.dir/model_deserialize.cpp.o \
${build_dir}/model-tools/src/CMakeFiles/model-tools_static.dir/model_tools.cpp.o"

allSrcs=`find ${build_dir} -name "*.o"`
skip_list=("static" "tests" "tools" "kits")
searchFiles
sharedLibrarySrcs=${srcs}

if [ -f "${BOLT_ROOT}/third_party/llvm/opencl/lib64/libOpenCL.so" ] && [ $use_mali == "ON" ];
then
    cp ${BOLT_ROOT}/third_party/llvm/opencl/lib64/libOpenCL.so ${build_dir}
    ${STRIP} ${build_dir}/libOpenCL.so || exit 1
fi

if [ -f "${build_dir}/libbolt.a" ];
then
    rm -rf ${build_dir}/libbolt.a
fi
if [ -f "${build_dir}/libbolt.so" ];
then
    rm -rf ${build_dir}/libbolt.so
fi
if [ -f "${build_dir}/libBoltModel.so" ];
then
    rm -rf ${build_dir}/libBoltModel.so
fi

if [ -f "${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib/libkernelbin.so" ] && [ $use_mali == "ON" ];
then
    if [ $use_debug == "ON" ] && [ $use_android == "ON" ];
    then
        ${STRIP} ${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib/libkernelbin.so || exit 1
        ${CXX} -shared -o ${build_dir}/libBoltModel.so ${jniLibrarySrcs} \
            -L${BOLT_ROOT}/third_party/llvm/opencl/lib64 -lOpenCL \
            -L${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib -lkernelbin -llog || exit 1
    else
        ${CXX} -shared -o ${build_dir}/libBoltModel.so ${jniLibrarySrcs} \
            -L${BOLT_ROOT}/third_party/llvm/opencl/lib64 -lOpenCL \
            -L${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib -lkernelbin || exit 1
    fi
    ${CXX} -shared -o ${build_dir}/libbolt.so ${sharedLibrarySrcs} \
        -L${BOLT_ROOT}/third_party/llvm/opencl/lib64 -lOpenCL \
        -L${BOLT_ROOT}/gcl/tools/kernel_lib_compile/lib -lkernelbin || exit 1
else
    if [ $use_debug == "ON" ] && [ $use_android == "ON" ];
    then
        ${CXX} -shared -o ${build_dir}/libBoltModel.so ${jniLibrarySrcs} -llog || exit 1
    else
        ${CXX} -shared -o ${build_dir}/libBoltModel.so ${jniLibrarySrcs} || exit 1
    fi
    ${CXX} -shared -o ${build_dir}/libbolt.so ${sharedLibrarySrcs} || exit 1
fi
${AR} -rc ${build_dir}/libbolt.a ${staticLibrarySrcs} || exit 1

if [ $use_debug == "OFF" ];
then
    ${STRIP} ${build_dir}/libBoltModel.so || exit 1
    ${STRIP} ${build_dir}/libbolt.so || exit 1
    ${STRIP} -g -S -d --strip-debug --strip-unneeded ${build_dir}/libbolt.a || exit 1
fi
