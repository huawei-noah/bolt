#!/bin/bash

script_dir=$(cd `dirname $0` && pwd)
BOLT_ROOT=${script_dir}/..

CXX=$1
AR=$2
STRIP=$3
build_dir=$4
use_mali=$5
use_debug=$6
use_android=$7
use_android_log=$8
use_ios=$9
use_openmp=${10}

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
            srcs="${srcs} ${build_dir}/${line}"
        fi
    done
}
if [ $use_ios == "OFF" ];
then
    allSrcs=`find ${build_dir} -name "*.o" -printf "%P\n"`
    skip_list=("static" "model_tools" "tests" "tools" "examples" "flow" "data_loader")
    searchFiles
    jniLibrarySrcs="${srcs} \
    ${build_dir}/model_tools/src/CMakeFiles/model_tools.dir/model_tools.cpp.o"
fi

allSrcs=`find ${build_dir} -name "*.o" -printf "%P\n"| grep "static.dir"`
skip_list=("tests" "tools" "examples" "BoltModel_Jni" "flow" "data_loader")
searchFiles
staticLibrarySrcs="${srcs} \
${build_dir}/model_tools/src/CMakeFiles/model_tools_static.dir/model_tools.cpp.o"

allSrcs=`find ${build_dir} -name "*.o" -printf "%P\n"`
skip_list=("static" "tests" "tools" "examples" "BoltModel_Jni" "flow" "data_loader")
searchFiles
sharedLibrarySrcs="${srcs} \
${build_dir}/model_tools/src/CMakeFiles/model_tools_static.dir/model_tools.cpp.o"

if [ -f "${build_dir}/common/gcl/tools/kernel_source_compile/libkernelsource.so" ] && [ $use_mali == "ON" ];
then
    gclLibrarySrcs="${build_dir}/common/gcl/tools/kernel_source_compile/CMakeFiles/kernelsource.dir/src/cl/gcl_kernel_source.cpp.o \
        ${build_dir}/common/gcl/tools/kernel_source_compile/CMakeFiles/kernelsource.dir/src/cl/inline_cl_source.cpp.o \
        ${build_dir}/common/gcl/tools/kernel_source_compile/CMakeFiles/kernelsource.dir/src/option/gcl_kernel_option.cpp.o \
        ${build_dir}/common/gcl/tools/kernel_source_compile/CMakeFiles/kernelsource.dir/src/option/inline_cl_option.cpp.o"
    jniLibrarySrcs="${jniLibrarySrcs} ${gclLibrarySrcs}"
    staticLibrarySrcs="${staticLibrarySrcs} ${gclLibrarySrcs}"
    sharedLibrarySrcs="${sharedLibrarySrcs} ${gclLibrarySrcs}"
fi

if [ -f "${BOLT_ROOT}/third_party/arm_llvm/opencl/lib64/libOpenCL.so" ] && [ $use_mali == "ON" ];
then
    cp ${BOLT_ROOT}/third_party/arm_llvm/opencl/lib64/libOpenCL.so ${build_dir}
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
if [ -f "${build_dir}/libbolt.dylib" ];
then
    rm -rf ${build_dir}/libbolt.dylib
fi
if [ -f "${build_dir}/libBoltModel.so" ];
then
    rm -rf ${build_dir}/libBoltModel.so
fi

lib=""
if [ $use_android_log == "ON" ] && [ $use_android == "ON" ];
then
    lib="${lib} -llog"
fi
if [ $use_openmp == "ON" ];
then
    lib="${lib} -fopenmp"
fi
if [ -f "${build_dir}/common/gcl/tools/kernel_source_compile/libkernelsource.so" ] && [ $use_mali == "ON" ];
then
    ${STRIP} ${build_dir}/common/gcl/tools/kernel_source_compile/libkernelsource.so || exit 1
    lib="${lib} -L${BOLT_ROOT}/third_party/arm_llvm/opencl/lib64 -lOpenCL"
fi

if [ $use_ios == "ON" ];
then
    ${CXX} -shared -o ${build_dir}/libbolt.dylib ${sharedLibrarySrcs} ${lib} || exit 1
else
    ${CXX} -shared -o ${build_dir}/libBoltModel.so ${jniLibrarySrcs} ${lib} -Wl,-soname,libBoltModel.so || exit 1
    ${CXX} -shared -o ${build_dir}/libbolt.so ${sharedLibrarySrcs} ${lib} -Wl,-soname,libbolt.so || exit 1
fi

${AR} -rc ${build_dir}/libbolt.a ${staticLibrarySrcs} || exit 1

if [ $use_debug == "OFF" ];
then
    if [ $use_ios == "OFF" ];
    then
        ${STRIP} ${build_dir}/libBoltModel.so || exit 1
    fi
    if [ $use_ios == "OFF" ];
    then
        ${STRIP} ${build_dir}/libbolt.so || exit 1
        ${STRIP} -g -S -d --strip-debug --strip-unneeded ${build_dir}/libbolt.a || exit 1
    fi
fi
