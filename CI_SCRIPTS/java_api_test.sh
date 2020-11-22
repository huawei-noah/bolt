#!/bin/bash

device=$1
script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
BOLT_ROOT=${script_dir}/..
if [ ${device} == "x86_HOST" ]; then
    ci_dir=/data/bolt
    build_dir=${BOLT_ROOT}/build_x86_gnu
    install_dir=${BOLT_ROOT}/install_x86_gnu
else
    ci_dir=/data/local/tmp/CI
    build_dir=${BOLT_ROOT}/build_arm_llvm
    install_dir=${BOLT_ROOT}/install_arm_llvm
    device_dir=${ci_dir}/java
fi

current_dir=${PWD}

cd ${build_dir}
cp ${install_dir}/include/java/* .
cp ${BOLT_ROOT}/inference/examples/java_api/test_api_java.java .
javac BoltResult.java || exit 1
javac BoltModel.java || exit 1
javac test_api_java.java || exit 1

if [ ${device} != "x86_HOST" ]; then
    dx --dex --output=test_java_api.jar *.class || exit 1
    adb -s ${device} shell rm -rf ${device_dir}
    adb -s ${device} shell mkdir ${device_dir} || exit 1
    adb -s ${device} push ${install_dir}/lib/libBoltModel.so ${device_dir} > /dev/null || exit 1
    if [ -f "${install_dir}/lib/libkernelsource.so" ]; then
        adb -s ${device} push ${install_dir}/lib/libkernelsource.so ${device_dir} > /dev/null || exit 1
    fi
    if [ -f "${BOLT_ROOT}/third_party/arm_llvm/opencl/lib64/libc++_shared.so" ]; then
        adb -s ${device} push ${BOLT_ROOT}/third_party/arm_llvm/opencl/lib64/libc++_shared.so ${device_dir} > /dev/null || exit 1
    fi
    if [ -f "${install_dir}/lib/libOpenCL.so" ]; then
        adb -s ${device} push ${install_dir}/lib/libOpenCL.so ${device_dir} > /dev/null || exit 1
    fi
    adb -s ${device} push ./test_java_api.jar ${device_dir} > /dev/null || exit 1
    
    adb -s ${device} shell "cd ${device_dir} && export LD_LIBRARY_PATH=/apex/com.android.runtime/lib64/bionic:/system/lib64 && dalvikvm -cp ./test_java_api.jar test_api_java ${device} ${ci_dir}" 2> status.txt
else
    java test_api_java ${device} ${ci_dir} 2> status.txt
fi

if [ "$?" != 0 ]; then
    cat status.txt
    if cat ./status.txt | grep "couldn't find an OpenCL implementation" > /dev/null
    then
        echo "GPU environment error"
    else
        exit 1
    fi
fi

if [ ${device} != "x86_HOST" ]; then
    adb -s ${device} shell rm -rf ${device_dir}
fi

cd ${current_dir}
