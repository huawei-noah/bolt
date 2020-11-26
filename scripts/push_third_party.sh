#!/bin/bash

script_name=$0
script_dir=$(cd `dirname $0` && pwd)

host_lib_dir=""
device=""
device_dir=""
compiler=""

TEMP=`getopt -o l:d:p:c: --long lib:device:path:compiler: \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -l|--lib)
            host_lib_dir=$2
            echo "[INFO] use library in ${host_lib_dir}" ;
            shift 2 ;;
        -d|--device)
            device=$2
            echo "[INFO] test on device ${device}" ;
            shift 2 ;;
        -p|--path)
            device_dir=$2
            echo "[INFO] test on device directory ${device_dir}" ;
            shift 2 ;;
        -c|--compiler)
            compiler=$2
            echo "[INFO] push ${compiler} library" ;
            shift 2 ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] $1" ; exit 1 ;;
    esac
done

adb -s ${device} push ${host_lib_dir}/protobuf/lib/libprotobuf.so.11 ${device_dir} > /dev/null || exit 1
if [[ "${compiler}" == "arm_llvm" ]]; then
    adb -s ${device} push ${host_lib_dir}/opencl/lib64 ${device_dir} > /dev/null || exit 1
    if [[ -f "${host_lib_dir}/opencl/lib64/libc++_shared.so" ]]; then
        cxx_shared_path=${host_lib_dir}/opencl/lib64/libc++_shared.so
    else
        clang_path=`which aarch64-linux-android21-clang++`
        clang_dir=$(dirname ${clang_path})
        cxx_shared_path=${clang_dir}/../sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
    fi
fi
if [[ "${compiler}" == "arm_ndkv7" ]]; then
    clang_path=`which armv7a-linux-androideabi19-clang++`
    clang_dir=$(dirname ${clang_path})
    cxx_shared_path=${clang_dir}/../sysroot/usr/lib/arm-linux-androideabi/libc++_shared.so
fi
if [[ -f "${cxx_shared_path}" ]]; then
    adb -s ${device} push ${cxx_shared_path} ${device_dir} > /dev/null || exit 1
fi
adb -s ${device} push ${host_lib_dir}/jpeg/lib/libjpeg.so.9 ${device_dir} > /dev/null || exit 1
adb -s ${device} push ${host_lib_dir}/jsoncpp/lib/libjsoncpp.so ${device_dir} > /dev/null || exit 1
