#!/bin/bash

script_name=$0
script_dir=$(cd `dirname $0` && pwd)
log_file="upload.log"

platform=""
device=""
device_dir=""

getopt --test
if [[ "$?" != "4" ]]; then
    echo -e "[ERROR] you are using BSD getopt, not GNU getopt. If you are runing on Mac, please use this command to install gnu-opt.\n    brew install gnu-getopt && brew link --force gnu-getopt"
    exit 1
fi
TEMP=`getopt -o "d:p:" --long device:,path:,platform: \
     -n ${script_name} -- "$@"`
if [[ $? != 0 ]]; then
    echo "[ERROR] ${script_name} terminating..." >&2
    exit 1
fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -d|--device)
            device=$2
            shift 2 ;;
        -p|--path)
            device_dir=$2
            shift 2 ;;
        --platform)
            platform=$2
            shift 2 ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] ${script_name} can not recognize $1" ; exit 1 ;;
    esac
done

source ${script_dir}/../third_party/${platform}.sh

adb -s ${device} shell "mkdir -p ${device_dir}" || exit 1
if [[ "${Protobuf_ROOT}" != "" && -d "${Protobuf_ROOT}/lib" && -f "${Protobuf_ROOT}/lib/libprotobuf.so" ]]; then
    for file in `ls ${Protobuf_ROOT}/lib/*.so*`
    do
        adb -s ${device} push ${file} ${device_dir} > ${log_file} || exit 1
    done
fi
adb -s ${device} shell cp /vendor/lib64/libOpenCL.so ${device_dir} > ${log_file} || exit 1
if [[ "${JPEG_ROOT}" != "" && -d "${JPEG_ROOT}/lib" && -f "${JPEG_ROOT}/lib/libjpeg.so" ]]; then
    for file in `ls ${JPEG_ROOT}/lib/*.so*`
    do
        adb -s ${device} push ${file} ${device_dir}  > ${log_file} || exit 1
    done
fi
if [[ "${JPEG_ROOT}" != "" && -d "${JSONCPP_ROOT}/lib" && -f "${JSONCPP_ROOT}/lib/libjsoncpp.so" ]]; then
    for file in `ls ${JSONCPP_ROOT}/lib/*.so*`
    do
        adb -s ${device} push ${file} ${device_dir} > ${log_file} || exit 1
    done
fi
cxx_shared_path=""
if [[ ${platform} =~ android-aarch64 ]]; then
    clang_path=`which aarch64-linux-android21-clang++`
    clang_dir=$(dirname ${clang_path})
    cxx_shared_path=${clang_dir}/../sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
fi
if [[ ${platform} =~ android-armv7 ]]; then
    clang_path=`which armv7a-linux-androideabi19-clang++`
    clang_dir=$(dirname ${clang_path})
    cxx_shared_path=${clang_dir}/../sysroot/usr/lib/arm-linux-androideabi/libc++_shared.so
fi
if [[ -f "${cxx_shared_path}" ]]; then
    adb -s ${device} push ${cxx_shared_path} ${device_dir} > ${log_file} || exit 1
fi

if [[ -f "${log_file}" ]]; then
    rm ${log_file}
fi
