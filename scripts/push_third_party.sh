#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

host_lib_dir=""
device=""
device_dir=""

TEMP=`getopt -o l:d:p: --long lib:device:path \
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
        --) shift ;
            break ;;
        *) echo "[ERROR] $1" ; exit 1 ;;
    esac
done

adb -s ${device} push ${host_lib_dir}/protobuf/lib/libprotobuf.so.11  ${device_dir} || exit 1
adb -s ${device} push ${host_lib_dir}/opencl/lib64  ${device_dir} || exit 1
adb -s ${device} push ${host_lib_dir}/jpeg/lib/libjpeg.so.9  ${device_dir} || exit 1
