#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

host_dir=""
use_static_library=true
host_lib_dir=""
exe_on_device=true
array=($(adb devices | sed 's/\r//' | grep ".device$"))
device=${array[0]}
array=($(adb devices | grep ".device$"))
#device=${array[0]}
device="GCL5T19822000030"
cpu_mask="40"
device_dir=""
gpu=false

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Run operator test.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -t, --test <PATH>          run specified program in <PATH>/[tests|examples|kits].
  -l, --lib <PATH>           use specified library in <PATH>.
  -d, --device <device_id>   run test on device.
  -c, --cpu_mask <mask>      taskset cpu mask(default: 40).
  -g, --gpu                  run gpu test.
  -p, --path <path>          run test on device in specified PATH.
EOF
    exit 1;
}


TEMP=`getopt -o t:c:hl:d:p:g --long test:cpu_mask:help,lib:device:path:gpu \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -t|--test)
            host_dir=$2
            echo "[INFO] run test in '${host_dir}'" ;
            shift 2 ;;
        -c|--cpu_mask)
            cpu_mask=$2
            echo "[INFO] CPU mask '${cpu_mask}'" ;
            shift 2 ;;
        -l|--lib)
            use_static_library=false;
            host_lib_dir=$2
            echo "[INFO] use library in ${host_lib_dir}" ;
            shift 2 ;;
        -d|--device)
            device=$2
            exe_on_device=true
            echo "[INFO] test on device \`${device}'" ;
            shift 2 ;;
        -p|--path)
            device_dir=$2
            echo "[INFO] test on device directory \`${device_dir}'" ;
            shift 2 ;;
        -g|--gpu)
            gpu=true;
            shift ;;
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR]" ; exit 1 ;;
    esac
done


if [ ${exe_on_device}  == true ] ; then
    status=`adb -s ${device} shell "ls ${device_dir}/ && echo 'success'" | tail -n 1`
    if [ "${status}" != "success" ] ; then
        adb -s ${device} shell "rm -rf ${device_dir}"
        adb -s ${device} shell "mkdir ${device_dir}"
    fi
    if [ ${use_static_library} != true ] ; then
        for file in `ls ${host_lib_dir}/*.so`
        do
            adb -s ${device} push ${file} ${device_dir} > /dev/null || exit 1
        done
        ${script_dir}/push_third_party.sh -l ${script_dir}/../third_party/arm_llvm -d ${device} -p ${device_dir} -c arm_llvm || exit 1
    fi
fi

function device_excute(){
    adb -s ${device} shell "export LD_LIBRARY_PATH=${device_dir} && taskset ${cpu_mask} $@ || echo '[FAILURE]'" &> status.txt
    cat status.txt || exit 1
    if [ `grep -c "\[FAILURE\]" status.txt` -ne '0' ] ; then
        exit 1
    fi
    rm status.txt
}

# mmm
adb -s ${device} push ${host_dir}/tests/test_mmm_int8 ${device_dir} > /dev/null || exit 1
adb -s ${device} push ${host_dir}/tests/test_mmm ${device_dir} > /dev/null || exit 1
echo " " ; echo "--- Matrix Matrix Multiplication"
device_excute ${device_dir}/test_mmm 384 768 768

# conv_ic=3
adb -s ${device} push ${host_dir}/tests/test_convolution ${device_dir} > /dev/null || exit 1
echo " " ; echo "--- Conv IC=3"
device_excute ${device_dir}/test_convolution 1 3 227 227 96 3 11 11 1 4 0 1 96 55 55

# conv_5x5
adb -s ${device} push ${host_dir}/tests/test_convolution_bnn ${device_dir} > /dev/null || exit 1
adb -s ${device} push ${host_dir}/tests/test_convolution_int8 ${device_dir} > /dev/null || exit 1
echo " " ; echo "--- Conv 5x5"
device_excute ${device_dir}/test_convolution_bnn 1 96 27 27 256 96 5 5 1 2 0 1 256 13 13
device_excute ${device_dir}/test_convolution_int8 1 96 27 27 256 96 5 5 1 2 0 1 256 13 13
device_excute ${device_dir}/test_convolution 1 96 27 27 256 96 5 5 1 2 0 1 256 13 13

# conv_3x3
echo " " ; echo "--- Conv 3x3"
device_excute ${device_dir}/test_convolution_bnn 1 128 28 28 256 128 3 3 1 1 1 1 256 28 28
device_excute ${device_dir}/test_convolution_int8 1 128 28 28 256 128 3 3 1 1 1 1 256 28 28
device_excute ${device_dir}/test_convolution 1 128 28 28 256 128 3 3 1 1 1 1 256 28 28

# depthwise-pointwise convolution
adb -s ${device} push ${host_dir}/tests/test_depthwise_convolution ${device_dir} > /dev/null || exit 1
echo " " ; echo "--- Depthwise-Pointwise Conv"
device_excute ${device_dir}/test_depthwise_convolution 1 256 28 28 256 256 3 3 1 1 1 1 256 28 28

# OCL
if [ ${gpu}  == true ] ; then
    adb -s ${device} push ${host_dir}/examples/hdr ${device_dir} > /dev/null || exit 1
    echo " " ; echo " " ; echo "--- GPU Network Test (HDR_OCL)"
    echo " " ; echo "=== Input FP16"
    device_excute ${device_dir}/hdr 1 3 720 1280
    echo " " ; echo "=== Input UCHAR"
    device_excute ${device_dir}/hdr 1 3 720 1280 UCHAR

    adb -s ${device} push ${host_dir}/tests/test_convolution_ocl ${device_dir} > /dev/null || exit 1
    adb -s ${device} push ${host_dir}/tests/test_depthwise_convolution_ocl ${device_dir} > /dev/null || exit 1
    adb -s ${device} push ${host_dir}/tests/test_fully_connected_ocl ${device_dir} > /dev/null || exit 1
    device_excute ${device_dir}/test_convolution_ocl 64 112 112 64 5 5 1 2
    device_excute ${device_dir}/test_convolution_ocl 64 112 112 64 3 3 1 1
    device_excute ${device_dir}/test_depthwise_convolution_ocl 64 112 112 64 3 3 1 1
    device_excute ${device_dir}/test_fully_connected_ocl 24 1 1 96 
fi

adb -s ${device} shell "rm -rf ${device_dir}"
