#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
driver_script_path="${script_dir}/operator_driver.sh"

host_bin_dir=""
use_static_library=true
host_lib_dir=""
excute_on_device=false
device=""
cpu_mask="2"
device_dir=""

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Run operator test.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -b, --bin <PATH>           run specified program in <PATH>.
  -l, --lib <PATH>           use sprcified library in <PATH>.
  -d, --device <device_id>   run test on device.
  -c, --cpu_mask <mask>      taskset cpu mask(default: 2).
  -p, --path <path>          run test on device in specified PATH.
EOF
    exit 1;
}


TEMP=`getopt -o b:c:hl:d:p: --long bin:cpu_mask:help,lib:device:path: \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -b|--bin)
            host_bin_dir=$2
            echo "[INFO] run test in '${host_bin_dir}'" ;
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
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR]" ; exit 1 ;;
    esac
done

run_command() {
    params=" -c ${cpu_mask} -e $1 -i $2"
    if [ ${exe_on_device} == true ] ; then
        params="${params} -p ${device_dir} -d ${device}"
    fi
    if [ ${use_static_library} == true ] ; then
        params="${params} -s ${use_static_library}"
    fi
    ${driver_script_path} ${params} || exit 1
}

if [ ${exe_on_device}  == true ] ; then
    status=`adb -s ${device} shell "ls ${device_dir} && echo 'success'" | tail -n 1`
    if [ "${status}" == "success" ] ; then
        if [ ${use_static_library} != true ] ; then
            adb -s ${device} push ${host_lib_dir}/libblas-enhance.so ${device_dir}
            adb -s ${device} push ${host_lib_dir}/libtensor_computing.so ${device_dir}
        fi
    else
        adb -s ${device} shell "mkdir ${device_dir}"
    fi
fi


# FP32 & FP16 operator test
# blas-enhance
run_command ${host_bin_dir}/test_mmm ${script_dir}/params/mmm.csv
run_command ${host_bin_dir}/test_mvm ${script_dir}/params/mvm.csv

# tensor_computing
run_command ${host_bin_dir}/test_activation ${script_dir}/params/activation.csv
run_command ${host_bin_dir}/test_attention ${script_dir}/params/attention.csv
run_command ${host_bin_dir}/test_axis_mean ${script_dir}/params/axis_mean.csv
run_command ${host_bin_dir}/test_clip ${script_dir}/params/clip.csv
run_command ${host_bin_dir}/test_concat ${script_dir}/params/concat.csv
run_command ${host_bin_dir}/test_convolution ${script_dir}/params/convolution.csv
run_command ${host_bin_dir}/test_convolution ${script_dir}/params/alexnet_convolution.csv
run_command ${host_bin_dir}/test_convolution ${script_dir}/params/googlenet_convolution.csv
run_command ${host_bin_dir}/test_convolution ${script_dir}/params/resnet50_convolution.csv
run_command ${host_bin_dir}/test_deconvolution ${script_dir}/params/deconvolution.csv
run_command ${host_bin_dir}/test_depthwise_convolution ${script_dir}/params/mobilenetv1_depthwise_convolution.csv
run_command ${host_bin_dir}/test_depthwise_convolution ${script_dir}/params/mobilenetv2_depthwise_convolution.csv
run_command ${host_bin_dir}/test_depthwise_convolution ${script_dir}/params/mobilenetv3_depthwise_convolution.csv
run_command ${host_bin_dir}/test_dilated_convolution ${script_dir}/params/dilated_convolution.csv
run_command ${host_bin_dir}/test_eltwise ${script_dir}/params/eltwise.csv
run_command ${host_bin_dir}/test_fully_connected ${script_dir}/params/lenet_fully_connected.csv
run_command ${host_bin_dir}/test_lstm ${script_dir}/params/lstm.csv
run_command ${host_bin_dir}/test_multiply ${script_dir}/params/multiply.csv
run_command ${host_bin_dir}/test_pooling ${script_dir}/params/pooling.csv
run_command ${host_bin_dir}/test_reshape ${script_dir}/params/reshape.csv
run_command ${host_bin_dir}/test_softmax ${script_dir}/params/softmax.csv
run_command ${host_bin_dir}/test_split ${script_dir}/params/split.csv
run_command ${host_bin_dir}/test_slice ${script_dir}/params/slice.csv
run_command ${host_bin_dir}/test_scale ${script_dir}/params/scale.csv
run_command ${host_bin_dir}/test_transpose ${script_dir}/params/transpose.csv

# INT8 operator test
# blas-enhance
run_command ${host_bin_dir}/test_mmm_int8 ${script_dir}/params/mmm.csv
run_command ${host_bin_dir}/test_mvm_int8 ${script_dir}/params/mvm.csv

# tensor_computing
run_command ${host_bin_dir}/test_concat_int8 ${script_dir}/params/concat.csv
run_command ${host_bin_dir}/test_pooling_int8 ${script_dir}/params/pooling.csv
run_command ${host_bin_dir}/test_convolution_int8 ${script_dir}/params/alexnet_convolution.csv
run_command ${host_bin_dir}/test_convolution_int8 ${script_dir}/params/googlenet_convolution.csv
run_command ${host_bin_dir}/test_convolution_int8 ${script_dir}/params/resnet50_convolution.csv
run_command ${host_bin_dir}/test_depthwise_convolution_int8 ${script_dir}/params/mobilenetv1_depthwise_convolution.csv
run_command ${host_bin_dir}/test_depthwise_convolution_int8 ${script_dir}/params/mobilenetv2_depthwise_convolution.csv
run_command ${host_bin_dir}/test_depthwise_convolution_int8 ${script_dir}/params/mobilenetv3_depthwise_convolution.csv

# BNN operator test
run_command ${host_bin_dir}/test_convolution_bnn ${script_dir}/params/bnn_convolution.csv


if [ ${exe_on_device}  == true ] ; then
    if [ ${use_static_library} != true ] ; then
        adb -s ${device} shell "rm -rf ${device_dir}/libblas-enhance.so"
        adb -s ${device} shell "rm -rf ${device_dir}/libtensor_computing.so"
    fi
fi
