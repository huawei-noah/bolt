#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

host_bin_dir=""
host_lib_dir=""
excute_on_device=false
use_static_library=true
memory_reuse=true
remove=true
device=""
cpu_mask="2"
device_dir=""
model_zoo_dir=""

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Run model_tools test.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -b, --bin <PATH>           run specified program in <PATH>.
  -l, --lib <PATH>           use dynamic library in <PATH>.
  -d, --device <device_id>   run test on device.
  -c, --cpu_mask <mask>      taskset cpu mask(default: 2).
  -p, --path <PATH>          run test on device in specified <PATH>.
  -m, --model_zoo <PATH>     use prepared models in model_zoo(<PATH>/[caffe|onnx|tflite]_models)
  -r, --remove               remove device tmp directory or not
EOF
    exit 1;
}

TEMP=`getopt -o b:c:hl:d:p:r:m: --long bin:cpu_mask:help,lib:device:path:remove:model_zoo \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -b|--bin)
            host_bin_dir=$2
            echo "[INFO] run test in ${host_bin_dir}" ;
            shift 2 ;;
        -c|--cpu_mask)
            cpu_mask=$2
            echo "[INFO] CPU mask ${cpu_mask}" ;
            shift 2 ;;
        -l|--lib)
            host_lib_dir=$2
            use_static_library=false
            echo "[INFO] use library in ${host_lib_dir}" ;
            shift 2 ;;
        -d|--device)
            device=$2
            exe_on_device=true
            echo "[INFO] test on device ${device}" ;
            shift 2 ;;
        -m|--model_zoo)
            model_zoo_dir=$2
            echo "[INFO] use model_zoo ${model_zoo_dir}" ;
            shift 2 ;;
        -p|--path)
            device_dir=$2
            echo "[INFO] test on device directory ${device_dir}" ;
            shift 2 ;;
        -r|--remove)
            remove=$2
            echo "[INFO] clear tmp directory ${remove}" ;
            shift 2;;
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] $1" ; exit 1 ;;
    esac
done

run_command() {
    params=$*

    prefix="cd ${device_dir}/tmp"
    if [[ ${memory_reuse} == true ]] ; then
        prefix="$prefix && export BOLT_MEMORY_REUSE_OPTIMIZATION=ON"
    else
        prefix="$prefix && export BOLT_MEMORY_REUSE_OPTIMIZATION=OFF"
    fi
    if [[ ${exe_on_device} == true ]] ; then
        if [[ ${use_static_library} == true ]] ; then
            adb -s ${device} shell "$prefix && taskset ${cpu_mask} ./${params} || echo '[FAILURE]'" &> status.txt
        else
            adb -s ${device} shell "$prefix && export LD_LIBRARY_PATH=. && taskset ${cpu_mask} ./${params} || echo '[FAILURE]'" &> status.txt
        fi
    else
        if [[ ${use_static_library} == true ]] ; then
             $prefix && taskset ${cpu_mask} ${host_bin_dir}/${params} || echo '[FAILURE]' &> status.txt
        else
            export LD_LIBRARY_PATH=${host_lib_dir}:${LD_LIBRARY_PATH} && $prefix && taskset ${cpu_mask} ${host_bin_dir}/${params} || echo '[FAILURE]' &> status.txt
        fi
    fi
    cat status.txt || exit 1
    if [ `grep -c "\[FAILURE\]" status.txt` -ne '0' ] ; then
        exit 1
    fi
    rm status.txt
}

if [[ ${exe_on_device}  == true ]] ; then
    adb -s ${device} shell "mkdir ${device_dir}"
    adb -s ${device} shell "rm -rf ${device_dir}/tmp"
    adb -s ${device} shell "mkdir ${device_dir}/tmp"
    adb -s ${device} shell "cp -r ${model_zoo_dir}/* ${device_dir}/tmp/"
    adb -s ${device} shell "find ${device_dir}/tmp -name \"*\.bolt\" | xargs rm -rf"
    if [[ ${use_static_library} != true ]] ; then
        adb -s ${device} push ${host_lib_dir}/libuni.so ${device_dir}/tmp > /dev/null || exit 1
        adb -s ${device} push ${host_lib_dir}/libmodel_tools.so ${device_dir}/tmp > /dev/null || exit 1
        adb -s ${device} push ${host_lib_dir}/libmodel_tools_caffe.so ${device_dir}/tmp > /dev/null || exit 1
        adb -s ${device} push ${host_lib_dir}/libmodel_tools_onnx.so ${device_dir}/tmp > /dev/null || exit 1
        adb -s ${device} push ${host_lib_dir}/libmodel_tools_tflite.so ${device_dir}/tmp > /dev/null || exit 1
        bash ${script_dir}/../scripts/push_third_party.sh -l ${script_dir}/../third_party/arm_llvm -d ${device} -p  ${device_dir}/tmp -c arm_llvm
    fi
    adb -s ${device} push ${host_bin_dir}/X2bolt  ${device_dir}/tmp  > /dev/null || exit 1
else
    mkdir ${host_bin_dir}/tmp
    cp -r ${model_zoo_dir}/* ${host_bin_dir}/tmp/
fi

# caffe model
# INT8
run_command X2bolt -d caffe_models/squeezenet -m squeezenet -i INT8_Q
run_command X2bolt -d caffe_models/tinybert384 -m tinybert384 -i INT8_Q
run_command X2bolt -d caffe_models/tinybert -m tinybert -i INT8_Q
# FP16
run_command X2bolt -d caffe_models/mobilenet_v1 -m mobilenet_v1 -i FP16
run_command X2bolt -d caffe_models/mobilenet_v2 -m mobilenet_v2 -i FP16
run_command X2bolt -d caffe_models/mobilenet_v3 -m mobilenet_v3 -i FP16
run_command X2bolt -d caffe_models/resnet50 -m resnet50 -i FP16
run_command X2bolt -d caffe_models/squeezenet -m squeezenet -i FP16
run_command X2bolt -d caffe_models/fingerprint_resnet18 -m fingerprint_resnet18 -i FP16
run_command X2bolt -d caffe_models/tinybert384 -m tinybert384 -i FP16
run_command X2bolt -d caffe_models/tinybert -m tinybert -i FP16
run_command X2bolt -d caffe_models/tinybert_disambiguate -m tinybert_disambiguate -i FP16
run_command X2bolt -d caffe_models/nmt -m nmt FP16
run_command X2bolt -d caffe_models/nmt_tsc_encoder -m nmt_tsc_encoder -i FP16
run_command X2bolt -d caffe_models/nmt_tsc_decoder -m nmt_tsc_decoder -i FP16
run_command X2bolt -d caffe_models/tts_encoder_decoder -m tts_encoder_decoder -i FP16
run_command X2bolt -d caffe_models/asr_rnnt -m asr_rnnt -i FP16
run_command X2bolt -d caffe_models/tts_postnet -m tts_postnet -i FP16
# FP32 
run_command X2bolt -d caffe_models/mobilenet_v1 -m mobilenet_v1 -i FP32
run_command X2bolt -d caffe_models/mobilenet_v2 -m mobilenet_v2 -i FP32
run_command X2bolt -d caffe_models/mobilenet_v3 -m mobilenet_v3 -i FP32
run_command X2bolt -d caffe_models/resnet50 -m resnet50 -i FP32
run_command X2bolt -d caffe_models/squeezenet -m squeezenet -i FP32
run_command X2bolt -d caffe_models/fingerprint_resnet18 -m fingerprint_resnet18 -i FP32
run_command X2bolt -d caffe_models/tinybert384 -m tinybert384 -i FP32
run_command X2bolt -d caffe_models/tinybert -m tinybert -i FP32
run_command X2bolt -d caffe_models/tinybert_disambiguate -m tinybert_disambiguate -i FP32
run_command X2bolt -d caffe_models/nmt -m nmt -i FP32
run_command X2bolt -d caffe_models/nmt_tsc_encoder -m nmt_tsc_encoder -i FP32
run_command X2bolt -d caffe_models/nmt_tsc_decoder -m nmt_tsc_decoder -i FP32
run_command X2bolt -d caffe_models/tts_encoder_decoder -m tts_encoder_decoder -i FP32
run_command X2bolt -d caffe_models/asr_rnnt -m asr_rnnt -i FP32
run_command X2bolt -d caffe_models/tts_postnet -m tts_postnet -i FP32

# onnx model
# BNN
run_command X2bolt -d onnx_models/birealnet18 -m birealnet18 -i FP16
run_command X2bolt -d onnx_models/birealnet18 -m birealnet18 -i FP32
# FP16
run_command X2bolt -d onnx_models/tts_melgan_vocoder -m tts_melgan_vocoder -i FP16
# FP32
run_command X2bolt -d onnx_models/tts_melgan_vocoder -m tts_melgan_vocoder -i FP32

if [[ ${remove} == true ]] ; then
    if [[ ${exe_on_device}  == true ]] ; then
        adb -s ${device} shell rm -rf ${device_dir}/tmp
    else
        rm -rf ${host_bin_dir}/tmp
    fi
fi
