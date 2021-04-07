#!/bin/bash

script_name=$0
script_dir=$(cd `dirname $0` && pwd)
bolt_root=${script_dir}/../../..

device=""
arch="linux-aarch64"
device_dir=/data/local/tmp/CI/test
model_zoo_dir=/data/local/tmp/CI/model_zoo

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Run Flow ASR Example.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -d, --device               which device to run example.
  -a, --arch <android-aarch64|linux-aarch64>  use to set device architecture(default: linux-aarch64).
  -p, --path                 device test directory.
EOF
    exit 1;
}

TEMP=`getopt -o d:a:p:m:h --long device:arch:path:model_zoo:help, \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -d|--device)
            device=$2
            echo "[INFO] run on '${device}'" ;
            shift 2 ;;
        -a|--arch)
            arch=$2
            echo "[INFO] device architecture ${arch}" ;
            shift 2 ;;
        -p|--path)
            device_dir=$2
            echo "[INFO] run in '${device_dir}'" ;
            shift 2 ;;
        -m|--model_zoo)
            model_zoo_dir=$2
            echo "[INFO] use model_zoo ${model_zoo_dir}" ;
            shift 2 ;;
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR]" ; exit 1 ;;
    esac
done

echo "[WARNING] Please make sure ${model_zoo_dir} is valid to find all inference models"
echo "[WARNING] Please make sure to use models in ${model_zoo_dir} in ${script_dir}/*.prototxt configure files"
echo "[WARNING] Please make sure you have modified ${script_dir}/pinyin2hanzi_flow.prototxt to find pinyin_lm_embedding.bin file"

adb -s $device shell "mkdir ${device_dir}"
adb -s $device push ${script_dir}/encoder_flow.prototxt $device_dir > /dev/null
adb -s $device push ${script_dir}/prediction_flow.prototxt $device_dir > /dev/null
adb -s $device push ${script_dir}/joint_flow.prototxt $device_dir > /dev/null
adb -s $device push ${script_dir}/pinyin2hanzi_flow.prototxt $device_dir > /dev/null
adb -s $device push ${script_dir}/example.wav $device_dir > /dev/null
adb -s $device push ${script_dir}/asr_labels.txt $device_dir > /dev/null
adb -s $device push ${script_dir}/pinyin_lm_embedding.bin $device_dir > /dev/null
adb -s $device push ${bolt_root}/install_${arch}/examples/flow_asr $device_dir > /dev/null
adb -s $device push ${bolt_root}/install_${arch}/tools/X2bolt $device_dir > /dev/null
adb -s $device shell "mkdir ${device_dir}/lib"
for file in `ls ${bolt_root}/install_${arch}/lib/*.so`
do
    adb -s ${device} push ${file} ${device_dir}/lib > /dev/null
done

# prepare inference models
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && ${device_dir}/X2bolt -d ${model_zoo_dir}/caffe_models/asr_convolution_transformer_encoder -m asr_convolution_transformer_encoder -i FP32"
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && ${device_dir}/X2bolt -d ${model_zoo_dir}/caffe_models/asr_convolution_transformer_prediction_net -m asr_convolution_transformer_prediction_net -i FP32"
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && ${device_dir}/X2bolt -d ${model_zoo_dir}/caffe_models/asr_convolution_transformer_joint_net -m asr_convolution_transformer_joint_net -i FP32"
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && ${device_dir}/X2bolt -d ${model_zoo_dir}/tflite_models/cnn_pinyin_lm_b7h512e4_cn_en_20200518_cloud_fp32 -m cnn_pinyin_lm_b7h512e4_cn_en_20200518_cloud_fp32 -i FP32"

# inference
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && cd ${device_dir} && ./flow_asr ./encoder_flow.prototxt ./prediction_flow.prototxt ./joint_flow.prototxt pinyin2hanzi_flow.prototxt asr_labels.txt example.wav" | tee tmp.txt

# clean work directory
adb -s $device shell "rm -rf ${device_dir}"

check=$(grep -I "\[RESULT\] hanzi: 打电话给杜娟" tmp.txt)
rm tmp.txt
if [[ ${check} < 1 ]]
then
    exit 1
fi
