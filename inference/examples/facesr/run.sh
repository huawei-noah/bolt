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
Run Flow FaceSR Example.

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

adb -s $device shell "mkdir ${device_dir}"
adb -s $device push ${script_dir}/flow_facesr.prototxt $device_dir > /dev/null
adb -s $device push ${bolt_root}/install_${arch}/examples/flow_facesr $device_dir > /dev/null
adb -s $device push ${bolt_root}/install_${arch}/tools/X2bolt $device_dir > /dev/null
adb -s $device shell "mkdir ${device_dir}/lib"
for file in `ls ${bolt_root}/install_${arch}/lib/*.so`
do
    adb -s ${device} push ${file} ${device_dir}/lib > /dev/null
done

# prepare inference models
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && ${device_dir}/X2bolt -d ${model_zoo_dir}/caffe_models/facesr -m facesr -i FP16"

# inference
adb -s $device shell "export LD_LIBRARY_PATH=${device_dir}/lib && cd ${device_dir} && ./flow_facesr ./flow_facesr.prototxt 1 || echo '[FAILURE]'" | tee status.txt

# clean work directory
adb -s $device shell "rm -rf ${device_dir}"

cat status.txt || exit 1
if [ `grep -c "\[FAILURE\]" status.txt` -ne '0' ] ; then
    exit 1
fi
rm status.txt
