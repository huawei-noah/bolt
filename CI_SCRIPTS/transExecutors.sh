#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
bolt_dir=${script_dir}/..
compiler=$1
device_dir=/data/local/tmp/CI/${compiler}

echo "[INFO] compiler ${compiler}"

upload_program() {
    host_dir=$1
    device=$2
    device_dir=$3

    adb -s ${device} shell "rm -rf ${device_dir}"
    adb -s ${device} shell "mkdir ${device_dir}"
    adb -s ${device} shell "mkdir ${device_dir}/bin ${device_dir}/lib"
    for file in `ls ${host_dir}/examples/*`
    do
        adb -s ${device} push ${file} ${device_dir}/bin > /dev/null || exit 1
    done
    for file in `ls ${host_dir}/lib/*.so`
    do
        adb -s ${device} push ${file} ${device_dir}/lib > /dev/null || exit 1
    done
    adb -s ${device} push ${host_dir}/tools/X2bolt ${device_dir}/bin > /dev/null || exit 1
    adb -s ${device} push ${host_dir}/tools/post_training_quantization ${device_dir}/bin > /dev/null || exit 1
    if [[ "${compiler}" == "arm_llvm" ]] || [[ "${compiler}" == "arm_ndkv7" ]]; then
        bash ${script_dir}/../scripts/push_third_party.sh -l ${script_dir}/../third_party/${compiler} -d ${device} -p  ${device_dir}/lib -c ${compiler} || exit 1
    fi
}

# Kirin 810
upload_program ${bolt_dir}/install_${compiler} E5B0119506000260 ${device_dir}

# Kirin 990
upload_program ${bolt_dir}/install_${compiler} GCL5T19822000030 ${device_dir}
