#!/bin/bash

benchmark_verify() {
    device=$1
    x2bolt_path=$2
    benchmark_path=$3
    model_zoo_directory=$4
    model_type=$5
    model_name=$6
    precision=$7
    affinity=$8
    loops=$9
    result=${10}

    model_directory=${model_zoo_directory}/${model_type}_models/${model_name}
    if [[ "${precision}" == "FP32" ]]; then
        precision_suffix="_f32"
    fi
    if [[ "${precision}" == "FP16" ]]; then
        precision_suffix="_f16"
    fi
    if [[ "${precision}" == "INT8_Q" ]]; then
        precision_suffix="_int8"
    fi
    model_convert_command="${x2bolt_path} -d ${model_directory} -m ${model_name} -i ${precision}"
    benchmark_command="${benchmark_path} -m ${model_directory}/${model_name}${precision_suffix}.bolt -a ${affinity} -l ${loops}"
    if [[ "${device}" == "host" ]]; then
        ${model_convert_command} > /dev/null && ${benchmark_command} &> engine_result.txt
    else
        adb -s ${device} shell "${model_convert_command} && ${benchmark_command}" &> engine_result.txt
    fi

    avg_time=$(grep -I "avg_time:" ./engine_result.txt)
    verify_result=$(grep -I "${result}" ./engine_result.txt)

    rm -rf engine_result.txt

    if [[ ${#verify_result} > 0 ]]
    then
        echo "${model_name} on ${device} in ${precision} precision ${avg_time}"
    else
        echo "${model_name} on ${device} in ${precision} precision fail!"
        exit 1
    fi
}
