#!/bin/bash

script_name=$0
script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

source ${script_dir}/benchmark_verify.sh

BOLT_ROOT=${script_dir}/..
loops=6
phone=$1

# arm gnu
arch=arm_gnu
x2bolt_path=/data/local/tmp/CI/${arch}/tools/X2bolt
benchmark_path=/data/local/tmp/CI/${arch}/bin/benchmark
model_zoo_directory=/data/local/tmp/CI/model_zoo
#benchmark_verify ${phone} ${x2bolt_path} ${benchmark_path} ${model_zoo_directory} tflite mbmelgan FP32 CPU_AFFINITY_HIGH_PERFORMANCE ${loops} ''

# x86 gnu
arch=x86_gnu
x2bolt_path=${BOLT_ROOT}/install_${arch}/tools/X2bolt
benchmark_path=${BOLT_ROOT}/install_${arch}/examples/benchmark
model_zoo_directory=/data/bolt/model_zoo
benchmark_verify host ${x2bolt_path} ${benchmark_path} ${model_zoo_directory} tflite mbmelgan FP32 CPU_AFFINITY_HIGH_PERFORMANCE ${loops} '\-0.295808 0.563926 1.235842'
