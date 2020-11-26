#!/bin/bash

script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

export BOLT_ROOT=${script_dir}/../../

cp ${BOLT_ROOT}/install_arm_llvm/lib/libbolt.a ${script_dir}/image_classification/app/src/main/cpp/libbolt/
cp ${BOLT_ROOT}/install_arm_llvm/lib/libflow.a ${script_dir}/image_classification/app/src/main/cpp/libbolt/
cp ${BOLT_ROOT}/third_party/arm_llvm/protobuf/lib/libprotobuf.a ${script_dir}/image_classification/app/src/main/cpp/libbolt/

cp ${BOLT_ROOT}/common/memory/include/* ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/
cp ${BOLT_ROOT}/common/uni/include/* ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/

cp ${BOLT_ROOT}/inference/engine/include/cnn.h ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/
cp ${BOLT_ROOT}/inference/engine/include/memory_tracker.hpp ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/
cp ${BOLT_ROOT}/inference/engine/include/model.hpp ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/
cp ${BOLT_ROOT}/inference/engine/include/operator.hpp ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/

cp ${BOLT_ROOT}/inference/flow/include/* ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/
rm -rf ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/protobuf/
cp -r ${BOLT_ROOT}/third_party/arm_llvm/protobuf/include/ ${script_dir}/image_classification/app/src/main/cpp/libbolt/headers/protobuf/

cp ${BOLT_ROOT}/kit/assets/image_classification/* ${script_dir}/image_classification/app/src/main/assets
