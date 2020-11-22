#!/bin/bash

script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)

export BOLT_ROOT=${script_dir}/../../

cp ${BOLT_ROOT}/install_arm_ios/lib/libbolt.a ${script_dir}/image_classification/ImageClassificationDemo/libbolt/
cp ${BOLT_ROOT}/install_arm_ios/lib/libflow.a ${script_dir}/image_classification/ImageClassificationDemo/libbolt/
cp ${BOLT_ROOT}/third_party/arm_ios/protobuf/lib/libprotobuf.a ${script_dir}/image_classification/ImageClassificationDemo/libbolt/

rm -rf ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/memory/
cp -r ${BOLT_ROOT}/common/memory/include/ ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/memory/
rm -rf ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/uni/
cp -r ${BOLT_ROOT}/common/uni/include/ ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/uni/

rm -rf ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/engine/
mkdir ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/engine/
cp ${BOLT_ROOT}/inference/engine/include/cnn.h ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/engine/
cp ${BOLT_ROOT}/inference/engine/include/memory_tracker.hpp ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/engine/
cp ${BOLT_ROOT}/inference/engine/include/model.hpp ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/engine/
cp ${BOLT_ROOT}/inference/engine/include/operator.hpp ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/engine/
cp ${BOLT_ROOT}/inference/flow/include/flow.h ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/flow/
cp ${BOLT_ROOT}/inference/flow/include/flow_function_factory.h ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/flow/
cp ${BOLT_ROOT}/inference/flow/include/node.h ${script_dir}/image_classification/ImageClassificationDemo/libbolt/headers/flow/
