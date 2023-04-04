#!/bin/bash

script_dir=$(cd `dirname $0` && pwd)
BOLT_ROOT=${script_dir}/..

platform=$1
kit_flow=$2
project_dir=""

# inference demos
demos=("ImageClassification" "Semantics" "ReadingComprehension")
xdemos=("SimpleImageClassification" "Semantics" "ReadingComprehension")
if [[ ${CXX} =~ android ]]; then
    for((i=0; i<${#demos[@]}; i++)) do
        demo=${demos[$i]};
        xdemo=${xdemos[$i]};
        echo "[INFO] setup kit ${xdemo} ..."
        project_dir="${BOLT_ROOT}/kit/Android/${xdemo}/app/src/main"
        mkdir -p ${project_dir}/assets
        cp ${BOLT_ROOT}/kit/assets/${demo}/* ${project_dir}/assets/ || exit 1
        lib_dir=${project_dir}/jniLibs
        mkdir -p ${lib_dir}
        clang_path=`which ${CXX}`
        clang_dir=$(dirname ${clang_path})
        if [[ ${CXX} =~ aarch64 ]]; then
            lib_dir=${lib_dir}/arm64-v8a
            cxx_shared_path=${clang_dir}/../sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
        else
            lib_dir=${lib_dir}/armeabi-v7a
            cxx_shared_path=${clang_dir}/../sysroot/usr/lib/arm-linux-androideabi/libc++_shared.so
        fi
        mkdir -p ${lib_dir}
        cp ${BOLT_ROOT}/install_${platform}/lib/libbolt.so ${lib_dir}/ || exit 1
        cp ${BOLT_ROOT}/install_${platform}/lib/libbolt_jni.so ${lib_dir}/ || exit 1
        if [[ -f ${cxx_shared_path} ]]; then
            cp ${cxx_shared_path} ${lib_dir}/ || exit 1
        fi
        cp -r ${BOLT_ROOT}/install_${platform}/include/java/* ${project_dir}/java/
    done
fi

demos=("ImageClassification")
if [[ ${CXX} =~ darwin ]]; then
    for((i=0; i<${#demos[@]}; i++)) do
        demo=${demos[$i]};
        xdemo="Simple${demo}"
        echo "[INFO] setup kit ${xdemo} ..."
        project_dir="${BOLT_ROOT}/kit/iOS/${xdemo}/${xdemo}/libbolt"
        mkdir -p ${project_dir}
        cp ${BOLT_ROOT}/kit/assets/${demo}/* ${project_dir}/ || exit 1
        cp ${BOLT_ROOT}/install_${platform}/lib/libbolt.a  ${project_dir}/ || exit 1
        mkdir -p ${project_dir}/headers
        cp ${BOLT_ROOT}/inference/engine/api/c/bolt.h ${project_dir}/headers || exit 1
    done
fi

# flow demos
if [[ ${kit_flow} != true ]]; then
    echo "[INFO] setup kit end."
    exit 0
fi
demos=("ImageClassification" "CameraEnlarge" "ChineseSpeechRecognition" "FaceDetection")
for((i=0; i<${#demos[@]}; i++)) do
    demo=${demos[$i]};
    if [[ ${CXX} =~ android ]]; then
        project_dir="${BOLT_ROOT}/kit/Android/${demo}/app/src/main/cpp/libbolt"
        mkdir -p ${project_dir}
        mkdir -p ${BOLT_ROOT}/kit/Android/${demo}/app/src/main/assets
        cp ${BOLT_ROOT}/kit/assets/${demo}/* ${BOLT_ROOT}/kit/Android/${demo}/app/src/main/assets/ || exit 1
        kit_flags_h=${BOLT_ROOT}/kit/assets/headers/android_kit_flags.h
    fi
    if [[ ${CXX} =~ darwin ]]; then
        project_dir="${BOLT_ROOT}/kit/iOS/${demo}/${demo}/libbolt"
        mkdir -p ${project_dir}
        cp ${BOLT_ROOT}/kit/assets/${demo}/* ${project_dir}/ || exit 1
        kit_flags_h=${BOLT_ROOT}/kit/assets/headers/ios_kit_flags.h
    fi

    if [[ ${project_dir} == "" ]]; then
        exit 0
    fi

    echo "[INFO] setup kit ${demo} ..."

    mkdir -p ${project_dir}/headers
    cp ${kit_flags_h} ${project_dir}/headers/kit_flags.h || exit 1
    cp ${BOLT_ROOT}/install_${platform}/lib/libbolt.a ${project_dir}/ || exit 1
    cp -r ${BOLT_ROOT}/common/memory/include/* ${project_dir}/headers/ || exit 1
    cp -r ${BOLT_ROOT}/common/uni/include/* ${project_dir}/headers/ || exit 1
    cp -r ${BOLT_ROOT}/common/model_spec/include/* ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/engine/include/cnn.h ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/engine/include/memory_tracker.hpp ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/engine/include/model.hpp ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/engine/include/operator.hpp ${project_dir}/headers/ || exit 1
    if [[ "${demo}" == "FaceDetection" ]]; then
        cp ${BOLT_ROOT}/inference/examples/ultra_face/ultra_face.h ${project_dir}/headers/ || exit 1
        continue;
    fi
    cp ${BOLT_ROOT}/install_${platform}/lib/libflow.a ${project_dir}/ || exit 1
    cp ${Protobuf_ROOT}/lib/libprotobuf.a ${project_dir}/ || exit 1
    cp ${BOLT_ROOT}/inference/flow/include/flow.h ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/flow/include/flow_function_factory.h ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/flow/include/node.h ${project_dir}/headers/ || exit 1
    cp ${BOLT_ROOT}/inference/flow/include/flow.pb.h ${project_dir}/headers/ || exit 1
    cp -r ${Protobuf_ROOT}/include/* ${project_dir}/headers/ || exit 1
    if [[ "${demo}" == "ChineseSpeechRecognition" ]]; then
        if [[ -d "${FFTS_ROOT}" ]]; then
            cp ${FFTS_ROOT}/lib/libffts.a ${project_dir}/ || exit 1
            cp -r ${FFTS_ROOT}/include/ffts ${project_dir}/headers/ || exit 1
        else
            echo "[WARNING] incomplete kit demo(${demo}), because of lack of ffts library. If you want to use ${demo}, please rebuild bolt with '--example --flow' options."
        fi
        cp ${BOLT_ROOT}/inference/examples/automatic_speech_recognition/audio_feature.* ${project_dir}/headers/ || exit 1
        cp ${BOLT_ROOT}/inference/examples/automatic_speech_recognition/flow_asr.h ${project_dir}/headers/ || exit 1
        cp ${BOLT_ROOT}/inference/examples/automatic_speech_recognition/*.prototxt ${BOLT_ROOT}/kit/Android/${demo}/app/src/main/assets/ || exit 1
        cp ${BOLT_ROOT}/inference/examples/automatic_speech_recognition/pinyin_lm_embedding.bin ${BOLT_ROOT}/kit/Android/${demo}/app/src/main/assets/ || exit 1
        cp ${BOLT_ROOT}/inference/examples/automatic_speech_recognition/asr_labels.txt ${BOLT_ROOT}/kit/Android/${demo}/app/src/main/assets/ || exit 1
    fi
done
echo "[INFO] setup kit end."
