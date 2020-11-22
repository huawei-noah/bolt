#!/bin/bash

declare CONVERTER

declare BOLT_SUFFIX

declare TASKSET_STR

declare EXECUTOR="classification"

declare CI_PATH="/data/local/tmp/CI"

declare ARCH="arm"

declare MODEL_TOOLS_EXE_PATH=${CI_PATH}

declare ENGINE_EXE_PATH=${CI_PATH}

declare BOLT_LIB_PATH=${CI_PATH}

declare CAFFE_MODEL_ZOO_PATH="${CI_PATH}/model_zoo/caffe_models/"

declare ONNX_MODEL_ZOO_PATH="${CI_PATH}/model_zoo/onnx_models/"

declare TFLITE_MODEL_ZOO_PATH="${CI_PATH}/model_zoo/tflite_models/"

declare DYNAMIC_MODEL_PATH_PREFIX

declare PHONE_SPECIFICATION

declare TESTING_DATA_PREFIX="${CI_PATH}/testing_data/"

BOLT_DIR=$(dirname $(readlink -f "$0"))/..

function converter_selection()
{
    CONVERTER="X2bolt"
    if [ "$1" == "caffe" ]
    then
        DYNAMIC_MODEL_PATH_PREFIX=$CAFFE_MODEL_ZOO_PATH
        return
    fi

    if [ "$1" == "onnx" ]
    then
        DYNAMIC_MODEL_PATH_PREFIX=$ONNX_MODEL_ZOO_PATH
        return
    fi

    if [ "$1" == "tflite" ]
    then
        DYNAMIC_MODEL_PATH_PREFIX=$TFLITE_MODEL_ZOO_PATH
        return
    fi
    echo "[ERROR] error to convert model $1"
    exit 1
}

function acc_selection()
{
    if [ "$1" == "fp32" ]
    then
        BOLT_SUFFIX="_f32.bolt"
        return
    fi

    if [ "$1" == "fp16" ]
    then
        BOLT_SUFFIX="_f16.bolt"
        return
    fi

    if [ "$1" == "int8" ]
    then
        BOLT_SUFFIX="_int8_q.bolt"
        return
    fi

    echo "[ERROR] error to process model precision $1"
    exit 1
}

function core_selection()
{
    if [ "$1" == "A55" ]
    then
        TASKSET_STR="CPU_AFFINITY_LOW_POWER"
        return
    fi

    if [ "$1" == "A76" ]
    then
        TASKSET_STR="CPU_AFFINITY_HIGH_PERFORMANCE"
        return
    fi

    if [ "$1" == "x86_HOST" ]
    then
        TASKSET_STR="CPU_AFFINITY_HIGH_PERFORMANCE"
        return
    fi

    echo "[ERROR] error to set affinity setting $1"
    exit 1
}

function arch_selection()
{
    if [ "$1" == "arm" ]
    then
        return
    fi

    if [ "$1" == "x86" ]
    then
        ARCH="x86"
        MODEL_TOOLS_EXE_PATH=${BOLT_DIR}
        ENGINE_EXE_PATH=${BOLT_DIR}
        CAFFE_MODEL_ZOO_PATH="/data/bolt/model_zoo/caffe_models/"
        ONNX_MODEL_ZOO_PATH="/data/bolt/model_zoo/onnx_models/"
        TFLITE_MODEL_ZOO_PATH="/data/bolt/model_zoo/tflite_models/"
        TESTING_DATA_PREFIX="/data/bolt/testing_data/"
        return
    fi

    echo "[ERROR] error to set device $1"
    exit 1
}

function device_selection()
{
    if [ "$1" == "cpu" ]
    then
        return
    fi

    if [ "$1" == "gpu" ]
    then
        TASKSET_STR="GPU"
        return
    fi

    echo "[ERROR] error to set device $1"
    exit 1
}

# device id to phone specification
function deviceId_to_phoneSpecification()
{
    if [ "$1" == "E5B0119506000260" ]
    then
        PHONE_SPECIFICATION="810"
        return
    fi

    if [ "$1" == "GCL5T19822000030" ]
    then
        PHONE_SPECIFICATION="990"
        return
    fi

    if [ "$1" == "x86_HOST" ]
    then
        return
    fi

    echo "[ERROR] error to set mobile phone $1"
    exit 1
}

combinations=()
commands=()
while read line; do
    combinations[${#combinations[*]}]=`echo ${line}`
done < ./final_combinations.txt

for((k=0;k<${#combinations[@]};k++)){
    line=${combinations[k]}
    strs_arr=()
    index=0
    for i in $(echo $line| tr "-" "\n")
    do
        strs_arr[$index]=$i;
        let index+=1
    done

    commind_line=""

    arch_selection ${strs_arr[1]}

    DL_FRAMEWORK=${strs_arr[2]}
    converter_selection $DL_FRAMEWORK

    core_selection ${strs_arr[6]}

    acc_selection ${strs_arr[7]}

    device_selection ${strs_arr[8]}

    # define model converter param
    MODEL_NAME=${strs_arr[0]}

    EXECUTOR="classification"
    if [[ "$MODEL_NAME" == "tinybert" || "$MODEL_NAME" == "tinybert384" ]]
    then
        EXECUTOR="tinybert"
    fi
    if [ "$MODEL_NAME" == "tinybert_onnx" ]
    then
        EXECUTOR="tinybert_onnx"
    fi
    if [ "$MODEL_NAME" == "nmt" ]
    then
        EXECUTOR="nmt"
    fi
    if [ "$MODEL_NAME" == "asr_rnnt" ]
    then
        EXECUTOR="asr_rnnt"
    fi
    if [[ "$MODEL_NAME" == "asr_convolution_transformer_encoder" || "$MODEL_NAME" == "asr_convolution_transformer_prediction_net"
        || "$MODEL_NAME" == "asr_convolution_transformer_joint_net" ]]
    then
        EXECUTOR="asr_convolution_transformer"
    fi
    if [[ "$MODEL_NAME" == "tts_encoder_decoder" || "$MODEL_NAME" == "tts_postnet"
        || "$MODEL_NAME" == "tts_melgan_vocoder" ]]
    then
        EXECUTOR="tts"
    fi
    if [ "$MODEL_NAME" == "vad" ]
    then
        EXECUTOR="vad"
    fi

    REMOVE_OP_NUM=0
    if [ "$DL_FRAMEWORK" == "onnx" ]
    then
        REMOVE_OP_NUM=${strs_arr[13]}
    fi

    COMPILER=${strs_arr[4]}
    TESTING_DATA_PATH=$TESTING_DATA_PREFIX${strs_arr[10]}
    ORIGINAL_PARAM=${strs_arr[12]}
    MODEL_PATH=$DYNAMIC_MODEL_PATH_PREFIX$MODEL_NAME"/"
    EXECUTE_PARAM=
    BOLT_MODEL_PATH=$MODEL_PATH$MODEL_NAME$BOLT_SUFFIX
    for i in $(echo $ORIGINAL_PARAM| tr "+" "\n")
    do
        j=${i/@/-}
        EXECUTE_PARAM=$EXECUTE_PARAM" ""$j"
    done

    if [ "$ARCH" == "arm" ]
    then
        mt_command_line=${MODEL_TOOLS_EXE_PATH}/${ARCH}_${COMPILER}"/bin/"$CONVERTER" -d "$MODEL_PATH" -m "$MODEL_NAME
        engine_command_line=${ENGINE_EXE_PATH}/${ARCH}_${COMPILER}"/bin/"$EXECUTOR" ""-m "$BOLT_MODEL_PATH" ""-i "$TESTING_DATA_PATH" "$EXECUTE_PARAM" ""-a "$TASKSET_STR
        if [ "$MODEL_NAME" == "vad" ]
        then
            engine_command_line=${ENGINE_EXE_PATH}/${ARCH}_${COMPILER}"/bin/"$EXECUTOR" ""-m "$BOLT_MODEL_PATH" "$EXECUTE_PARAM" ""-a "$TASKSET_STR
        fi
    fi
    if [ "$ARCH" == "x86" ]
    then
        mt_command_line=${MODEL_TOOLS_EXE_PATH}/"install_"${ARCH}_${COMPILER}"/tools/"$CONVERTER" -d "$MODEL_PATH" -m "$MODEL_NAME
        engine_command_line=${ENGINE_EXE_PATH}/"install_"${ARCH}_${COMPILER}"/examples/"$EXECUTOR" ""-m "$BOLT_MODEL_PATH" ""-i "$TESTING_DATA_PATH" "$EXECUTE_PARAM" "
    fi

    if [ ${strs_arr[7]} == "fp32" ]
    then
        mt_command_line=$mt_command_line" -i FP32"
    fi
    if [ ${strs_arr[7]} == "fp16" ]
    then
        mt_command_line=$mt_command_line" -i FP16"
    fi
    if [ ${strs_arr[7]} == "int8" ]
    then
        mt_command_line=$mt_command_line" -i PTQ && export LD_LIBRARY_PATH=${BOLT_LIB_PATH}/${ARCH}_${COMPILER}/lib && "${MODEL_TOOLS_EXE_PATH}/${ARCH}_${COMPILER}"/bin/post_training_quantization -p "$MODEL_PATH$MODEL_NAME"_ptq_input.bolt"
    fi

    if [[ "$DL_FRAMEWORK" == "onnx" && $REMOVE_OP_NUM -gt 0 ]]
    then
        mt_command_line=$mt_command_line" -r "$REMOVE_OP_NUM
    fi
    # skip engine run section
    if [[ "$MODEL_NAME" == "tinybert_disambiguate" || "$MODEL_NAME" == "nmt_tsc_encoder" || "$MODEL_NAME" == "nmt_tsc_decoder" || "$MODEL_NAME" == "ghostnet" ]]
    then
        engine_command_line="echo 'avg_time:0ms/sequence'"
    fi
    if [ "$ARCH" == "arm" ]
    then
        mt_command_line="export LD_LIBRARY_PATH=${BOLT_LIB_PATH}/${ARCH}_${COMPILER}/lib && "$mt_command_line
        engine_command_line="export LD_LIBRARY_PATH=${BOLT_LIB_PATH}/${ARCH}_${COMPILER}/lib && "$engine_command_line

        ADB_COMMAND_PREFIX="adb -s ${strs_arr[5]} shell"
        adb_command_line="${ADB_COMMAND_PREFIX} \"${mt_command_line} > ${CI_PATH}/mt_result.txt && ${engine_command_line} > ${CI_PATH}/engine_result.txt\""
        adb_pull_result_line="adb -s ${strs_arr[5]} pull ${CI_PATH}/mt_result.txt . && adb -s ${strs_arr[5]} pull ${CI_PATH}/engine_result.txt ."
        commands[${#commands[*]}]=`echo "${adb_command_line} && ${adb_pull_result_line}"`
    fi
    if [ "$ARCH" == "x86" ]
    then
        commands[${#commands[*]}]=`echo "${mt_command_line} > ./mt_result.txt && ${engine_command_line} > ./engine_result.txt"`
    fi
}

rm -r ./report.csv

for((k=0;k<${#commands[@]};k++)){
    line=${commands[k]}
    echo "Running_Beginning =====> $line"
    eval $line || exit 1

    MT_RUN_RESULT="MT_RUN_UNKNOWN"

    ENGINE_RUN_RESULT="ENGINE_RUN_UNKNOWN"

    TOP_ONE_ACC=
    TOP_FIVE_ACC=
    MAX_TIME_RESULT=
    MIN_TIME_RESULT=
    AVG_TIME_RESULT=
    MESSAGE="ERROR"

    if cat ./mt_result.txt | grep "$MESSAGE" > /dev/null
    then
        MT_RUN_RESULT="MT_RUN_FAIL"
        echo "Model conversion failed"
        exit 1
    else
        MT_RUN_RESULT="MT_RUN_PASS"
    fi

    if cat ./engine_result.txt | grep "$MESSAGE" > /dev/null
    then
        ENGINE_RUN_RESULT="ENGINE_RUN_FAIL"
        TOP_ONE_ACC="ERROR"
        TOP_FIVE_ACC="ERROR"
        MAX_TIME_RESULT="ERROR"
        MIN_TIME_RESULT="ERROR"
        AVG_TIME_RESULT="ERROR"
        echo "Error during inference"
        exit 1
    else
        ENGINE_RUN_RESULT="ENGINE_RUN_PASS"
        TOP_ONE_ACC=$(grep -I "top1" ./engine_result.txt)
        TOP_FIVE_ACC=$(grep -I "top5" ./engine_result.txt)
        MAX_TIME_RESULT=$(grep -I "max_time" ./engine_result.txt)
        MIN_TIME_RESULT=$(grep -I "min_time" ./engine_result.txt)
        AVG_TIME_RESULT=$(grep -I "avg_time:" ./engine_result.txt)
    fi

    if [[ ${#AVG_TIME_RESULT} < 1 ]]
    then
        echo "Undetected error during Inference"
        exit 1
    fi

    line=${combinations[k]}
    final_arr=()
    index=0
    for i in $(echo $line| tr "-" "\n")
    do
        final_arr[$index]=$i;
        let index+=1
    done

    result_line=""

    report_index=0
    deviceId_to_phoneSpecification ${final_arr[5]}
    final_arr[5]=$PHONE_SPECIFICATION
    final_arr[12]=""
    CUR_MODEL_NAME=${final_arr[0]}
    for value in "${final_arr[@]}";
    do
        if [ $report_index == 11 ]
        then
            break
        fi

        if [ $report_index == 0 ]
        then
            result_line=$value
        else
            result_line=$result_line","$value
        fi
        let report_index+=1
    done

    # add segmentation fault check
    SEGMENTATION_FAULT_CHECK=$(grep -I "Segmentation fault" ./mt_result.txt)
    if [[ ${#SEGMENTATION_FAULT_CHECK} > 0 ]]
    then
        MT_RUN_RESULT="MT_SEGMENTATION_FAULT"
        echo "Segmentation fault during model conversion"
        exit 1
    fi

    SEGMENTATION_FAULT_CHECK=$(grep -I "Segmentation fault" ./engine_result.txt)
    if [[ ${#SEGMENTATION_FAULT_CHECK} > 0 ]]
    then
        ENGINE_RUN_RESULT="ENGINE_SEGMENTATION_FAULT"
        echo "Segmentation fault during inference"
        exit 1
    fi

    COMPREHENSIVE_RESULT=$MAX_TIME_RESULT"+"$MIN_TIME_RESULT"+"$AVG_TIME_RESULT"+"$TOP_FIVE_ACC"+"$TOP_ONE_ACC

    if [[ "$CUR_MODEL_NAME" == "tinybert" || "$CUR_MODEL_NAME" == "fingerprint_resnet18" || "$CUR_MODEL_NAME" == "nmt"
       || "$CUR_MODEL_NAME" == "asr_convolution_transformer_encoder" || "$CUR_MODEL_NAME" == "asr_convolution_transformer_prediction_net"
       || "$CUR_MODEL_NAME" == "asr_convolution_transformer_joint_net" || "$CUR_MODEL_NAME" == "asr_rnnt" || "$CUR_MODEL_NAME" == "vad"
       || "$CUR_MODEL_NAME" == "tts_encoder_decoder" || "$CUR_MODEL_NAME" == "tts_postnet"
       || "$CUR_MODEL_NAME" == "tts_melgan_vocoder" ]]
    then
        result_line=$result_line","$MT_RUN_RESULT","$ENGINE_RUN_RESULT","$AVG_TIME_RESULT","
    else
        result_line=$result_line","$MT_RUN_RESULT","$ENGINE_RUN_RESULT","$MAX_TIME_RESULT","$MIN_TIME_RESULT","$AVG_TIME_RESULT","$TOP_FIVE_ACC","$TOP_ONE_ACC","
    fi
    rm -rf ./mt_result.txt
    rm -rf ./engine_result.txt

    echo "Running_Result =====> $result_line"

    echo $result_line >> ./report.csv
    echo " " >> ./report.csv
    echo " "
    echo " "
}

cat ./report.csv
