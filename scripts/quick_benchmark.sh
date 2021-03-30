#!/bin/bash

script_name=$0
script_dir=$(cd `dirname $0` && pwd)
log_file="upload.log"

host_dir=""
device=""
device_dir=""
use_static_library=true
cpu_mask="40"
gpu=false

getopt --test
if [[ "$?" != "4" ]]; then
    echo -e "[ERROR] you are using BSD getopt, not GNU getopt. If you are runing on Mac, please use this command to install gnu-opt.\n    brew install gnu-getopt && brew link --force gnu-getopt"
    exit 1
fi
print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Run quick benchmark.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                    display this help and exit.
  --host_dir=<path>             run test in specified PATH.
  -d, --device <android|host>   run test on android or host.
  --device_dir=<path>           run test on device in specified path.
  -c, --cpu_mask <mask>         taskset cpu mask(default: 40).
  -g, --gpu                     run gpu test.
  --shared                      use shared library.
EOF
    exit 1;
}

TEMP=`getopt -o "hd:c:g" --long help,host_dir:,device:,device_dir:,cpu_mask:,gpu,shared, \
     -n ${script_name} -- "$@"`
if [[ ${TEMP} != *-- ]]; then
    echo "[ERROR] ${script_name} can not recognize ${TEMP##*-- }"
    echo "maybe it contains invalid character(such as Chinese)."
    exit 1
fi
if [[ $? != 0 ]]; then
    echo "[ERROR] ${script_name} terminating..." >&2
    exit 1
fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        --host_dir)
            host_dir=$2
            shift 2 ;;
        -d|--device)
            device=$2
            shift 2 ;;
        --device_dir)
            device_dir=$2
            shift 2 ;;
        -c|--cpu_mask)
            cpu_mask=$2
            shift 2 ;;
        -g|--gpu)
            gpu=true;
            shift ;;
        --shared)
            use_static_library=false;
            shift ;;
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] ${script_name} can not recognize $1" ; exit 1 ;;
    esac
done

platform=${host_dir##*install_}
platform=${platform%/*}

if [[ ${platform} =~ android-x86_64 ]]; then
    exit 0
fi
if [[ ${device} != "android" && ${device} != "host" ]]; then
    exit 0
fi

checkExe(){
    if type $1 &> /dev/null;
    then
        return 1
    else
        return 0
    fi
}

if [[ "${device}" == "android" ]] ; then
    checkExe adb 
    if [[ $? == 0 ]] ; then
        echo "[WARNING] detect not install adb tools and skip test on android device. If you want to run test, please install adb and set shell environment PATH to find it."
        exit 0
    fi
    array=($(adb devices | sed 's/\r//' | grep ".device$"))
    device=${array[0]}
    if [[ "${device}" == "" ]]; then
        echo "[WARNING] detect there is no android device. If you want to run test, please connect one android device."
        exit 0
    fi
    echo "[INFO] run quick benchmark on device ${device} in directory ${device_dir} with cpu ${cpu_mask}, platform ${platform}"
    adb -s ${device} shell "rm -rf ${device_dir}"
    adb -s ${device} shell "mkdir -p ${device_dir}" || exit 1
    if [[ ${use_static_library} != true ]] ; then
        for file in `ls ${host_dir}/lib/*.so`
        do
            adb -s ${device} push ${file} ${device_dir} > ${log_file} || exit 1
        done
    fi
    ${script_dir}/push_third_party.sh -d ${device} -p ${device_dir} --platform=${platform} || exit 1
else
    echo "[INFO] run quick benchmark on ${device} with cpu ${cpu_mask}"
    export LD_LIBRARY_PATH=${host_dir}/lib:${LD_LIBRARY_PATH}
fi

upload(){
    if [[ -f "$1" && "${device}" != "host" && "${device}" != "apple" ]] ; then
        adb -s ${device} push $1 ${device_dir} > ${log_file} || exit 1
    fi
}

run_command() {
    if [[ "${device}" == "host" ]] ; then
        ${host_dir}/tests/$@ || exit 1
    elif [[ "${device}" != "apple" ]] ; then
        adb -s ${device} shell "cd ${device_dir} && export LD_LIBRARY_PATH=./ && taskset -a ${cpu_mask} ./$@" || exit 1
    fi
}

# mmm
echo " " ; echo "--- Matrix Matrix Multiplication"
upload ${host_dir}/tests/test_mmm_int8 ${device_dir} || exit 1
upload ${host_dir}/tests/test_mmm ${device_dir} || exit 1
run_command test_mmm_int8 384 768 768
run_command test_mmm 384 768 768

# conv_ic=3
echo " " ; echo "--- Conv IC=3"
upload ${host_dir}/tests/test_convolution ${device_dir} || exit 1
run_command test_convolution 1 3 227 227 96 3 11 11 1 4 0 1 96 55 55

# conv_5x5
echo " " ; echo "--- Conv 5x5"
upload ${host_dir}/tests/test_convolution_bnn ${device_dir} || exit 1
upload ${host_dir}/tests/test_convolution_int8 ${device_dir} || exit 1
run_command test_convolution_bnn 1 96 27 27 256 96 5 5 1 2 0 1 256 13 13
run_command test_convolution_int8 1 96 27 27 256 96 5 5 1 2 0 1 256 13 13
run_command test_convolution 1 96 27 27 256 96 5 5 1 2 0 1 256 13 13

# conv_3x3
echo " " ; echo "--- Conv 3x3"
run_command test_convolution_bnn 1 128 28 28 256 128 3 3 1 1 1 1 256 28 28
run_command test_convolution_int8 1 128 28 28 256 128 3 3 1 1 1 1 256 28 28
run_command test_convolution 1 128 28 28 256 128 3 3 1 1 1 1 256 28 28

# depthwise-pointwise convolution
echo " " ; echo "--- Depthwise-Pointwise Conv"
upload ${host_dir}/tests/test_depthwise_convolution ${device_dir} || exit 1
run_command test_depthwise_convolution 1 256 28 28 256 256 3 3 1 1 1 1 256 28 28

# OCL
if [[ ${gpu}  == true ]] ; then
    upload ${host_dir}/tests/test_convolution_ocl ${device_dir} || exit 1
    upload ${host_dir}/tests/test_depthwise_convolution_ocl ${device_dir} || exit 1
    upload ${host_dir}/tests/test_fully_connected_ocl ${device_dir} || exit 1
    run_command test_convolution_ocl 64 112 112 64 5 5 1 2
    run_command test_convolution_ocl 64 112 112 64 3 3 1 1
    run_command test_depthwise_convolution_ocl 64 112 112 64 3 3 1 1
    run_command test_fully_connected_ocl 24 1 1 96 
fi

if [[ "${device}" != "host" && "${device}" != "apple" ]] ; then
    adb -s ${device} shell "rm -rf ${device_dir}" || exit 1
fi
if [[ -f "${log_file}" ]]; then
    rm ${log_file} || exit 1
fi
