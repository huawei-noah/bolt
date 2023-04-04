#!/bin/bash

script_name=$0
script_dir=$(cd `dirname ${script_name}` && pwd)
current_dir=${PWD}
PID=$$

getopt --test
if [[ "$?" != "4" ]]; then
    echo -e "[ERROR] you are using BSD getopt, not GNU getopt. If you are runing on Mac, please use this command to install gnu-opt.\n    brew install gnu-getopt && brew link --force gnu-getopt"
    exit 1
fi
print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Build Bolt OpenCL library.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -t, --threads=8            use parallel build(default: 8).
  --device=xxx               set to use android device xxx.
  --target=<android-aarch64|android-armv7|windows-x86_64|windows-x86_64_avx2> target device system and hardware setting.
  -d, --directory=xxx        bolt model directory on android device.
EOF
    exit 1;
}

build_threads=8
target=android-aarch64
TEMP=`getopt -o "ht:d:" -al help,threads:,target:,device:,directory: -- "$@"`
if [[ ${TEMP} != *-- ]]; then
    echo "[ERROR] ${script_name} can not recognize ${TEMP##*-- }"
    echo "maybe it contains invalid character(such as Chinese)."
    exit 1
fi
if [ $? != 0 ] ; then echo "[ERROR] ${script_name} terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -h|--help)
            print_help ;
            shift ;;
        -t|--threads)
            build_threads=$2
            shift 2 ;;
        --device)
            device=$2
            shift 2 ;;
        --target)
            target=$2
            shift 2 ;;
        -d|--directory)
            device_bolt_models=$2
            shift 2 ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] ${script_name} can not recognize $1" ; exit 1 ;;
    esac
done

BOLT_ROOT=${script_dir}/../../../../
preprocess_ocl=${BOLT_ROOT}/install_${target}/tools/preprocess_ocl/preprocess_ocl
if [[ ! -f ${preprocess_ocl} ]]; then
    BOLT_ROOT=${script_dir}/../../
    preprocess_ocl=${BOLT_ROOT}/install_${target}/tools/preprocess_ocl/preprocess_ocl
    if [[ ! -f "${preprocess_ocl}" ]]; then
        if [[ -f "./preprocess_ocl" ]]; then
            BOLT_ROOT="none"
            preprocess_ocl=${PWD}/preprocess_ocl
	else
            echo "[ERROR] can not find preprocess_ocl under"
            exit 1
	fi
    fi
fi

host_work_local=${script_dir}/tmp
rm -rf ${host_work_local} 
mkdir -p ${host_work_local} || exit 1

if [[ ${target} =~ android ]]; then
    device_work_local=/data/local/tmp/bolt_${PID}
    device_algo_files=${device_work_local}/algoFiles
    device_include=${device_work_local}/include
    device_cpp=${device_work_local}/cpp
    adb -s ${device} shell "rm -rf ${device_work_local}" || exit 1
    adb -s ${device} shell "mkdir ${device_work_local}" || exit 1
    adb -s ${device} shell "mkdir ${device_work_local}/lib" || exit 1
    adb -s ${device} shell "mkdir ${device_algo_files}" || exit 1
    adb -s ${device} shell "mkdir ${device_include}" || exit 1
    adb -s ${device} shell "mkdir ${device_cpp}" || exit 1
    adb -s ${device} push ${preprocess_ocl} ${device_work_local} > /dev/null || exit 1
    
    echo "[INFO] GPU preprocess on device ${device} ..."
    adb -s ${device} shell "cd ${device_work_local} && chmod +x preprocess_ocl && export LD_LIBRARY_PATH=./lib && ./preprocess_ocl ${device_bolt_models} ${device_algo_files} ${device_include} ${device_cpp}" || exit 1
    echo "[INFO] GPU preprocess on device ${device} end"
    
    echo "[INFO] Download files from device ${device}"
    adb -s ${device} pull ${device_include} ${host_work_local}/ > /dev/null || exit 1
    adb -s ${device} pull ${device_cpp} ${host_work_local}/ > /dev/null || exit 1
    adb -s ${device} shell "rm -rf ${device_work_local}"
else
    device_work_local=${host_work_local}
    device_algo_files=${device_work_local}/algoFiles
    device_include=${device_work_local}/include
    device_cpp=${device_work_local}/cpp
    mkdir ${device_work_local}/lib || exit 1
    mkdir ${device_algo_files} || exit 1
    mkdir ${device_include} || exit 1
    mkdir ${device_cpp} || exit 1

    echo "[INFO] GPU preprocess on host..."
    ${preprocess_ocl} ${device_bolt_models} ${device_algo_files} ${device_include} ${device_cpp} || exit 1
    echo "[INFO] GPU preprocess on host end"
fi

if [[ -f "${BOLT_ROOT}/scripts/setup_compiler.sh" ]]; then
    source ${BOLT_ROOT}/scripts/setup_compiler.sh || exit 1
elif [[ -f "./setup_compiler.sh" ]]; then
    source ./setup_compiler.sh || exit 1
else
    echo "[ERROR] can not find setup_compiler.sh"
    exit 1
fi
#source ${BOLT_ROOT}/third_party/${target}.sh || exit 1
out_dir=${script_dir}
cmake_options="${cmake_options} -DCMAKE_INSTALL_PREFIX=${out_dir}"
echo "[INFO] use cmake options ${cmake_options}"
cd ${host_work_local}
rm -rf build
mkdir -p build || exit 1
cd build
cmake -G"${CMAKE_GENERATOR}" ${cmake_options} ${script_dir} || exit 1
${MAKE} -j${build_threads} || exit 1
${MAKE} install || exit 1

cd ${current_dir}
rm -rf ${host_work_local}
out=`ls lib*map* | xargs`
if [[ -f "./kernel_def.h" ]]; then
    rm -f ./kernel_def.h
fi
if [[ -f "./kernel.bin" ]]; then
    out="${out} kernel.bin"
fi
echo "[INFO] Generate ${out} in ${out_dir} directory."
