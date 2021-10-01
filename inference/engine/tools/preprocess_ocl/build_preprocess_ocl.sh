#!/bin/bash

script_name=$0
script_dir=$(cd `dirname ${script_name}` && pwd)
BOLT_ROOT=${script_dir}/../../../../

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
  --target=<android-aarch64|android-armv7> target device system and hardware setting.
  -d, --directory=xxx        bolt model directory on android device.
EOF
    exit 1;
}

threads=8
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
            threads=$2
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

source ${BOLT_ROOT}/scripts/setup_compiler.sh || exit 1

#Set your preprocess_ocl program file location of host
preprocess_ocl=${BOLT_ROOT}/install_android-aarch64/tools/preprocess_ocl

current_dir=${PWD}
host_work_local=${PWD}/tmp
host_include=${host_work_local}/include
host_cpp=${host_work_local}/cpp
host_extern=${host_work_local}/extern
host_lib=${current_dir}
host_build=${host_work_local}/build
rm -rf ${host_work_local} 
mkdir ${host_work_local} ${host_include} ${host_cpp} ${host_extern} ${host_build}

#Set your work location on device, make sure it is read-write avaliable, sh will build filefolds automatically
device_work_local=/data/local/tmp/preprocess
device_algo_files=${device_work_local}/algoFiles
device_include=${device_work_local}/include
device_cpp=${device_work_local}/cpp
adb -s ${device} shell "rm -rf ${device_work_local}"
adb -s ${device} shell "mkdir ${device_work_local}"
adb -s ${device} shell "mkdir ${device_work_local}/lib"
adb -s ${device} shell "mkdir ${device_algo_files}"
adb -s ${device} shell "mkdir ${device_include}"
adb -s ${device} shell "mkdir ${device_cpp}"

adb -s ${device} push ${preprocess_ocl} ${device_work_local} > /dev/null || exit 1

echo "Running GPU preprocess on device ${device}"
adb -s ${device} shell "cd ${device_work_local} && chmod +x preprocess_ocl && export LD_LIBRARY_PATH=./lib && ./preprocess_ocl ${device_bolt_models} ${device_algo_files} ${device_include} ${device_cpp}" || exit 1
echo "Finish GPU preprocess on device ${device}"

echo "Aquire kernelBins from device ${device}"
adb -s ${device} pull ${device_include} ${host_include} > /dev/null || exit 1
adb -s ${device} pull ${device_cpp} ${host_cpp} > /dev/null || exit 1

if [[ -d ${host_include}/include ]]; then    
    mv ${host_include}/include/* ${host_include}
    rm -rf ${host_include}/include
fi

if [[ -d ${host_cpp}/cpp ]]; then    
    mv ${host_cpp}/cpp/* ${host_cpp}
    rm -rf ${host_cpp}/cpp
fi

cp ${BOLT_ROOT}/common/gcl/include/gcl_kernel_binmap.h ${host_extern}

lib_name=libkernelbin
cpp_files_name=$(ls ${host_cpp})
for p in ${cpp_files_name[@]}
do
    lib_name=${p%.*}
done

cd ${host_build}
export CXXFLAGS="${CXXFLAGS} -fstack-protector-all"
cmake ../.. ${CMAKE_OPTIONS} || exit 1
${MAKE} -j ${threads} || exit 1

allSrcs=`find . -name "*.o"` || exit 1
for file in ${allSrcs}
do
    sharedSrcs="${sharedSrcs} ${file}"
done
${CXX} ${CXXFLAGS} -shared -o ${host_lib}/lib${lib_name}.so ${sharedSrcs} \
    -L${BOLT_ROOT}/third_party/android-aarch64/opencl/lib -lOpenCL -Wl,-soname,lib${lib_name}.so || exit 1
${STRIP} ${host_lib}/lib${lib_name}.so || exit 1

adb -s ${device} shell "rm -rf ${device_work_local}"
rm -rf ${host_work_local}
cd ${current_dir}
echo "Generate ${host_lib}/lib${lib_name}.so"
