#!/bin/bash

script_name=$0
script_dir=$(cd `dirname ${script_name}` && pwd)
BOLT_ROOT=${script_dir}

target=""
build_threads="8"
converter="on"
use_serial="on"
use_neon="on"
use_fp32="on"
use_fp16="on"
use_int8="on"
use_int32="on"
clean="off"
build_test="on"
build_example="off"
use_lite="off"

find . -name "*.sh" | xargs chmod +x

source ${script_dir}/scripts/target.sh

check_getopt

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Build bolt library.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  --target=<???>             target device system and hardware setting. xxx_blank will use shell environment variables CC, CXX, CFLAGS and CXXFLAGS, e.g. linux-aarch64_blank is for ARM64 server. currently only support theses targets:
EOF
    print_targets
    cat <<EOF
  --converter=<ON|OFF>       set to use model converter(default: ON).
  --test=<ON|OFF>            set to build test(default: ON).
  --example                  set to build example(default: OFF).
  --opencv                   set to build OpenCV(default: OFF).
  --debug                    set to use debug(default: OFF).
  --lite                     set to build lite verion, library size is about 1MB, run in serial mode.(default: OFF).
  --profile                  set to print performance profiling information(default: OFF).
  --shared                   set to use shared library(default: OFF).
  --securec                  set to use Huawei secure C(default: OFF).
  --gpu                      set to use arm mali/qualcomm gpu(default: OFF).
  --openmp                   set to use OpenMP multi-threads parallel operator(default: OFF).
  --flow                     set to use flow to process pipeline data(default: OFF).
  --train                    set to use train(default: OFF).
  --serial=<ON|OFF>          set to use serial calculation(default: ON).
  --neon=<ON|OFF>            set to use arm neon calculation(default: ON when using for arm platform).
  --fp32=<ON|OFF>            set to use float32 calculation(default: ON).
  --fp16=<ON|OFF>            set to use float16 calculation on arm aarch64(default: ON on aarch64, OFF on others).
  --int8=<ON|OFF>            set to use int8 calculation on arm aarch64(default: ON on aarch64, OFF on others).
  --cpp_api                  set to use C++ API(default: OFF)
  --python_api               set to use Python API(default: OFF)
  -t, --threads=8            use parallel build(default: 8).
  --clean                    remove build and install files.
EOF
    exit 1;
}

cmake_options=""
TEMP=`getopt -o "ht:c:" -al target:,threads:,help,converter:,test:,lite,example,opencv,debug,profile,shared,gpu,openmp,flow,serial:,neon:,fp32:,fp16:,int8:,train,clean,securec,cpp_api,python_api -- "$@"`
if [[ $? != 0 ]]; then
    echo "[ERROR] ${script_name} terminating..." >&2
    exit 1
fi
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
        --target)
            target=$2
            shift 2 ;;
        -c|--converter)
            converter=$2
            shift 2 ;;
        --test)
            build_test=$2
            shift 2 ;;
        --example)
            build_example="on"
            shift ;;
        --lite)
            use_lite="on"
            shift ;;
        --opencv)
            cmake_options="${cmake_options} -DUSE_OPENCV=ON"
            shift ;;
        --debug)
            cmake_options="${cmake_options} -DUSE_DEBUG=ON"
            shift ;;
        --profile)
            cmake_options="${cmake_options} -DUSE_PROFILE=ON -DUSE_PROFILE_STATISTICS=ON"
            shift ;;
        --shared)
            cmake_options="${cmake_options} -DUSE_DYNAMIC_LIBRARY=ON"
            shift ;;
        --gpu)
            cmake_options="${cmake_options} -DUSE_GPU=ON"
            shift ;;
        --openmp)
            cmake_options="${cmake_options} -DUSE_OPENMP=ON"
            shift ;;
        --flow)
            cmake_options="${cmake_options} -DUSE_FLOW=ON -DUSE_THREAD_SAFE=ON"
            shift ;;
        --serial)
            use_serial=$2
            shift 2 ;;
        --neon)
            use_neon=$2
            shift 2 ;;
        --fp32)
            use_fp32=$2
            shift 2 ;;
        --fp16)
            use_fp16=$2
            shift 2 ;;
        --int8)
            use_int8=$2
            shift 2 ;;
        --train)
            cmake_options="${cmake_options} -DUSE_TRAINING=ON -DRAUL_CONFIG_BLAS_VENDOR=Huawei"
            shift ;;
        --securec)
            cmake_options="${cmake_options} -DUSE_SECURE_C=ON"
            shift ;;
        --cpp_api)
            cmake_options="${cmake_options} -DUSE_API_CPP=ON"
            shift ;;
        --python_api)
            cmake_options="${cmake_options} -DUSE_API_CPP=ON -DUSE_API_PYTHON=ON"
            shift ;;
        --clean)
            clean="on"
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] ${script_name} can not recognize $1" ; exit 1 ;;
    esac
done

if [[ "${target}" == "" ]]; then
    echo "[ERROR] please set target option to specify the target deployment platform."
    print_help
    exit 1
fi

if [[ "${use_lite}" == "on" || "${use_lite}" == "ON" ]]; then
    use_fp16=off
    use_int32=off
    use_general=on
    build_test=off
    build_example=off
    cmake_options="${cmake_options} -DUSE_LITE=ON"
fi
target=$(map_target ${target})
check_target ${target}

if [[ "${converter}" == "ON" || "${converter}" == "on" ]]; then
    cmake_options="${cmake_options} -DUSE_CAFFE=ON -DUSE_ONNX=ON -DUSE_TFLITE=ON -DUSE_TENSORFLOW=ON -DUSE_MINDSPORE=ON"
fi

source ${script_dir}/scripts/setup_compiler.sh || exit 1
cmake_options="${CMAKE_OPTIONS} ${cmake_options}"

platform="${target}"
if [[ "${clean}" == "on" || "${clean}" == "ON" ]]; then
    ${script_dir}/third_party/install.sh --target=${target} --threads=${build_threads} --clean || exit 1
    if [[ -d ${script_dir}/build_${platform} ]]; then
        rm -rf ${script_dir}/build_${platform} || exit 1
    fi
    if [[ -d ${script_dir}/install_${platform} ]]; then
        rm -rf ${script_dir}/install_${platform} || exit 1
    fi
    exit 0
fi

aarch64_fp16_cflags=""
aarch64_int8_cflags=""
if [[ ${CC} =~ clang ]]; then
    aarch64_fp16_cflags="-march=armv8-a+fp16"
    aarch64_int8_cflags="-march=armv8-a+fp16+dotprod"
fi
if [[ ${CC} =~ gcc ]]; then
    aarch64_fp16_cflags="-march=armv8.2-a+fp16"
    aarch64_int8_cflags="-march=armv8.2-a+fp16+dotprod"
fi

if [[ "${use_serial}" == "ON" || "${use_serial}" == "on" ]]; then
    cmake_options="${cmake_options} -DUSE_GENERAL=ON"
else
    cmake_options="${cmake_options} -DUSE_GENERAL=OFF"
fi
if [[ "${use_int32}" == "ON" || "${use_int32}" == "on" ]]; then
    cmake_options="${cmake_options} -DUSE_INT32=ON"
else
    cmake_options="${cmake_options} -DUSE_INT32=OFF"
fi
if [[ "${use_fp32}" == "ON" || "${use_fp32}" == "on" ]]; then
    cmake_options="${cmake_options} -DUSE_FP32=ON"
else
    cmake_options="${cmake_options} -DUSE_FP32=OFF"
fi
if [[ ${target} =~ aarch64 ]]; then
    if [[ "${use_neon}" == "ON" || "${use_neon}" == "on" ]]; then
        cmake_options="${cmake_options} -DUSE_NEON=ON"
    fi
    if [[ "${use_fp16}" == "ON" || "${use_fp16}" == "on" ]]; then
        ${CC} ${CFLAGS} ${aarch64_fp16_cflags} ${script_dir}/common/cmakes/blank_main.c -o main &> test.log && cmake_options="${cmake_options} -DUSE_FP16=ON"
        if [[ ! ${cmake_options} =~ USE_FP16=ON ]]; then
            echo "[WARNING] not build armv8.2 fp16 library."
        fi
        if [[ "${use_int8}" == "ON" || "${use_int8}" == "on" ]]; then
            ${CC} ${CFLAGS} ${aarch64_int8_cflags} ${script_dir}/common/cmakes/blank_main.c -o main &> test.log && cmake_options="${cmake_options} -DUSE_INT8=ON"
            if [[ ! ${cmake_options} =~ USE_INT8=ON ]]; then
                echo "[WARNING] not build armv8.2 int8 library."
            fi
        fi
        rm -rf test.log main
    else
        if [[ "${use_int8}" == "ON" || "${use_int8}" == "on" ]]; then
            cmake_options="${cmake_options} -DUSE_INT8=ON"
        fi
    fi
    if [[ ${target} =~ aarch64_v9 ]]; then
        cmake_options="${cmake_options} -DUSE_MATRIX=ON"
    fi
else
    if [[ ${target} =~ avx ]]; then
        cmake_options="${cmake_options} -DUSE_X86=ON"
        if [[ ${target} =~ avx_vnni ]]; then
            if [[ "${use_int8}" == "ON" || "${use_int8}" == "on" ]]; then
                cmake_options="${cmake_options} -DUSE_INT8=ON"
            fi
            cmake_options="${cmake_options} -DUSE_AVX_VNNI=ON"
        fi
        if [[ ${target} =~ avx512_vnni ]]; then
            if [[ "${use_int8}" == "ON" || "${use_int8}" == "on" ]]; then
                cmake_options="${cmake_options} -DUSE_INT8=ON"
            fi
            cmake_options="${cmake_options} -DUSE_AVX512_VNNI=ON"
        fi
    fi
fi
if [[ "${target}" == "linux-arm_himix100" || ${target} =~ armv7 || "${target}" == "linux-arm_musleabi" ]]; then
    if [[ "${use_neon}" == "ON" || "${use_neon}" == "on" ]]; then
        cmake_options="${cmake_options} -DUSE_NEON=ON"
    fi
    if [[ "${use_int8}" == "ON" || "${use_int8}" == "on" ]]; then
        cmake_options="${cmake_options} -DUSE_INT8=ON"
    fi
fi
if [[ ${target} =~ android ]]; then
    cmake_options="${cmake_options} -DUSE_API_JAVA=ON"
    cmake_options="${cmake_options} -DUSE_ANDROID_LOG=ON"
fi
cmake_options="${cmake_options} -DCMAKE_INSTALL_PREFIX=${script_dir}/install_${platform}"

if [[ ${build_test} == "on" || ${build_test} == "ON" ]]; then
    cmake_options="${cmake_options} -DBUILD_TEST=ON"
else
    cmake_options="${cmake_options} -DBUILD_TEST=OFF"
fi
if [[ ${build_example} == "on" || ${build_example} == "ON" ]]; then
    cmake_options="${cmake_options} -DBUILD_EXAMPLE=ON"
else
    cmake_options="${cmake_options} -DBUILD_EXAMPLE=OFF"
fi

export cmake_options="${cmake_options}"

${script_dir}/third_party/install.sh --target=${target} --threads=${build_threads} || exit 1
echo "[INFO] use ${script_dir}/third_party/${platform}.sh to set environment variable..."
source ${script_dir}/third_party/${platform}.sh || exit 1

cd ${BOLT_ROOT}
if [[ ${cmake_options} =~ USE_GPU=ON ]]; then
    ./common/gcl/tools/kernel_source_compile/buildKernelSourceLib.sh || exit 1
fi

export cmake_options="${cmake_options} ${cmake_env_options}"
echo "[INFO] use ${build_threads} threads to parallel build bolt on ${host} for target ${target} in directory ${BOLT_ROOT}..."
rm -rf build_${platform} install_${platform}
mkdir build_${platform} install_${platform}
cd build_${platform}
echo "[INFO] use cmake options ${cmake_options}"
cmake -G"${CMAKE_GENERATOR}" ${cmake_options} .. || exit 1
${MAKE} -j${build_threads}
${MAKE} install || exit 1

kit_flow=false
if [[ ${cmake_options} =~ USE_FLOW=ON ]]; then
    kit_flow=true
fi
${BOLT_ROOT}/kit/setup.sh ${platform} ${kit_flow} || exit 1

${MAKE} test ARGS="-V" || exit 1

cd ..
