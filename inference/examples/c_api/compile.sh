#!/bin/bash

script_dir=$(cd `dirname $0` && pwd)
BOLT_ROOT=${script_dir}/../../..

target=$1
use_openmp=$2

unset OpenCL_ROOT

if [[ ${target} == "" || ! -f ${BOLT_ROOT}/third_party/${target}.sh ]]; then
    echo "[ERROR] target parameter(${target}) is invalid. Please use command: ./compile.sh [target]"
    exit 1
fi
source ${BOLT_ROOT}/third_party/${target}.sh || exit 1
source ${BOLT_ROOT}/scripts/setup_compiler.sh || exit 1 

if [[ ${use_openmp} == "off" || ${use_openmp} == "OFF" ]]; then
    openmp=""
else
    openmp="-fopenmp"
fi
if [[ ${OpenCL_ROOT} != "" && -d "${OpenCL_ROOT}/lib" ]]; then
    opencl_lib="-L${OpenCL_ROOT}/lib -lOpenCL"
fi
pthread=""
if [[ ${target} =~ "android" ]]; then
    android_lib="-llog"
    cxx_lib_static="-lc++_static -lc++abi"
elif [[ ${target} =~ "windows" ]]; then
    cxx_lib_shared="-lstdc++ -lssp"
    cxx_lib_static=${cxx_lib_shared}
    pthread="-pthread"
else
    cxx_lib_shared="-lstdc++"
    cxx_lib_static=${cxx_lib_shared}
fi

CFLAGS="${CFLAGS} -O3 -fPIC -fPIE -fstack-protector-all -fstack-protector-strong -I${BOLT_ROOT}/inference/engine/include ${openmp} ${pthread}"
${CC} ${CFLAGS} -c ${BOLT_ROOT}/inference/examples/c_api/c_test.c -o c_test.o || exit 1
${CC} ${CFLAGS} -c ${BOLT_ROOT}/inference/examples/c_api/c_image_classification.c -o c_image_classification.o || exit 1
${CC} ${CFLAGS} -c ${BOLT_ROOT}/inference/examples/c_api/c_input_method.c -o c_input_method.o || exit 1

# link dynamic library
LDFLAGS="-L${BOLT_ROOT}/install_${target}/lib -lbolt -lm ${android_lib} ${cxx_lib_shared} ${opencl_lib} ${openmp} ${pthread}"
${CC} c_test.o c_image_classification.o -o c_image_classification_share ${LDFLAGS} || exit 1
${CC} c_test.o c_input_method.o -o c_input_method_share ${LDFLAGS} || exit 1

# link static library
LDFLAGS="${BOLT_ROOT}/install_${target}/lib/libbolt.a -lm ${android_lib} ${cxx_lib_static} ${opencl_lib} ${openmp} ${pthread}"
${CC} c_test.o c_image_classification.o -o c_image_classification_static ${LDFLAGS} || exit 1
${CC} c_test.o c_input_method.o -o c_input_method_static ${LDFLAGS} || exit 1

if [[ `file ./c_input_method_static` =~ "ELF" ]]; then
    check_cxx_shared=`${READELF} -d ./c_input_method_static | grep "libc++_shared"`
    if [[ ${check_cxx_shared} != "" ]]; then
        echo "[ERROR] not package libc++_shared.so into bolt."
        exit 1
    fi
fi
rm *.o c_input_method_*
