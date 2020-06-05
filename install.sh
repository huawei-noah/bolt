#!/bin/bash

script_name=$0
compiler_arch="gnu"
skip=false
build_threads="8"

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Build Bolt.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -c, --compiler <llvm|gnu|himix100>  use to set compiler(default: gnu).
  -s, --skip <true|false>    skip dependency library install and option set(default: false).
  -t, --threads              use parallel build(default: 8).
EOF
    exit 1;
}

TEMP=`getopt -o c:hs:t: --long compiler:help,skip:threads: \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -c|--compiler)
            compiler_arch=$2
            echo "[INFO] build library for '${compiler_arch}'" ;
            shift 2 ;;
        -s|--skip)
            skip=$2
            echo "[INFO] skip dependency library install... ${skip}" ;
            shift 2 ;;
        -t|--threads)
            build_threads=$2
            echo "[INFO] '${build_threads}' threads parallel to build" ;
            shift 2 ;;
        -h|--help)
            print_help ;
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR]" ; exit 1 ;;
    esac
done

exeIsValid(){
    if type $1 2>/dev/null;
    then
        return 1
    else
        return 0
    fi
}

exeIsValid cmake
if [ $?  == 0 ] ; then
    echo "[ERROR] please install cmake tools and set shell environment PATH to find it"
    exit 1
fi

script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
current_dir=${PWD}


export BOLT_ROOT=${script_dir}
echo "[INFO] build bolt in ${BOLT_ROOT}..."
cd ${BOLT_ROOT}
rm -rf build_${compiler_arch} install_${compiler_arch}
mkdir  build_${compiler_arch} install_${compiler_arch}

options=""
if [ ${skip} != true ] ; then
    if [ ! -f "./third_party/${compiler_arch}.sh" ]; then
        ./third_party/install.sh -c ${compiler_arch} -t ${build_threads} || exit 1
    fi
    echo "[INFO] use ./third_party/${compiler_arch}.sh to set environment variable..."
    source ./third_party/${compiler_arch}.sh

    options="-DUSE_CROSS_COMPILE=ON \
            -DBUILD_TEST=ON "
    if [ "${compiler_arch}" == "gnu" ] ; then
        exeIsValid aarch64-linux-gnu-g++
        if [ $? == 0 ] ; then
            echo "[ERROR] please install GNU gcc ARM compiler and set shell environment PATH to find it"
            exit 1
        fi
        options="${options} \
            -DUSE_GNU_GCC=ON \
            -DUSE_LLVM_CLANG=OFF \
            -DUSE_MALI=OFF \
            -DCMAKE_C_COMPILER=`which aarch64-linux-gnu-gcc` \
            -DCMAKE_CXX_COMPILER=`which aarch64-linux-gnu-g++` \
            -DCMAKE_STRIP=`which aarch64-linux-gnu-strip` "
    fi
    if [ "${compiler_arch}" == "llvm" ] ; then
        exeIsValid aarch64-linux-android21-clang++
        if [ $? == 0 ] ; then
            echo "[ERROR] please install android ndk aarch64-linux-android21-clang++ compiler and set shell environment PATH to find it"
            exit 1
        fi
        options="${options} \
            -DUSE_GNU_GCC=OFF \
            -DUSE_LLVM_CLANG=ON \
            -DUSE_MALI=ON \
            -DUSE_DYNAMIC_LIBRARY=ON \
            -DCMAKE_C_COMPILER=`which aarch64-linux-android21-clang` \
            -DCMAKE_CXX_COMPILER=`which aarch64-linux-android21-clang++` \
            -DCMAKE_STRIP=`which aarch64-linux-android-strip` "
    fi
    if [ "${compiler_arch}" == "himix100" ] ; then
        exeIsValid arm-himix100-linux-g++
        if [ $? == 0 ] ; then
            echo "[ERROR] please install Himix100 GNU gcc ARM compiler and set shell environment PATH to find it"
            exit 1
        fi
        options="${options} \
            -DUSE_GNU_GCC=ON \
            -DUSE_LLVM_CLANG=OFF \
            -DUSE_MALI=OFF \
            -DUSE_ARMV8=OFF \
            -DUSE_ARMV7=ON \
            -DUSE_FP32=ON \
            -DUSE_FP16=OFF \
            -DUSE_INT8=OFF \
            -DCMAKE_C_COMPILER=`which arm-himix100-linux-gcc` \
            -DCMAKE_CXX_COMPILER=`which arm-himix100-linux-g++` \
            -DCMAKE_STRIP=`which arm-himix100-linux-strip` "
    fi
    if [ "${compiler_arch}" == "ndkv7" ] ; then
        exeIsValid armv7a-linux-androideabi19-clang++
        if [ $? == 0 ] ; then
            echo "[ERROR] please install Himix100 ndk armv7a-linux-androideabi19-clang++ compiler and set shell environment PATH to find it"
            exit 1
        fi
        options="${options} \
            -DUSE_GNU_GCC=OFF \
            -DUSE_LLVM_CLANG=ON \
            -DUSE_MALI=OFF \
            -DUSE_DYNAMIC_LIBRARY=ON \
            -DUSE_ARMV8=OFF \
            -DUSE_ARMV7=ON \
            -DUSE_FP32=ON \
            -DUSE_FP16=OFF \
            -DUSE_INT8=OFF \
            -DCMAKE_C_COMPILER=`which armv7a-linux-androideabi19-clang` \
            -DCMAKE_CXX_COMPILER=`which armv7a-linux-androideabi19-clang++` \
            -DCMAKE_STRIP=`which arm-linux-androideabi-strip` "
    fi
fi

cd ${BOLT_ROOT}
cd build_${compiler_arch}
cmake .. -DCMAKE_INSTALL_PREFIX=${BOLT_ROOT}/install_${compiler_arch} ${options}
make -j${build_threads} || exit 1
make install -j${build_threads} || exit 1
if [ "${compiler_arch}" == "llvm" ] ; then
    make test ARGS="-V"
fi
cd ..
