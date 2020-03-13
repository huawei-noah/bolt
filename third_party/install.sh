#!/bin/bash

script_name=$0
compiler_arch="gnu"
build_threads="8"

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Build third party library.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -c, --compiler <llvm|gnu>  use to set compiler(default: gnu).
  -t, --threads              use parallel build(default: 8).
EOF
    exit 1;
}

TEMP=`getopt -o c:ht: --long compiler:help,threads: \
     -n ${script_name} -- "$@"`
if [ $? != 0 ] ; then echo "[ERROR] terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -c|--compiler)
            compiler_arch=$2
            echo "[INFO] build library for '${compiler_arch}'" ;
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

exeIsValid wget
if [ $? == 0 ] ; then
    echo "[ERROR] please install wget tools and set shell environment PATH to find it"
    exit 1
fi

exeIsValid git
if [ $? == 0 ] ; then
    echo "[ERROR] please install git tools and set shell environment PATH to find it"
    exit 1
fi

exeIsValid unzip
if [ $? == 0 ] ; then
    echo "[ERROR] please install unzip tools and set shell environment PATH to find it"
    exit 1
fi

exeIsValid tar
if [ $? == 0 ] ; then
    echo "[ERROR] please install tar tools and set shell environment PATH to find it"
    exit 1
fi

if [ "${compiler_arch}" == "llvm" ] ; then
    exeIsValid aarch64-linux-android21-clang
    if [ $? == 0 ] ; then
        echo "[ERROR] please install android ndk aarch64-linux-android21-clang compiler and set shell environment PATH to find it"
        exit 1
    fi
    exeIsValid aarch64-linux-android21-clang++
    if [ $? == 0 ] ; then
        echo "[ERROR] please install android ndk aarch64-linux-android21-clang++ compiler and set shell environment PATH to find it"
        exit 1
    fi
    export CC=aarch64-linux-android21-clang
    export CXX=aarch64-linux-android21-clang++
else
    exeIsValid aarch64-linux-gnu-gcc
    if [ $? == 0 ] ; then
        echo "[ERROR] please install GNU gcc ARM compiler and set shell environment PATH to find it"
        exit 1
    fi
    exeIsValid aarch64-linux-gnu-g++
    if [ $? == 0 ] ; then
        echo "[ERROR] please install GNU gcc ARM compiler and set shell environment PATH to find it"
        exit 1
    fi
    export CC=aarch64-linux-gnu-gcc
    export CXX=aarch64-linux-gnu-g++
fi

script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
current_dir=${PWD}

if [ ! -d "${script_dir}/sources" ]; then
    mkdir ${script_dir}/sources
fi

rm -rf ${script_dir}/${compiler_arch}
mkdir ${script_dir}/${compiler_arch}
env_file="${script_dir}/${compiler_arch}.sh"
PROTOC_ROOT=${script_dir}/${compiler_arch}/protoc
Protobuf_ROOT=${script_dir}/${compiler_arch}/protobuf
FlatBuffers_ROOT=${script_dir}/${compiler_arch}/flatbuffers
TFLite_ROOT=${script_dir}/${compiler_arch}/tflite
OpenCL_ROOT=${script_dir}/${compiler_arch}/opencl
JPEG_ROOT=${script_dir}/${compiler_arch}/jpeg


# download prebuilt protoc
echo "[INFO] install protoc in ${script_dir}..."
rm -rf ${PROTOC_ROOT}
mkdir ${PROTOC_ROOT}
cd ${PROTOC_ROOT}
if [ ! -f "${script_dir}/sources/protoc-3.1.0-linux-x86_64.zip" ]; then
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip || exit 1
    cp protoc-3.1.0-linux-x86_64.zip ${script_dir}/sources/
else
    cp ${script_dir}/sources/protoc-3.1.0-linux-x86_64.zip .
fi
unzip protoc-3.1.0-linux-x86_64.zip
rm protoc-3.1.0-linux-x86_64.zip
export PATH=${PROTOC_ROOT}/bin:$PATH


# download and build protobuf
echo "[INFO] install protobuf in ${script_dir}..."
rm -rf ${Protobuf_ROOT}
mkdir ${Protobuf_ROOT}
cd ${Protobuf_ROOT}
if [ ! -f "${script_dir}/sources/v3.1.0.tar.gz" ]; then
    wget https://github.com/protocolbuffers/protobuf/archive/v3.1.0.tar.gz || exit 1
    cp v3.1.0.tar.gz ${script_dir}/sources/
else
    cp ${script_dir}/sources/v3.1.0.tar.gz .
fi
tar xzf v3.1.0.tar.gz
cd protobuf-3.1.0
if [ ! -f "./configure" ]; then
    ./autogen.sh
fi
./configure --host=arm-linux --with-protoc=${PROTOC_ROOT}/bin/protoc\
            --prefix=${Protobuf_ROOT}
make -j${build_threads} || exit 1
make install -j${build_threads} || exit 1
cp ${PROTOC_ROOT}/bin/protoc ${Protobuf_ROOT}/bin
cd ..
rm -rf v3.1.0.tar.gz protobuf-3.1.0


# download flatbuffers header file
echo "[INFO] install flatbuffers in ${script_dir}..."
rm -rf ${FlatBuffers_ROOT}
mkdir ${FlatBuffers_ROOT}
cd ${FlatBuffers_ROOT}
if [ ! -d "${script_dir}/sources/flatbuffers" ]; then
    git init
    git remote add -f origin https://github.com/google/flatbuffers || exit 1
    git config core.sparsecheckout true
    echo "include" >> .git/info/sparse-checkout
    git pull origin master || exit 1
    rm -rf .git*
    cp -r ../flatbuffers ${script_dir}/sources/
else
    cp -r ${script_dir}/sources/flatbuffers/* .
fi


# download tensorflow-lite header file
echo "[INFO] install TFLite in ${script_dir}..."
rm -rf ${TFLite_ROOT}
mkdir ${TFLite_ROOT}
cd ${TFLite_ROOT}
if [ ! -d "${script_dir}/sources/tflite" ]; then
    mkdir include
    cd include
    git init
    git remote add -f origin https://github.com/tensorflow/tensorflow || exit 1
    git config core.sparsecheckout true
    echo "lite/schema/schema_generated.h" >> .git/info/sparse-checkout
    git pull origin master || exit 1
    rm -rf .git*
    cp -r ../../tflite ${script_dir}/sources/
else
    cp -r ${script_dir}/sources/tflite/* .
fi


# download and install OpenCL
echo "[INFO] install opencl in ${script_dir}..."
rm -rf ${OpenCL_ROOT}
mkdir ${OpenCL_ROOT}
cd ${OpenCL_ROOT}
if [ ! -d "${script_dir}/sources/opencl" ]; then
    mkdir include
    cd include 
    git init
    git remote add -f origin https://github.com/KhronosGroup/OpenCL-Headers || exit 1
    git config core.sparsecheckout true
    echo "CL" >> .git/info/sparse-checkout
    git pull origin master || exit 1
    rm -rf .git*
    cd ..

    mkdir lib64
    android_device=`adb devices | head -n 2 | tail -n 1 | awk '{print $1}'`
    adb -s ${android_device} pull /vendor/lib64/libOpenCL.so lib64/
    adb -s ${android_device} pull /vendor/lib64/egl/libGLES_mali.so lib64/
    cp -r ../opencl ${script_dir}/sources/
else
    cp -r ${script_dir}/sources/opencl/* .
fi


# download and build jpeg
echo "[INFO] install jpeg in ${script_dir}..."
rm -rf ${JPEG_ROOT}
mkdir ${JPEG_ROOT}
cd ${JPEG_ROOT}
if [ ! -f "${script_dir}/sources/jpegsrc.v9c.tar.gz" ]; then
    wget http://www.ijg.org/files/jpegsrc.v9c.tar.gz || exit 1
    cp jpegsrc.v9c.tar.gz ${script_dir}/sources/
else
    cp ${script_dir}/sources/jpegsrc.v9c.tar.gz .
fi
tar xzf jpegsrc.v9c.tar.gz
cd jpeg-9c
if [ ! -f "./configure" ]; then
    ./autogen.sh
fi
./configure --host=arm-linux --prefix=${JPEG_ROOT}
make -j${build_threads} || exit 1
make install -j${build_threads} || exit 1
cd ..
rm -rf jpeg-9c jpegsrc.v9c.tar.gz


echo "[INFO] generate environment file to ${env_file}..."
echo "#!/bin/bash
export Protobuf_ROOT=${script_dir}/${compiler_arch}/protobuf
export FlatBuffers_ROOT=${script_dir}/${compiler_arch}/flatbuffers
export TFLite_ROOT=${script_dir}/${compiler_arch}/tflite
export OpenCL_ROOT=${script_dir}/${compiler_arch}/opencl
export JPEG_ROOT=${script_dir}/${compiler_arch}/jpeg
export PATH=\${Protobuf_ROOT}/bin:\$PATH
export LD_LIBRARY_PATH=\${Protobuf_ROOT}/lib:\${OpenCL_ROOT}/lib64:\${JPEG_ROOT}/lib:\$LD_LIBRARY_PATH
" > ${env_file}
chmod a+x ${env_file}
echo "[INFO] please source ${env_file} before use..."

cd ${current_dir}
