#!/bin/bash

project_dir=`dirname $0`
export PROTOC_ROOT=${project_dir}/protoc
export Protobuf_ROOT=${project_dir}/protobuf


export PATH=${PROTOC_ROOT}/bin:$PATH
export LD_IBRARY_PATH=${Protobuf_ROOT}/lib:$LD_LIBRARY_PATH


export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++ 
export threads=31


# download prebuilt protoc
mkdir ${PROTOC_ROOT}
cd ${PROTOC_ROOT}
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip
unzip protoc-3.1.0-linux-x86_64.zip
cd ..


# download and build protobuf
wget https://github.com/protocolbuffers/protobuf/archive/v3.1.0.tar.gz
tar xzf v3.1.0.tar.gz
cd protobuf-3.1.0
./configure --host=arm-linux --with-protoc=${PROTOC_ROOT}bin/protoc\
            --prefix=${Protobuf_ROOT}
make -j${threads}
make install -j${threads}
cd ..
