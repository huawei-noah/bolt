#!/bin/bash

script_dir=$(cd `dirname $0` && pwd)

safe_mkdir() {
    if [[ -d $1 || -f $1 ]]; then
        rm -rf $1 || exit 1
    fi
    mkdir -p $1 || exit 1
}

current_dir=${PWD}

cd ${script_dir}
safe_mkdir src
safe_mkdir src/cl
safe_mkdir src/option
safe_mkdir include

cd kernel_cl2char
FLAGS="-std=c++11 -O3 -I${script_dir}/../../../uni/include"
g++ ${FLAGS} cl2char.cpp -o gcl_cl2char || exit 1
export BOLT_ROOT=${script_dir}/../../../..
./gcl_cl2char || exit 1
rm gcl_cl2char || exit 1
cd ${current_dir}
