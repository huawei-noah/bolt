#!/bin/bash

script_name=$0
script_dir=$(cd `dirname ${script_name}` && pwd)
current_dir=${PWD}
log_file=${script_dir}/install.log

target=""
build_threads="8"
clean="off"
source ${script_dir}/../scripts/target.sh

check_getopt

print_help() {
    cat <<EOF
Usage: ${script_name} [OPTION]...
Build third party library.

Mandatory arguments to long options are mandatory for short options too.
  -h, --help                 display this help and exit.
  -t, --threads              use parallel build(default: 8).
  --clean                    remove build and install files.
  --target=<???>             target device system and hardware setting, currently only support theses targets:
EOF
    print_targets
    exit 1;
}
TEMP=`getopt -o "ht:" -al target:,threads:,help,clean -- "$@"`
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
        --clean)
            clean="on"
            shift ;;
        --) shift ;
            break ;;
        *) echo "[ERROR] ${script_name} can not recognize $1" ; exit 1 ;;
    esac
done
platform="${target}"

if [[ "${clean}" == "on" ]]; then
    if [[ -d ${script_dir}/${platform} ]]; then
        rm -rf ${script_dir}/${platform} || exit 1
    fi
    if [[ -f ${script_dir}/${platform}.sh ]]; then
        rm -rf ${script_dir}/${platform}.sh || exit 1
    fi
    exit 0
fi

source ${script_dir}/../scripts/setup_compiler.sh

exeIsValid unzip
exeIsValid tar

mkdir -p ${script_dir}/sources
work_dir="${script_dir}/${platform}"
mkdir -p ${work_dir}
env_file="${work_dir}.sh"
echo "[INFO] use ${build_threads} threads to parallel build third party library on ${host} for target ${target} in directory ${work_dir}..."
echo "[INFO] use c language compiler `which ${CC% *}`"
echo "[INFO] use c++ language compiler `which ${CXX% *}`"
echo "[INFO] generate environment file to ${env_file}..."
rm -rf ${env_file}

echo "#!/bin/bash
cmake_env_options=\"\"
" >> ${env_file}

CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_INSTALL_LIBDIR=lib"
if [[ ${cmake_options} =~ USE_CAFFE=ON || ${cmake_options} =~ USE_ONNX=ON || ${cmake_options} =~ USE_FLOW=ON ]]; then
    PROTOC_ROOT=${work_dir}/protoc
    protobuf_version="3.14.0"
    protobuf_cmake_options=""
    if [[ ${target} =~ linux-aarch64 ]]; then
        protobuf_version="3.1.0"
    else
        protobuf_cmake_options="-DWITH_PROTOC=${PROTOC_ROOT}/bin/protoc -Dprotobuf_BUILD_PROTOC_BINARIES=OFF"
    fi
    # download prebuilt protoc
    if [ ! -f "${PROTOC_ROOT}/bin/protoc" ]; then
        echo "[INFO] build protoc in ${PROTOC_ROOT}..."
        rm -rf ${PROTOC_ROOT}
        mkdir ${PROTOC_ROOT}
        cd ${PROTOC_ROOT}
        if [[ ${host} =~ linux-x86_64 ]] ; then
            protoc_platform="linux-x86_64"
        fi
        if [[ ${host} =~ windows-x86_64 ]] ; then
            protoc_platform="win64"
        fi
        if [[ ${host} =~ macos-x86_64 ]] ; then
            protoc_platform="osx-x86_64"
        fi
        if [ ! -f "${script_dir}/sources/protoc-${protobuf_version}-${protoc_platform}.zip" ]; then
	    echo "${script_dir}/sources/protoc-${protobuf_version}-${protoc_platform}.zip"
            wget --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v${protobuf_version}/protoc-${protobuf_version}-${protoc_platform}.zip > ${log_file} || exit 1
            cp protoc-${protobuf_version}-${protoc_platform}.zip ${script_dir}/sources/
        else
            cp ${script_dir}/sources/protoc-${protobuf_version}-${protoc_platform}.zip .
        fi
        unzip protoc-${protobuf_version}-${protoc_platform}.zip > ${log_file} || exit 1
        rm protoc-${protobuf_version}-${protoc_platform}.zip
    fi
    export PATH=${PROTOC_ROOT}/bin:$PATH
    
    Protobuf_ROOT=${work_dir}/protobuf
    # download and build protobuf
    if [ ! -d "${Protobuf_ROOT}/lib" ]; then
        echo "[INFO] build protobuf in ${Protobuf_ROOT}..."
        mkdir -p ${Protobuf_ROOT}
        cd ${Protobuf_ROOT}
        if [ ! -d "./protobuf-${protobuf_version}" ]; then
            if [ ! -f "${script_dir}/sources/protobuf-${protobuf_version}.tar.gz" ]; then
                wget --no-check-certificate https://github.com/protocolbuffers/protobuf/archive/v${protobuf_version}.tar.gz > ${log_file} || exit 1
                mv v${protobuf_version}.tar.gz protobuf-${protobuf_version}.tar.gz
                cp protobuf-${protobuf_version}.tar.gz ${script_dir}/sources/
            else
                cp ${script_dir}/sources/protobuf-${protobuf_version}.tar.gz .
            fi
            tar xzf protobuf-${protobuf_version}.tar.gz > ${log_file} || exit 1
        fi
        cd protobuf-${protobuf_version}
        # windows mingw compile bug, https://github.com/protocolbuffers/protobuf/issues/8049
        if [[ ${target} =~ windows ]]; then
            sed '566c #if defined(_WIN32)' ./src/google/protobuf/port_def.inc > port_def.inc.new
            mv port_def.inc.new ./src/google/protobuf/port_def.inc
        fi
        rm -rf build
        mkdir build
        cd build
        protobuf_cmake_options="${protobuf_cmake_options} ../cmake -DCMAKE_INSTALL_PREFIX=../../ -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_WITH_ZLIB=OFF ${CMAKE_OPTIONS}"
        cmake -G"${CMAKE_GENERATOR}" ${protobuf_cmake_options} -DBUILD_SHARED_LIBS=ON > ${log_file} || exit 1
        ${MAKE} -j${build_threads} >> ${log_file}
        ${MAKE} install >> ${log_file} || exit 1
        rm -rf *
        cmake -G"${CMAKE_GENERATOR}" ${protobuf_cmake_options} -DBUILD_SHARED_LIBS=OFF > ${log_file} || exit 1
        ${MAKE} -j${build_threads} >> ${log_file}
        ${MAKE} install >> ${log_file} || exit 1
        mkdir -p ${Protobuf_ROOT}/bin
        cp ${PROTOC_ROOT}/bin/* ${Protobuf_ROOT}/bin
        cd ../../
        rm -rf protobuf-${protobuf_version}.tar.gz protobuf-${protobuf_version}
    fi
    echo "export Protobuf_ROOT=${Protobuf_ROOT}
if [[ ! -d \"\${Protobuf_ROOT}/bin\" || ! -d \"\${Protobuf_ROOT}/lib\" ]]; then
    echo \"[ERROR] Protobuf not install success\"
    exit 1
fi
export PATH=\${Protobuf_ROOT}/bin:\$PATH
export LD_LIBRARY_PATH=\${Protobuf_ROOT}/lib:\$LD_LIBRARY_PATH
cmake_env_options=\"\${cmake_env_options} -DProtobuf_ROOT=\${Protobuf_ROOT}\"
" >> ${env_file}
fi

if [[ ${cmake_options} =~ USE_TFLITE=ON ]]; then
    FlatBuffers_ROOT=${work_dir}/flatbuffers
    # download flatbuffers header file
    if [ ! -d "${FlatBuffers_ROOT}/include/flatbuffers" ]; then
        echo "[INFO] build flatbuffers in ${FlatBuffers_ROOT}..."
        rm -rf ${FlatBuffers_ROOT}
        mkdir -p ${FlatBuffers_ROOT}
        cd ${FlatBuffers_ROOT}
        if [ ! -d "${script_dir}/sources/flatbuffers" ]; then
            wget --no-check-certificate https://github.com/google/flatbuffers/archive/v1.12.0.zip > ${log_file} || exit 1
            unzip v1.12.0.zip > ${log_file} || exit 1
            cp -r flatbuffers-1.12.0/include .
            rm -rf v1.12.0.zip flatbuffers-1.12.0 
            cp -r ../flatbuffers ${script_dir}/sources/
        else
            cp -r ${script_dir}/sources/flatbuffers/* .
        fi
    fi
    echo "
export FlatBuffers_ROOT=${FlatBuffers_ROOT}
if [[ ! -d \"\${FlatBuffers_ROOT}/include/flatbuffers\" ]]; then
    echo \"[ERROR] FlatBuffers not install success\"
    exit 1
fi
cmake_env_options=\"\${cmake_env_options} -DFlatBuffers_ROOT=\${FlatBuffers_ROOT}\"
" >> ${env_file}

    TFLite_ROOT=${work_dir}/tflite
    # download tensorflow-lite header file
    if [ ! -f "${TFLite_ROOT}/include/tensorflow/lite/schema/schema_generated.h" ]; then
        echo "[INFO] build TFLite in ${TFLite_ROOT}..."
        rm -rf ${TFLite_ROOT}
        mkdir -p ${TFLite_ROOT}
        cd ${TFLite_ROOT}
        if [ ! -d "${script_dir}/sources/tflite" ]; then
            wget --no-check-certificate https://raw.githubusercontent.com/tensorflow/tensorflow/v1.15.0/tensorflow/lite/schema/schema_generated.h > ${log_file} || exit 1
            mkdir include
            mkdir include/tensorflow
            mkdir include/tensorflow/lite
            mkdir include/tensorflow/lite/schema
            mv schema_generated.h include/tensorflow/lite/schema/schema_generated.h
            cp -r ../tflite ${script_dir}/sources/
        else
            cp -r ${script_dir}/sources/tflite/* .
        fi
    fi
    echo "
export TFLite_ROOT=${TFLite_ROOT}
if [[ ! -f \"\${TFLite_ROOT}/include/tensorflow/lite/schema/schema_generated.h\" ]]; then
    echo \"[ERROR] TFLite not install success\"
    exit 1
fi
cmake_env_options=\"\${cmake_env_options} -DTFLite_ROOT=\${TFLite_ROOT}\"
" >> ${env_file}
fi

if [[ ${cmake_options} =~ USE_MALI=ON ]]; then
    OpenCL_ROOT=${work_dir}/opencl
    # download and install OpenCL
    if [ ! -d "${OpenCL_ROOT}/lib" ]; then
        echo "[INFO] build OpenCL in ${OpenCL_ROOT}..."
        mkdir -p ${OpenCL_ROOT}
        cd ${OpenCL_ROOT}
        rm -rf *
        cp -r ${script_dir}/sources/opencl/* . || exit 1
        if [ ! -f "./include/CL/cl.h" ]; then
            wget --no-check-certificate https://github.com/KhronosGroup/OpenCL-Headers/archive/v2020.06.16.zip > ${log_file} || exit 1
            unzip v2020.06.16.zip > ${log_file} || exit 1
            mkdir -p include
            cp -r OpenCL-Headers-2020.06.16/CL include/
            cp -r include ${script_dir}/sources/opencl/
            rm -rf v2020.06.16.zip OpenCL-Headers-2020.06.16
        fi
        mkdir -p build
        cd build
        cmake -G"${CMAKE_GENERATOR}" ${CMAKE_OPTIONS} .. &> ${log_file} || exit 1
        ${MAKE} -j${build_threads} &> ${log_file} || exit 1
        cd ..
        rm -rf build src CMakeLists.txt
    fi
    echo "
export OpenCL_ROOT=${OpenCL_ROOT}" >> ${env_file}
    if [[ ! -f "${OpenCL_ROOT}/lib/libOpenCL.so" ]]; then
        echo "
if [[ ! -d \"\${OpenCL_ROOT}/include\" || ! -f \"\${OpenCL_ROOT}/lib/libOpenCL.so\" ]]; then
    echo \"[ERROR] OpenCL not install success\"
    exit 1
fi
cmake_env_options=\"\${cmake_env_options} -DOpenCL_ROOT=\${OpenCL_ROOT}\"
" >> ${env_file}
    fi
fi

if [[ ${cmake_options} =~ BUILD_TEST=ON ]]; then
    JPEG_ROOT=${work_dir}/jpeg
    # download and build jpeg
    if [ ! -d "${JPEG_ROOT}/lib" ]; then
        echo "[INFO] build jpeg in ${JPEG_ROOT}..."
        mkdir -p ${JPEG_ROOT}
        cd ${JPEG_ROOT}
        if [ ! -d "./jpeg-9c" ]; then
            if [ ! -f "${script_dir}/sources/jpegsrc.v9c.tar.gz" ]; then
                wget --no-check-certificate http://www.ijg.org/files/jpegsrc.v9c.tar.gz > ${log_file} || exit 1
                cp jpegsrc.v9c.tar.gz ${script_dir}/sources/
            else
                cp ${script_dir}/sources/jpegsrc.v9c.tar.gz .
            fi
            tar xzf jpegsrc.v9c.tar.gz > ${log_file} || exit 1
        fi
        cd jpeg-9c
        if [ ! -f "./configure" ]; then
            ./autogen.sh || exit 1
        fi
        ./configure ${CONFIGURE_OPTIONS} --prefix=${JPEG_ROOT} >> ${log_file} || exit 1
        ${MAKE} -j${build_threads} >> ${log_file}
        ${MAKE} install >> ${log_file} || exit 1
        cd ..
        rm -rf jpeg-9c jpegsrc.v9c.tar.gz
    fi
    echo "
export JPEG_ROOT=${JPEG_ROOT}
export LD_LIBRARY_PATH=\${JPEG_ROOT}/lib:\$LD_LIBRARY_PATH
if [[ ! -d \"\${JPEG_ROOT}/lib\" ]]; then
    echo \"[ERROR] Jpeg not install success\"
    exit 1
fi
cmake_env_options=\"\${cmake_env_options} -DJPEG_ROOT=\${JPEG_ROOT}\"
" >> ${env_file}
fi

if [[ ${cmake_options} =~ USE_TENSORFLOW=ON ]]; then
    JSONCPP_ROOT=${work_dir}/jsoncpp
    # download and build jsoncpp
    if [ ! -d "${JSONCPP_ROOT}/lib" ]; then
        echo "[INFO] build jsoncpp in ${JSONCPP_ROOT}..."
        rm -rf ${JSONCPP_ROOT}
        mkdir -p ${JSONCPP_ROOT}
        cd ${JSONCPP_ROOT}
        if [ ! -d "./jsoncpp-1.9.4" ]; then
            if [ ! -f "${script_dir}/sources/jsoncpp-1.9.4.zip" ]; then
                wget --no-check-certificate https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.4.zip > ${log_file} || exit 1
                mv 1.9.4.zip jsoncpp-1.9.4.zip || exit 1
                cp jsoncpp-1.9.4.zip ${script_dir}/sources/
            else
                cp ${script_dir}/sources/jsoncpp-1.9.4.zip .
            fi
            unzip jsoncpp-1.9.4.zip > ${log_file} || exit 1
        fi
        mkdir -p jsoncpp-1.9.4/build
        cd jsoncpp-1.9.4/build
        cmake -G"${CMAKE_GENERATOR}" .. -DCMAKE_INSTALL_PREFIX=${JSONCPP_ROOT} -DJSONCPP_WITH_TESTS=OFF ${CMAKE_OPTIONS} > ${log_file} || exit 1
        ${MAKE} -j ${build_threads} >> ${log_file}
        ${MAKE} install >> ${log_file} || exit 1
        cd ../../
        rm -rf jsoncpp-1.9.4*
    fi
    echo "
export JSONCPP_ROOT=${JSONCPP_ROOT}
export LD_LIBRARY_PATH=\${JSONCPP_ROOT}/lib:\$LD_LIBRARY_PATH
if [[ ! -d \"\${JSONCPP_ROOT}/lib\" ]]; then
    echo \"[ERROR] Jsoncpp not install success\"
    exit 1
fi
cmake_env_options=\"\${cmake_env_options} -DJSONCPP_ROOT=\${JSONCPP_ROOT}\"
" >> ${env_file}
fi

if [[ ${cmake_options} =~ USE_FLOW=ON && ${cmake_options} =~ BUILD_TEST=ON ]]; then
    FFTS_ROOT=${work_dir}/ffts
    # download and build ffts
    if [ ! -d "${FFTS_ROOT}/lib" ]; then
        echo "[INFO] build ffts in ${FFTS_ROOT}..."
        mkdir -p ${FFTS_ROOT}
        cd ${FFTS_ROOT}
        if [ ! -d "./ffts-master" ]; then
            if [ ! -f "${script_dir}/sources/ffts-master.zip" ]; then
                wget --no-check-certificate https://codeload.github.com/anthonix/ffts/zip/master > ${log_file} || exit 1
                cp ffts-master.zip ${script_dir}/sources/
            else
                cp -r ${script_dir}/sources/ffts-master.zip .
            fi
            unzip ffts-master.zip > ${log_file} || exit 1
        fi
        mkdir -p ffts-master/build
        cd ffts-master/build
        # change static library name on windows
        sed '509c if (ON)' ../CMakeLists.txt > CMakeLists.txt.new
        mv CMakeLists.txt.new ../CMakeLists.txt
        sed '512c endif ()' ../CMakeLists.txt > CMakeLists.txt.new
        mv CMakeLists.txt.new ../CMakeLists.txt
        # skip build test
        sed '517c if (OFF)' ../CMakeLists.txt > CMakeLists.txt.new
        mv CMakeLists.txt.new ../CMakeLists.txt
        sed '533c endif ()' ../CMakeLists.txt > CMakeLists.txt.new
        mv CMakeLists.txt.new ../CMakeLists.txt
        # fix ffts code bug
        sed -i 's/ double_t/ ffts_double_t/' ../src/ffts_trig.c
        if [[ ${target} =~ ios ]]; then
            sed -i 's/__ARM_NEON__/__ARM_NEON__IOS__/' ../src/ffts_real.c
        fi
        ffts_cmake_options="-DCMAKE_INSTALL_PREFIX=${FFTS_ROOT} -DENABLE_SHARED=ON ${CMAKE_OPTIONS} -DDISABLE_DYNAMIC_CODE=ON"
        cmake -G"${CMAKE_GENERATOR}" .. ${ffts_cmake_options} > ${log_file} || exit 1
        ${MAKE} -j ${build_threads} >> ${log_file}
        ${MAKE} install >> ${log_file} || exit 1
        cd ../../
        rm -rf ffts-master*
    fi
    echo "
export FFTS_ROOT=${FFTS_ROOT}
export LD_LIBRARY_PATH=\${FFTS_ROOT}/lib:\$LD_LIBRARY_PATH
if [[ ! -d \"\${FFTS_ROOT}/lib\" ]]; then
    echo \"[ERROR] FFTS not install success\"
    exit 1
fi
cmake_env_options=\"\${cmake_env_options} -DFFTS_ROOT=\${FFTS_ROOT}\"
" >> ${env_file}
fi

rm -rf ${log_file}

chmod +x ${env_file}
echo "[INFO] please source ${env_file} to use..."

cd ${current_dir}
