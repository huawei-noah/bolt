#!/bin/bash

script_dir=$(cd `dirname $0` && pwd)
BOLT_ROOT=${script_dir}/..

SYSTEM=${1}
CXX=${2}
AR=${3}
STRIP=${4}
CXXFLAGS=${5}
OBJ_FILE_SUFFIX=${6}
SHARED_LIBRARY_PREFIX=${7}
SHARED_LIBRARY_SUFFIX=${8}
STATIC_LIBRARY_PREFIX=${9}
STATIC_LIBRARY_SUFFIX=${10}
build_dir=${11}

CXXFLAGS=`echo ${CXXFLAGS} | sed 's/-fPIC//g'`
NEWCXXFLAGS=`echo ${CXXFLAGS} | sed 's/-static-libstdc++//g'`
NEWCXXFLAGS=`echo ${NEWCXXFLAGS} | sed 's/-static-libgcc//g'`
NEWCXXFLAGS=`echo ${NEWCXXFLAGS} | sed 's/-static//g'`

apple_toolchain=false
if [[ ${CXX} =~ darwin || ${SYSTEM} =~ Darwin ]]; then
    apple_toolchain=true
fi
allObjs=""
skip_list=()
objs=""
searchFiles() {
    objs=""
    for line in ${allObjs}
    do
        skip=false
        for str in "${skip_list[@]}"
        do
            if [[ ${line} =~ ${str} ]];
            then
                skip=true
                break
            fi
        done
        if [[ ${skip} == false ]]
        then
            objs="${objs} ${line}"
        fi
    done
}

original_dir=${PWD}
cd ${build_dir}
allObjs=`find . -name "*${OBJ_FILE_SUFFIX}"`
skip_list=("static" "model_tools" "tests" "tools" "examples" "flow" "data_loader" "training" "model_calibration")
searchFiles
jniLibraryObjs=${objs}

allObjs=`find . -name "*${OBJ_FILE_SUFFIX}" | grep "static.dir"`
skip_list=("tests" "tools" "examples" "BoltModel_Jni" "flow" "data_loader" "training" "model_calibration")
searchFiles
staticLibraryObjs=${objs}

allObjs=`find . -name "*${OBJ_FILE_SUFFIX}"`
skip_list=("static" "tests" "tools" "examples" "BoltModel_Jni" "flow" "data_loader" "training" "model_calibration")
searchFiles
sharedLibraryObjs=${objs}

gcl_kernel_source_library="common/gcl/tools/kernel_source_compile/${SHARED_LIBRARY_PREFIX}kernelsource${SHARED_LIBRARY_SUFFIX}"
if [[ -f "${gcl_kernel_source_library}" && ${CXXFLAGS} =~ -D_USE_GPU ]]; then
    gclLibraryObjs="common/gcl/tools/kernel_source_compile/CMakeFiles/kernelsource.dir/src/cl/gcl_kernel_source.cpp${OBJ_FILE_SUFFIX} \
        common/gcl/tools/kernel_source_compile/CMakeFiles/kernelsource.dir/src/option/gcl_kernel_option.cpp${OBJ_FILE_SUFFIX}"
    jniLibraryObjs="${jniLibraryObjs} ${gclLibraryObjs}"
    staticLibraryObjs="${staticLibraryObjs} ${gclLibraryObjs}"
    sharedLibraryObjs="${sharedLibraryObjs} ${gclLibraryObjs}"
fi

BoltModel_shared_library="${SHARED_LIBRARY_PREFIX}BoltModel${SHARED_LIBRARY_SUFFIX}"
BoltModel_static_library="${STATIC_LIBRARY_PREFIX}BoltModel${STATIC_LIBRARY_SUFFIX}"
bolt_shared_library="${SHARED_LIBRARY_PREFIX}bolt${SHARED_LIBRARY_SUFFIX}"
bolt_static_library="${STATIC_LIBRARY_PREFIX}bolt${STATIC_LIBRARY_SUFFIX}"
if [[ -f "${BoltModel_shared_library}" ]]; then
    rm ${BoltModel_shared_library}
fi
if [[ -f "${BoltModel_static_library}" ]]; then
    rm ${BoltModel_static_library}
fi
if [[ -f "${bolt_shared_library}" ]]; then
    rm ${bolt_shared_library}
fi
if [[ -f "${bolt_static_library}" ]]; then
    rm ${bolt_static_library}
fi

LDFLAGS=""
if [[ ${CXXFLAGS} =~ -D_USE_ANDROID_LOG ]]; then
    LDFLAGS="${LDFLAGS} -llog"
fi
if [[ ${CXXFLAGS} =~ "-D_USE_SECURE_C" ]]; then
    if [[ "${SecureC_ROOT}" == "" ]]; then
        echo "[ERROR] please source third_party/<target>.sh before make."
        exit 1
    fi
    LDFLAGS="${LDFLAGS} -L${SecureC_ROOT}/lib -lsecurec"
fi
if [[ ${CXXFLAGS} =~ -D_USE_OPENMP ]]; then
    LDFLAGS="${LDFLAGS} -fopenmp"
fi
if [[ -f "${gcl_kernel_source_library}" && ${CXXFLAGS} =~ -D_USE_GPU ]]; then
    ${STRIP} ${gcl_kernel_source_library} || exit 1
    if [[ "${OpenCL_ROOT}" == "" ]]; then
        echo "[ERROR] please source third_party/<target>.sh before make."
        exit 1
    fi
    LDFLAGS="${LDFLAGS} -L${OpenCL_ROOT}/lib -lOpenCL"
fi

if [[ ${apple_toolchain} == true ]]; then
    BoltModel_write_so_name=""
    bolt_write_so_name=""
else
    BoltModel_write_so_name="-Wl,-soname,${SHARED_LIBRARY_PREFIX}BoltModel${SHARED_LIBRARY_SUFFIX}"
    bolt_write_so_name="-Wl,-soname,${SHARED_LIBRARY_PREFIX}bolt${SHARED_LIBRARY_SUFFIX}"
fi
if [[ ${SYSTEM} =~ Windows ]]; then
    BoltModel_write_so_name="${BoltModel_write_so_name} -Wl,--out-implib=BoltModel.lib"
    bolt_write_so_name="${bolt_write_so_name} -Wl,--out-implib=bolt.lib"
fi
${CXX} ${CXXFLAGS} -shared -o ${BoltModel_shared_library} ${jniLibraryObjs} ${LDFLAGS} ${BoltModel_write_so_name} &> .build_log.txt
if [[ $? -ne 0 ]]; then
    ${CXX} ${NEWCXXFLAGS} -shared -o ${BoltModel_shared_library} ${jniLibraryObjs} ${LDFLAGS} ${BoltModel_write_so_name} || exit 1
fi
${CXX} ${CXXFLAGS} -shared -o ${bolt_shared_library} ${sharedLibraryObjs} ${LDFLAGS} ${bolt_write_so_name} &> .build_log.txt
if [[ $? -ne 0 ]]; then
    ${CXX} ${NEWCXXFLAGS} -shared -o ${bolt_shared_library} ${sharedLibraryObjs} ${LDFLAGS} ${bolt_write_so_name} || exit 1
fi
rm .build_log.txt
${AR} -rc ${bolt_static_library} ${staticLibraryObjs} || exit 1

if [[ ! ${CXXFLAGS} =~ -D_DEBUG && ${apple_toolchain} == "false" ]]; then
    ${STRIP} ${BoltModel_shared_library} || exit 1
    ${STRIP} ${bolt_shared_library} || exit 1
    #${STRIP} -g -S -d --strip-debug --strip-unneeded ${bolt_static_library} || exit 1
fi
cd ${original_dir}
