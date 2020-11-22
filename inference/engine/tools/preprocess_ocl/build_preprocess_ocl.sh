#Ensure your target device is connected with host by adb
#Set your target devices number here
device=GCL5T19822000030

#Set your preprocess_ocl program file location of host
preprocess_ocl=${BOLT_ROOT}/install_arm_llvm/tools/preprocess_ocl

#Set your bolt models location on device, put all your bolt models need to preprocess here
device_bolt_models=/data/local/tmp/preprocess_bolt_models

#Set your work location on device, make sure it is read-write avaliable, sh will build filefolds automatically
device_work_local=/data/local/tmp/preprocess
device_algo_files=${device_work_local}/algoFiles
device_include=${device_work_local}/include
device_cpp=${device_work_local}/cpp

host_work_local=$(pwd)
host_algo_files=${host_work_local}/algoFiles
host_include=${host_work_local}/include
host_cpp=${host_work_local}/cpp
host_extern=${host_work_local}/extern
host_lib=${host_work_local}/lib
host_build=${host_work_local}/build
rm -rf ${host_algo_files} ${host_include} ${host_cpp} 
mkdir ${host_algo_files} ${host_include} ${host_cpp} 


adb -s ${device} shell "rm -rf ${device_work_local}"
adb -s ${device} shell "mkdir ${device_work_local}"
adb -s ${device} shell "mkdir ${device_work_local}/lib"
adb -s ${device} shell "mkdir ${device_algo_files}"
adb -s ${device} shell "mkdir ${device_include}"
adb -s ${device} shell "mkdir ${device_cpp}"

adb -s ${device} push ${preprocess_ocl} ${device_work_local} > /dev/null || exit 1
for file in `ls ${BOLT_ROOT}/install_arm_llvm/lib/*.so`
    do
        adb -s ${device} push ${file} ${device_work_local}/lib > /dev/null || exit 1
    done

echo "Running GPU preprocess on device ${device}"
adb -s ${device} shell "cd ${device_work_local} && chmod +x preprocess_ocl && export LD_LIBRARY_PATH=./lib && ./preprocess_ocl ${device_bolt_models} ${device_algo_files} ${device_include} ${device_cpp}"
echo "Finish GPU preprocess on device ${device}"

echo "Aquire algoFiles and kernelBins from device ${device}"
adb -s ${device} pull ${device_algo_files} ${host_algo_files} > /dev/null
adb -s ${device} pull ${device_include} ${host_include} > /dev/null
adb -s ${device} pull ${device_cpp} ${host_cpp} > /dev/null

echo "build kernel bin .so on host"
if [ -d ${host_algo_files}/algoFiles ]; then    
    mv ${host_algo_files}/algoFiles/* ${host_algo_files}
    rm -rf ${host_algo_files}/algoFiles
fi

if [ -d ${host_include}/include ]; then    
    mv ${host_include}/include/* ${host_include}
    rm -rf ${host_include}/include
fi

if [ -d ${host_cpp}/cpp ]; then    
    mv ${host_cpp}/cpp/* ${host_cpp}
    rm -rf ${host_cpp}/cpp
fi

rm -rf ${host_extern}
mkdir ${host_extern}
cp ${BOLT_ROOT}/common/gcl/include/gcl_kernel_type.h ${host_extern}
cp ${BOLT_ROOT}/common/gcl/include/gcl_kernel_binmap.h ${host_extern}

cpp_files_name=$(ls ${host_cpp})
lib_name=libkernelbin
for p in ${cpp_files_name[@]}
do
    postfix=${p##*.}
    if [ ${postfix} = "h" ]; then
        lib_name=${p%.*}
        lib_name=${lib_name#inline_}
    fi
done

lib_name=${lib_name%.*}

rm -rf ${host_build}
mkdir ${host_build}
cd ${host_build}
cmake .. -DCMAKE_C_COMPILER=`which aarch64-linux-android21-clang` \
    -DCMAKE_CXX_COMPILER=`which aarch64-linux-android21-clang++` \
    -DCMAKE_STRIP=`which aarch64-linux-android-strip`
make -j33

cd ${host_work_local}
rm -rf ${host_lib}
mkdir ${host_lib}
#mv ${host_build}/libkernelbin.so ${host_lib}/lib${lib_name}_map.so

allSrcs=`find ${host_build} -name "*.o" -printf "%P\n"`
for file in ${allSrcs}
do
    sharedSrcs="${sharedSrcs} ${host_build}/${file}"
done
CXX=aarch64-linux-android21-clang++
${CXX} -shared -o ${host_lib}/lib${lib_name}_map.so ${sharedSrcs} \
    -L${BOLT_ROOT}/third_party/arm_llvm/opencl/lib64 -lOpenCL -Wl,-soname,lib${lib_name}_map.so

cd ${host_lib}
STRIP=aarch64-linux-android-strip
${STRIP} lib${lib_name}_map.so

cd ${host_work_local}
rm -rf ${host_cpp} ${host_extern} ${host_build} ${host_include}
echo "Preprocess finish"
echo "Check algofiles in path ${host_algo_files}"
echo "Check lib${lib_name}_map.so in path ${host_lib}"
