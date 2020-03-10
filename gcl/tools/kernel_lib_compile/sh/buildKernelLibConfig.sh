#get work path#
workPath=$(pwd)

#set file.cl dir#
tensorCLPath=${BOLT_ROOT}/tensor_computing/src/gpu/mali/cl
sampleCLPath=${BOLT_ROOT}/gcl/tools/gcl_sample/cl
CLPath=(${tensorCLPath} ${sampleCLPath})
deviceNameFile=deviceBinmapNameFile

#get kernel compile option sh#
shPath=${workPath}/sh
compileConfigPath=${shPath}/compile
cd ${compileConfigPath}
compileConfigFiles=$(ls ${pwd})
cd ${workPath}

#set and build compile related dir#
binPath=${workPath}/bin
srcPath=${workPath}/src
incPath=${workPath}/include
clPath=${workPath}/cl
namePath=${workPath}/name
kernelBuildPath=/data/local/tmp/boltKernelBuild
bin2charPath=${workPath}/kernel_bin2char
kernelBinPath=${workPath}/kernel_bin
deviceNamePath=${workPath}/device_name
rm -rf ${binPath} ${srcPath} ${incPath} ${clPath} ${namePath}
mkdir  ${binPath} ${srcPath} ${incPath} ${clPath} ${namePath}

#set deviceNamesFile to record deviceName#
dNameFile=${namePath}/deviceNamesFile.dn

#build tool bin2char#
cd ${bin2charPath}
g++ bin2char.cpp -o bin2char
mv bin2char ${workPath}
cd ${workPath}

#build tool gcl_binary#
cd ${kernelBinPath}
rm -rf build
mkdir build
cd build
cmake_options="-DUSE_CROSS_COMPILE=ON \
         -DUSE_GNU_GCC=OFF \
         -DUSE_LLVM_CLANG=ON \
         -DUSE_MALI=ON \
         -DUSE_DYNAMIC_LIBRARY=ON \
         -DBUILD_TEST=ON "
cmake .. ${cmake_options}
make -j33
cp gcl_binary ${workPath}
cd ${workPath}
rm -rf ${kernelBinPath}/build

#build tool device_name#
cd ${deviceNamePath}
rm -rf build
mkdir build
cd build
cmake .. ${cmake_options}
make -j33
cp gcl_device_name ${workPath}
cd ${workPath}
rm -rf ${deviceNamePath}/build

#cp cl file to cl dir#
for cPath in "${CLPath[@]}"; do
    cp ${cPath}/*.cl ${clPath}
done

