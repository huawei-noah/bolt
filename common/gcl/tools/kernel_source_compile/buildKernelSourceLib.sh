workPath=${BOLT_ROOT}/common/gcl/tools/kernel_source_compile
#echo "Build OpenCL kernel source in ${workPath}"
cd ${workPath}

if [ -d "src" ]; then
  rm -rf src
fi
mkdir src
mkdir src/cl
mkdir src/option

if [ -d "include" ]; then
  rm -rf include
fi
mkdir include

headfile=${BOLT_ROOT}/common/uni/include/
cd ${workPath}/kernel_cl2char/
g++ -g -std=c++11 cl2char.cpp -o gcl_cl2char -I ${headfile} 
./gcl_cl2char
rm gcl_cl2char
cd ${workPath}
