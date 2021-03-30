options="-DBUILD_TEST=ON \
    -DUSE_MALI=ON \
    -DUSE_NEON=ON \
    -DUSE_DYNAMIC_LIBRARY=OFF \
    -DUSE_TRAINING=OFF \
    -DCMAKE_C_COMPILER=`which aarch64-linux-android21-clang` \
    -DCMAKE_CXX_COMPILER=`which aarch64-linux-android21-clang++` \
    -DCMAKE_STRIP=`which aarch64-linux-android-strip` "
rm -rf ./build_gcl_sample
mkdir ./build_gcl_sample
cd ./build_gcl_sample
cmake .. ${options}
make -j33
cd ..
