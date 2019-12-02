#!/bin/bash

project_dir=`dirname $0`
export ZLIB_ROOT=${project_dir}/z
export XAU_ROOT=${project_dir}/Xau
export XCB_PROTO_ROOT=${project_dir}/xcb-proto
export XCB_ROOT=${project_dir}/xcb
export X11_ROOT=${project_dir}/X11
export PNG_ROOT=${project_dir}/png
export JPEG_ROOT=${project_dir}/jpeg


export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
export threads=31


# download and build zlib
wget https://nchc.dl.sourceforge.net/project/libpng/zlib/1.2.11/zlib-1.2.11.tar.gz
tar xzf zlib-1.2.11.tar.gz
cd zlib-1.2.11
mkdir ${ZLIB_ROOT}
./configure -shared --prefix=${ZLIB_ROOT}
make -j${threads}
make install -j${threads}
cd ..


# download and build Xau
git clone https://gitlab.freedesktop.org/xorg/lib/libxau.git
cd libxau
mkdir ${XAU_ROOT}
./configure --host=arm-linux --prefix=${XAU_ROOT}
make -j${threads}
make install -j${threads}
cd ..


# download and build xcb
wget https://xcb.freedesktop.org/dist/xcb-proto-1.13.tar.gz
tar xzf xcb-proto-1.13.tar.gz
cd xcb-proto-1.13
mkdir ${XCB_PROTO_ROOT}
./configure --host=arm-linux --prefix=${XCB_PROTO_ROOT}
make -j${threads}
make install -j${threads}
export PKG_CONFIG_PATH=${XCB_PROTO_ROOT}/lib/pkgconfig:$PKG_CONFIG_PATH
cd ..

git clone https://gitlab.freedesktop.org/xorg/lib/libxcb.git
cd libxcb
mkdir ${XCB_ROOT}
export CFLAGS=" -I${XAU_ROOT}/include "
export CXXFLAGS=" -I$XAU_ROOT}/include "
export CPPFLAGS=" -I${XAU_ROOT}/include "
export LDFLAGS=" -L${XAU_ROOT}/lib "
./configure --host=arm-linux --prefix=${XCB_ROOT}
make -j${threads}
make install -j${threads}
cd ..


# download and build X11
git clone https://gitlab.freedesktop.org/xorg/lib/libx11.git
cd libx11
mkdir ${X11_ROOT}
export CFLAGS=" -I${XAU_ROOT}/include -I{XCB_ROOT}/include "
export CXXFLAGS=" -I$XAU_ROOT}/include -I{XCB_ROOT}/include "
export CPPFLAGS=" -I${XAU_ROOT}/include -I{XCB_ROOT}/include "
export LDFLAGS=" -L${XAU_ROOT}/lib -L${XCB_ROOT}/lib "
./configure --host=arm-linux --target=arm-linux --prefix=${X11_ROOT} --enable-malloc0returnsnull=false
make -j${threads}
make install -j${threads}
cd ..


# download and build png
wget https://nchc.dl.sourceforge.net/project/libpng/libpng16/1.6.37/libpng-1.6.37.tar.gz
tar xzf libpng-1.6.37.tar.gz
cd libpng-1.6.37
mkdir ${PNG_ROOT}
export ZLIBINC="${ZLIB_ROOT}/include"
export ZLIBLIB="${ZLIB_ROOT}/lib"
export LDFLAGS=" -L${ZLIBLIB} "
export CFLAGS=" -I${ZLIBINC}"
export CXXFLAGS=" -I${ZLIBINC} "
export CPPFLAGS=" -I${ZLIBINC} "
./configure --host=arm-linux --prefix=${PNG_ROOT}
make ZLIBINC=$ZLIBINC ZLIBLIB=$ZLIBLIB -j${threads}
#make ZLIBINC=$ZLIBINC ZLIBLIB=$ZLIBLIB test
make install -j${threads}
cd ..


# download and build jpeg
wget http://www.ijg.org/files/jpegsrc.v9c.tar.gz
tar xzf jpegsrc.v9c.tar.gz
cd jpeg-9c
mkdir ${JPEG_ROOT}
export ZLIBINC="${ZLIB_ROOT}/include"
export ZLIBLIB="${ZLIB_ROOT}/lib"
export CFLAGS=" -I${ZLIBINC}"
export CXXFLAGS=" -I${ZLIBINC} "
export CPPFLAGS=" -I${ZLIBINC} "
export LDFLAGS=" -L${ZLIBLIB} "
./configure --host=arm-linux --prefix=${JPEG_ROOT}
make ZLIBINC=$ZLIBINC ZLIBLIB=$ZLIBLIB -j${threads}
#make ZLIBINC=$ZLIBINC ZLIBLIB=$ZLIBLIB test
make install -j${threads}
cd ..
