#!/bin/bash

checkExe(){
    if [[ "$1" == "" ]]; then
        return 1
    fi
    if type $1 &> /dev/null; then
        return 1
    else
        return 0
    fi
}

exeIsValid(){
    checkExe $1
    if [[ $? == 0 ]]; then
        echo "[ERROR] please install $1 tools and set shell environment PATH to find it"
        exit 1
    fi
}

androidNDKIsValid(){
    checkExe $1
    if [[ $? == 0 ]]; then
        if [[ "${ANDROID_NDK_ROOT}" != "" ]]; then
            INNER_ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}
        fi
        if [[ "${ANDROID_NDK_HOME}" != "" ]]; then
            INNER_ANDROID_NDK_ROOT=${ANDROID_NDK_HOME}
        fi
        if [[ ${host} =~ macos ]]; then
            export PATH=${INNER_ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH
        elif [[ ${host} =~ windows ]]; then
            export PATH=${INNER_ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/windows-x86_64/bin:$PATH
        else
            export PATH=${INNER_ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
        fi
        checkExe $1
        if [[ $? == 0 ]]; then
            echo "[ERROR] please install Android NDK, and set shell environment ANDROID_NDK_ROOT or ANDROID_NDK_HOME to find it"
            exit 1
        fi
    fi
}

host_system=""
system_info=`uname -a`
if [[ ${system_info} =~ Linux ]]; then
    host_system="linux"
fi
if [[ ${system_info} =~ MINGW ]]; then
    host_system="windows"
fi
if [[ ${system_info} =~ Darwin ]]; then
    host_system="macos"
fi
if [[ "${host_system}" == "" ]]; then
    echo "[ERROR] can not recognize host system information(${system_info}), we currently support Linux/Windows/MacOS."
    exit 1
fi
host_hardware=""
if [[ ${system_info} =~ "x86_64" ]]; then
    host_hardware="x86_64"
fi
if [[ ${system_info} =~ "aarch64" ]]; then
    host_hardware="aarch64"
fi
if [[ "${host_hardware}" == "" ]]; then
    echo "[ERROR] can not recognize host hardware information(${system_info}), we currently support x86_64/aarch64."
    exit 1
fi
host="${host_system}-${host_hardware}"
if [[ ${target} =~ ${host} ]]; then
    host=$target
fi
if [[ "${target}" == "" ]]; then
     target=${host}
fi

CONFIGURE_OPTIONS=""
CCFLAGS=""
if [[ ! ${target} =~ blank ]]; then
    CC=gcc
    CXX=g++
    STRIP=strip
    AR=ar
    RANLIB=ranlib
fi
if [[ "${target}" == "android-aarch64" ]]; then
    CC="clang --target=aarch64-linux-android21"
    CXX="clang++ --target=aarch64-linux-android21"
    STRIP=aarch64-linux-android-strip
    AR=aarch64-linux-android-ar
    RANLIB=aarch64-linux-android-ranlib
    CONFIGURE_OPTIONS="--host=arm-linux --enable-neon"
    CCFLAGS="${CCFLAGS} --target=aarch64-linux-android21"
    androidNDKIsValid ${AR}
fi
if [[ "${target}" == "android-armv7" ]]; then
    CC="clang --target=armv7a-linux-androideabi16"
    CXX="clang++ --target=armv7a-linux-androideabi16"
    STRIP=arm-linux-androideabi-strip
    AR=arm-linux-androideabi-ar
    RANLIB=arm-linux-androideabi-ranlib
    CONFIGURE_OPTIONS="--host=arm-linux "
    CCFLAGS="${CCFLAGS} --target=armv7a-linux-androideabi16"
    androidNDKIsValid ${AR}
fi
if [[ "${target}" == "android-x86_64" ]]; then
    CC="clang --target=x86_64-linux-android21"
    CXX="clang++ --target=x86_64-linux-android21"
    STRIP=x86_64-linux-android-strip
    AR=x86_64-linux-android-ar
    RANLIB=x86_64-linux-android-ranlib
    CONFIGURE_OPTIONS="--host=x86-linux"
    CCFLAGS="${CCFLAGS} --target=x86_64-linux-android21"
fi
if [[ "${target}" == "ios-aarch64" || "${target}" == "ios-armv7" ]]; then
    if [[ ${host} =~ macos ]]; then
        if [[ "${IOS_SDK_ROOT}" == "" ]]; then
            IOS_SDK_ROOT=/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk
        fi
        if [[ ! -d ${IOS_SDK_ROOT} ]]; then
            echo "[ERROR] please set shell environment variable IOS_SDK_ROOT to iPhoneOS.sdk"
            exit 1
        fi
        CC=/usr/bin/clang
        CXX=/usr/bin/clang++
        STRIP=/usr/bin/strip
        AR=/usr/bin/ar
        RANLIB=/usr/bin/ranlib
        if [[ "${target}" == "ios-aarch64" ]]; then
            CCFLAGS="${CCFLAGS} -arch arm64"
        else
            CCFLAGS="${CCFLAGS} -arch armv7"
        fi
        CCFLAGS="${CCFLAGS} -isysroot ${IOS_SDK_ROOT}"
    else
        CC=arm-apple-darwin11-clang
        CXX=arm-apple-darwin11-clang++
        STRIP=arm-apple-darwin11-strip
        AR=arm-apple-darwin11-ar
        RANLIB=arm-apple-darwin11-ranlib
    fi
    CONFIGURE_OPTIONS="--host=arm-apple-darwin11"
fi
if [[ "${target}" == "linux-aarch64" ]]; then
    CC=aarch64-linux-gnu-gcc
    CXX=aarch64-linux-gnu-g++
    STRIP=aarch64-linux-gnu-strip
    AR=aarch64-linux-gnu-ar
    RANLIB=aarch64-linux-gnu-ranlib
    CONFIGURE_OPTIONS="--host=arm-linux"
fi
if [[ ${target} =~ linux-arm ]]; then
    CONFIGURE_OPTIONS="--host=arm-linux "
fi
if [[ "${target}" == "linux-arm_himix100" ]]; then
    CC=arm-himix100-linux-gcc
    CXX=arm-himix100-linux-g++
    STRIP=arm-himix100-linux-strip
    AR=arm-himix100-linux-ar
    RANLIB=arm-himix100-linux-ranlib
    CONFIGURE_OPTIONS="--host=arm-linux "
fi
if [[ "${target}" == "linux-arm_musleabi" ]]; then
    CC=arm-linux-musleabi-gcc
    CXX=arm-linux-musleabi-g++
    STRIP=arm-linux-musleabi-strip
    AR=arm-linux-musleabi-ar
    RANLIB=arm-linux-musleabi-ranlib
    CONFIGURE_OPTIONS="--host=arm-linux "
fi
if [[ ${host} =~ linux ]]; then
    if [[ "${target}" == "windows-x86_64" || "${target}" == "windows-x86_64_avx2" ]]; then
        CC=x86_64-w64-mingw32-gcc-posix
        CXX=x86_64-w64-mingw32-g++-posix
        STRIP=x86_64-w64-mingw32-strip
        AR=x86_64-w64-mingw32-ar
        RANLIB=x86_64-w64-mingw32-ranlib
        CONFIGURE_OPTIONS="--host=x86_64-windows "
    fi
fi

MAKE=make
CMAKE_GENERATOR="Unix Makefiles"
if [[ ${host} =~ windows ]]; then
    MAKE=mingw32-make
    CMAKE_GENERATOR="MinGW Makefiles"
fi

CMAKE_OPTIONS=""
if [[ "${host}" != "${target}" ]]; then
    if [[ ${target} =~ linux ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_NAME=Linux"
    fi
    if [[ ${target} =~ android ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_NAME=Linux -DANDROID=ON"
    fi
    if [[ ${target} =~ ios ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_NAME=Darwin -DCMAKE_SYSTEM_VERSION=1 -DUNIX=True -DAPPLE=True"
        if [[ ${host} =~ macos ]]; then
            CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_OSX_SYSROOT=${IOS_SDK_ROOT}"
        fi
    fi
    if [[ ${target} =~ windows ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_NAME=Windows"
    fi
    if [[ ${target} =~ armv7 || ${target} =~ arm_himix100 || ${target} =~ arm_musleabi ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_PROCESSOR=armv7-a"
        CCFLAGS="${CCFLAGS} -mfpu=neon-vfpv4"
        if [[ ${target} =~ hardfp ]]; then
            CCFLAGS="-mfloat-abi=hardfp ${CCFLAGS}"
        elif [[ ${target} =~ hard ]]; then
            CCFLAGS="-mfloat-abi=hard ${CCFLAGS}"
        else
            CCFLAGS="-mfloat-abi=softfp ${CCFLAGS}"
        fi
        if [[ ${target} =~ armv7ve ]]; then
            CCFLAGS="-march=armv7ve ${CCFLAGS} -mcpu=cortex-a7"
        else
            CCFLAGS="-march=armv7-a ${CCFLAGS}"
        fi
    fi
    if [[ ${target} =~ aarch64 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_PROCESSOR=aarch64"
    fi
    if [[ ${target} =~ x86_64 ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_SYSTEM_PROCESSOR=x86_64"
    fi
fi
if [[ ${target} =~ blank ]]; then
    CCFLAGS=""
fi

exeIsValid ${CC}
exeIsValid ${CXX}
exeIsValid ${STRIP}
exeIsValid ${AR}
exeIsValid ${RANLIB}
exeIsValid cmake
exeIsValid ${MAKE}

export CC="${CC}"
export CXX="${CXX}"
export AR="${AR}"
export STRIP="${STRIP}"
export MAKE="${MAKE}"
export CONFIGURE_OPTIONS="${CONFIGURE_OPTIONS}"
export CMAKE_GENERATOR="${CMAKE_GENERATOR}"
export CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_STRIP=`which ${STRIP}` -DCMAKE_RANLIB=`which ${RANLIB}`"
export CFLAGS="${CFLAGS} ${CCFLAGS}"
export CXXFLAGS="${CXXFLAGS} ${CCFLAGS}"
