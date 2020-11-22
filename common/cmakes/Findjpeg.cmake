find_path(JPEG_INCLUDE_DIR NAMES jpeglib.h HINTS $ENV{JPEG_ROOT}/include ${JPEG_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    if (USE_IOS_CLANG)
        find_library(JPEG_LIBRARY NAMES libjpeg.dylib HINTS $ENV{JPEG_ROOT}/lib ${JPEG_ROOT}/lib)
    else (USE_IOS_CLANG)
        find_library(JPEG_LIBRARY NAMES libjpeg.so HINTS $ENV{JPEG_ROOT}/lib ${JPEG_ROOT}/lib)
    endif (USE_IOS_CLANG)
else (USE_DYNAMIC_LIBRARY)
    find_library(JPEG_LIBRARY NAMES libjpeg.a HINTS $ENV{JPEG_ROOT}/lib ${JPEG_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (JPEG_INCLUDE_DIR AND JPEG_LIBRARY)
    set(JPEG_FOUND true)
endif (JPEG_INCLUDE_DIR AND JPEG_LIBRARY)

if (JPEG_FOUND)
    message(STATUS "Found jpeglib.h: ${JPEG_INCLUDE_DIR}")
    message(STATUS "Found jpeg: ${JPEG_LIBRARY}")
else (JPEG_FOUND)
    message(FATAL_ERROR "
FATAL: can not find jpeg library in <JPEG_ROOT>/[include|lib] directory,
       please set shell environment variable JPEG_ROOT.
    ")
endif (JPEG_FOUND)
