find_path(JPEG_INCLUDE_DIR NAMES jpeglib.h HINTS ${JPEG_ROOT}/include $ENV{JPEG_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(JPEG_LIBRARY NAMES libjpeg.so HINTS ${JPEG_ROOT}/lib $ENV{JPEG_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(JPEG_LIBRARY NAMES libjpeg.a HINTS ${JPEG_ROOT}/lib $ENV{JPEG_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (JPEG_INCLUDE_DIR AND JPEG_LIBRARY)
    set(JPEG_FOUND true)
endif (JPEG_INCLUDE_DIR AND JPEG_LIBRARY)

if (JPEG_FOUND)
    message(STATUS "Found jpeglib.h: ${JPEG_INCLUDE_DIR}")
    message(STATUS "Found jpeg: ${JPEG_LIBRARY}")
else (JPEG_FOUND)
    message(FATAL_ERROR "Could not find jpeg library")
endif (JPEG_FOUND)
