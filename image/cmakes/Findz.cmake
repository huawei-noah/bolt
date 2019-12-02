find_path(ZLIB_INCLUDE_DIR NAMES zlib.h HINTS ${ZLIB_ROOT}/include  $ENV{ZLIB_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(ZLIB_LIBRARY NAMES libz.so HINTS ${ZLIB_ROOT}/lib $ENV{ZLIB_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(ZLIB_LIBRARY NAMES libz.a HINTS ${ZLIB_ROOT}/lib $ENV{ZLIB_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (ZLIB_INCLUDE_DIR AND ZLIB_LIBRARY)
    set(ZLIB_FOUND true)
endif (ZLIB_INCLUDE_DIR AND ZLIB_LIBRARY)

if (ZLIB_FOUND)
    message(STATUS "Found zlib.h: ${ZLIB_INCLUDE_DIR}")
    message(STATUS "Found zlib: ${ZLIB_LIBRARY}")
else (ZLIB_FOUND)
    message(FATAL_ERROR "Could not find zlib library")
endif (ZLIB_FOUND)
