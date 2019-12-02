find_path(PNG_INCLUDE_DIR NAMES png.h HINTS $ENV{PNG_ROOT}/include ${PNG_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(PNG_LIBRARY NAMES libpng.so HINTS ${PNG_ROOT}/lib $ENV{PNG_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(PNG_LIBRARY NAMES libpng.a HINTS ${PNG_ROOT}/lib $ENV{PNG_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (PNG_INCLUDE_DIR AND PNG_LIBRARY)
    set(PNG_FOUND true)
endif (PNG_INCLUDE_DIR AND PNG_LIBRARY)

if (PNG_FOUND)
    message(STATUS "Found png.h: ${PNG_INCLUDE_DIR}")
    message(STATUS "Found png: ${PNG_LIBRARY}")
else (PNG_FOUND)
    message(FATAL_ERROR "Could not find png library")
endif (PNG_FOUND)
