find_path(XAU_INCLUDE_DIR NAMES X11/Xauth.h HINTS ${XAU_ROOT}/include $ENV{XAU_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(XAU_LIBRARY NAMES libXau.so HINTS ${XAU_ROOT}/lib $ENV{XAU_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(XAU_LIBRARY NAMES libXau.a HINTS ${XAU_ROOT}/lib $ENV{XAU_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (XAU_INCLUDE_DIR AND XAU_LIBRARY)
    set(XAU_FOUND true)
endif (XAU_INCLUDE_DIR AND XAU_LIBRARY)

if (XAU_FOUND)
    message(STATUS "Found Xauth.h: ${XAU_INCLUDE_DIR}")
    message(STATUS "Found Xau: ${XAU_LIBRARY}")
else (XAU_FOUND)
    message(FATAL_ERROR "Could not find Xau library")
endif (XAU_FOUND)
