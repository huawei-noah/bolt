find_path(XCB_INCLUDE_DIR NAMES xcb/xcb.h HINTS ${XCB_ROOT}/include $ENV{XCB_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(XCB_LIBRARY NAMES libxcb.so HINTS ${XCB_ROOT}/lib $ENV{XCB_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(XCB_LIBRARY NAMES libxcb.a HINTS ${XCB_ROOT}/lib $ENV{XCB_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (XCB_INCLUDE_DIR AND XCB_LIBRARY)
    set(XCB_FOUND true)
endif (XCB_INCLUDE_DIR AND XCB_LIBRARY)

if (XCB_FOUND)
    message(STATUS "Found xcb.h: ${XCB_INCLUDE_DIR}")
    message(STATUS "Found xcb: ${XCB_LIBRARY}")
else (XCB_FOUND)
    message(FATAL_ERROR "Could not find xcb library")
endif (XCB_FOUND)
