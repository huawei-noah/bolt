find_path(X11_INCLUDE_DIR NAMES X11/X.h HINTS ${X11_ROOT}/include $ENV{X11_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(X11_LIBRARY NAMES libX11.so HINTS ${X11_ROOT}/lib $ENV{X11_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(X11_LIBRARY NAMES libX11.a HINTS ${X11_ROOT}/lib $ENV{X11_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (X11_INCLUDE_DIR AND X11_LIBRARY)
    set(X11_FOUND true)
endif (X11_INCLUDE_DIR AND X11_LIBRARY)

if (X11_FOUND)
    message(STATUS "Found X.h: ${X11_INCLUDE_DIR}")
    message(STATUS "Found X11: ${X11_LIBRARY}")
else (X11_FOUND)
    message(FATAL_ERROR "Could not find X11 library")
endif (X11_FOUND)
