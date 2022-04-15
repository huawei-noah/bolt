find_path(JSONCPP_INCLUDE_DIR NAMES json/json.h HINTS $ENV{JSONCPP_ROOT}/include ${JSONCPP_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(JSONCPP_LIBRARY NAMES jsoncpp HINTS $ENV{JSONCPP_ROOT}/lib ${JSONCPP_ROOT}/lib)
    set(JSONCPP_SHARED_LIBRARY ${JSONCPP_LIBRARY})
else (USE_DYNAMIC_LIBRARY)
    find_library(JSONCPP_LIBRARY NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}jsoncpp_static${CMAKE_STATIC_LIBRARY_SUFFIX} HINTS $ENV{JSONCPP_ROOT}/lib ${JSONCPP_ROOT}/lib)
    find_library(JSONCPP_SHARED_LIBRARY NAMES jsoncpp HINTS $ENV{JSONCPP_ROOT}/lib ${JSONCPP_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (JSONCPP_INCLUDE_DIR AND JSONCPP_LIBRARY)
    set(JSONCPP_FOUND true)
endif (JSONCPP_INCLUDE_DIR AND JSONCPP_LIBRARY)

if (JSONCPP_FOUND)
    message(STATUS "Found jsoncpplib.h: ${JSONCPP_INCLUDE_DIR}")
    message(STATUS "Found jsoncpp: ${JSONCPP_LIBRARY}")
else (JSONCPP_FOUND)
    message(FATAL_ERROR "
FATAL: can not find jsoncpp library in <JSONCPP_ROOT>/[include|lib] directory,
       please set shell environment variable JSONCPP_ROOT.
    ")
endif (JSONCPP_FOUND)
