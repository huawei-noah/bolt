find_path(FFTS_INCLUDE_DIR NAMES ffts/ffts.h HINTS $ENV{FFTS_ROOT}/include ${FFTS_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(FFTS_LIBRARY NAMES ffts HINTS $ENV{FFTS_ROOT}/lib ${FFTS_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(FFTS_LIBRARY NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}ffts${CMAKE_STATIC_LIBRARY_SUFFIX} HINTS $ENV{FFTS_ROOT}/lib ${FFTS_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (FFTS_INCLUDE_DIR AND FFTS_LIBRARY)
    set(FFTS_FOUND true)
endif (FFTS_INCLUDE_DIR AND FFTS_LIBRARY)

if (FFTS_FOUND)
    include_directories(${FFTS_INCLUDE_DIR})
    set(FFTS_LIBRARIES "${FFTS_LIBRARY}")
    message(STATUS "Found ffts/ffts.h: ${FFTS_INCLUDE_DIR}")
    message(STATUS "Found ffts: ${FFTS_LIBRARIES}")
else (FFTS_FOUND)
    message(FATAL_ERROR "
FATAL: can not find ffts library in <FFTS_ROOT>/[include|lib] directory,
       please set shell environment variable FFTS_ROOT.
    ")
endif (FFTS_FOUND)
