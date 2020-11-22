find_path(FFTW_INCLUDE_DIR NAMES fftw3.h HINTS $ENV{FFTW_ROOT}/include ${FFTW_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(FFTW_LIBRARY NAMES libfftw3f.so HINTS $ENV{FFTW_ROOT}/lib ${FFTW_ROOT}/lib)
else (USE_DYNAMIC_LIBRARY)
    find_library(FFTW_LIBRARY NAMES libfftw3f.a HINTS $ENV{FFTW_ROOT}/lib ${FFTW_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (FFTW_INCLUDE_DIR AND FFTW_LIBRARY)
    set(FFTW_FOUND true)
endif (FFTW_INCLUDE_DIR AND FFTW_LIBRARY)

if (FFTW_FOUND)
    include_directories(${FFTW_INCLUDE_DIR})
    set(FFTW_LIBRARIES "${FFTW_LIBRARY}")
    message(STATUS "Found fftw3f.h: ${FFTW_INCLUDE_DIR}")
    message(STATUS "Found fftw3: ${FFTW_LIBRARIES}")
else (FFTW_FOUND)
    message(FATAL_ERROR "
FATAL: can not find fftw library in <FFTW_ROOT>/[include|lib] directory,
       please set shell environment variable FFTW_ROOT.
    ")
endif (FFTW_FOUND)
