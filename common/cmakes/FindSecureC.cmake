find_path(SecureC_INCLUDE_DIR NAMES securec.h HINTS $ENV{SecureC_ROOT}/include ${SecureC_ROOT}/include)

if (USE_DYNAMIC_LIBRARY)
    find_library(SecureC_LIBRARY NAMES securec HINTS $ENV{SecureC_ROOT}/lib ${SecureC_ROOT}/lib)
    set(SecureC_SHARED_LIBRARY ${SecureC_LIBRARY})
else (USE_DYNAMIC_LIBRARY)
    find_library(SecureC_LIBRARY NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}securec${CMAKE_STATIC_LIBRARY_SUFFIX} HINTS $ENV{SecureC_ROOT}/lib ${SecureC_ROOT}/lib)
    find_library(SecureC_SHARED_LIBRARY NAMES securec HINTS $ENV{SecureC_ROOT}/lib ${SecureC_ROOT}/lib)
endif (USE_DYNAMIC_LIBRARY)

if (SecureC_INCLUDE_DIR AND SecureC_LIBRARY)
    set(SecureC_FOUND true)
endif (SecureC_INCLUDE_DIR AND SecureC_LIBRARY)

if (SecureC_FOUND)
    include_directories(${SecureC_INCLUDE_DIR})
    message(STATUS "Found securec.h: ${SecureC_INCLUDE_DIR}")
    message(STATUS "Found securec: ${SecureC_LIBRARY}")
else (SecureC_FOUND)
    message(FATAL_ERROR "
FATAL: can not find securec library in <SecureC_ROOT>/[include|lib] directory,
       please set shell environment variable SecureC_ROOT.
    ")
endif (SecureC_FOUND)
