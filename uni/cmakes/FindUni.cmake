find_path(UNI_ROOT NAMES uni HINTS ${BOLT_ROOT} $ENV{BOLT_ROOT})
set(UNI_ROOT "${UNI_ROOT}/uni")

set(UNI_INCLUDE_DIR "${UNI_ROOT}/include")

if (UNI_INCLUDE_DIR)
    set(UNI_FOUND true)
endif (UNI_INCLUDE_DIR)

if (UNI_FOUND)
    include_directories(include ${UNI_INCLUDE_DIR})
    message(STATUS "Found type.h: ${UNI_INCLUDE_DIR}")
else (UNI_FOUND)
    message(FATAL_ERROR "Could not find uni library")
endif (UNI_FOUND)
