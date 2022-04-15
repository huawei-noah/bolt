find_path(GCL_ROOT NAMES gcl HINTS ${BOLT_ROOT}/common $ENV{BOLT_ROOT}/common)
set(GCL_ROOT "${GCL_ROOT}/gcl")

set(GCL_INCLUDE_DIR "${GCL_ROOT}/include")

if (GCL_INCLUDE_DIR)
    set(GCL_FOUND true)
endif (GCL_INCLUDE_DIR)

if (GCL_FOUND)
    include_directories(${GCL_INCLUDE_DIR})
    message(STATUS "Found gcl.h: ${GCL_INCLUDE_DIR}")
else (GCL_FOUND)
    message(FATAL_ERROR "
FATAL: can not find gcl.h in <BOLT_ROOT>/gcl/include directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (GCL_FOUND)

find_package(OpenCL)

set(GCL_KERNELSOURCE_INCLUDE_DIR "${GCL_ROOT}/tools/kernel_source_compile/include")

if (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
    set(GCL_KERNELSOURCE_LIBRARY "${GCL_ROOT}/tools/kernel_source_compile/lib/libkernelsource.a")
else (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
    set(GCL_KERNELSOURCE_LIBRARY "${GCL_ROOT}/tools/kernel_source_compile/lib/libkernelsource.so")
endif (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")

if(GCL_KERNELSOURCE_INCLUDE_DIR)
    set(KERNELSOURCE_INCLUDE_FOUND true)
endif(GCL_KERNELSOURCE_INCLUDE_DIR)

if(GCL_KERNELSOURCE_LIBRARY)
    set(KERNELSOURCE_LIB_FOUND true)
endif(GCL_KERNELSOURCE_LIBRARY)

if (KERNELSOURCE_INCLUDE_FOUND)
    include_directories(${GCL_KERNELSOURCE_INCLUDE_DIR})
    message(STATUS "Found libkernelsource.h: ${GCL_KERNELSOURCE_INCLUDE_DIR}")
else (KERNELSOURCE_INCLUDE_FOUND)
    message(FATAL_ERROR "
FATAL: can not find libkernelsource.h in <BOLT_ROOT>/gcl/tools/kernel_source_compile/include/ directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (KERNELSOURCE_INCLUDE_FOUND)

if (KERNELSOURCE_LIB_FOUND)
    set(KERNELSOURCE_LIBRARIES "${GCL_KERNELSOURCE_LIBRARY}")
    message(STATUS "Found kernelsource: ${KERNELSOURCE_LIBRARIES}")
else (KERNELSOURCE_LIB_FOUND)
    message(FATAL_ERROR "
FATAL: can not find libkernelsource.a in <BOLT_ROOT>/gcl/tools/kernel_source_compile/lib directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (KERNELSOURCE_LIB_FOUND)
