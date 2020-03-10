find_path(GCL_ROOT NAMES gcl HINTS ${BOLT_ROOT} $ENV{BOLT_ROOT})
set(GCL_ROOT "${GCL_ROOT}/gcl")

set(GCL_INCLUDE_DIR "${GCL_ROOT}/include")

if (GCL_INCLUDE_DIR)
    set(GCL_FOUND true)
endif (GCL_INCLUDE_DIR)

if (GCL_FOUND)
    include_directories(include ${GCL_INCLUDE_DIR})
    message(STATUS "Found gcl.h: ${GCL_INCLUDE_DIR}")
else (GCL_FOUND)
    message(FATAL_ERROR "
FATAL: can not find gcl.h in <BOLT_ROOT>/gcl/include directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (GCL_FOUND)

find_package(OpenCL)

set(GCL_KERNELBIN_INCLUDE_DIR "${GCL_ROOT}/kernelBin/include")
if (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
    set(GCL_KERNELBIN_LIBRARY "${GCL_ROOT}/tools/kernel_lib_compile/lib/libkernelbin.a")
else (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")
    set(GCL_KERNELBIN_LIBRARY "${GCL_ROOT}/tools/kernel_lib_compile/lib/libkernelbin.so")
endif (${USE_DYNAMIC_LIBRARY} STREQUAL "OFF")

if(GCL_KERNELBIN_INCLUDE_DIR)
    set(KERNELBIN_HEAD_FOUND true)
endif(GCL_KERNELBIN_INCLUDE_DIR)

if(GCL_KERNELBIN_LIBRARY)
    set(KERNELBIN_LIB_FOUND true)
endif(GCL_KERNELBIN_LIBRARY)


if (KERNELBIN_HEAD_FOUND)
    include_directories(include ${GCL_KERNELBIN_INCLUDE_DIR})
    message(STATUS "Found kernel bin head file: ${GCL_KERNELBIN_INCLUDE_DIR}")
else (KERNELBIN_HEAD_FOUND)
    message(FATAL_ERROR "
FATAL: can not find kernelbin header files in <BOLT_ROOT>/gcl/kernelBin/include directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (KERNELBIN_HEAD_FOUND)

if (KERNELBIN_LIB_FOUND)
    set(KERNELBIN_LIBRARIES "${GCL_KERNELBIN_LIBRARY}")
else (KERNELBIN_LIB_FOUND)
    message(FATAL_ERROR "
FATAL: can not find libkernelbin.a in <BOLT_ROOT>/gcl/tools/kernel_lib_compile directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (KERNELBIN_LIB_FOUND)
