find_path(OPENCL_INCLUDE_DIR NAMES CL/cl.h HINTS ${OpenCL_ROOT}/include $ENV{OpenCL_ROOT}/include /usr/local/include)
find_path(OPENCL_LIB_DIR NAMES libOpenCL.so HINTS ${OpenCL_ROOT}/lib64 $ENV{OpenCL_ROOT}/lib64 /usr/local/lib)
find_path(GLES_MALI_LIB_DIR NAMES libGLES_mali.so HINT ${OpenCL_ROOT}/lib64 $ENV{OpenCL_ROOT}/lib64 /usr/local/lib)

if (OPENCL_INCLUDE_DIR)
    set(OPENCL_HEAD_FOUND true)
endif (OPENCL_INCLUDE_DIR)

set (LIB_NOT_FOUND true)

if (OPENCL_LIB_DIR)
    set(OPENCL_LIB_FOUND true)
    set (LIB_NOT_FOUND false)
endif (OPENCL_LIB_DIR)

if(GLES_MALI_LIB_DIR)
    set(GLES_MALI_LIB_FOUND true)
    set (LIB_NOT_FOUND false)
endif(GLES_MALI_LIB_DIR)

if (OPENCL_HEAD_FOUND)
    include_directories(include ${OPENCL_INCLUDE_DIR})
    message(STATUS "Found CL/cl.h: ${OPENCL_INCLUDE_DIR}")
else (OPENCL_HEAD_FOUND)
    message(FATAL_ERROR "
FATAL: can not find CL/cl.h in <OpenCL_ROOT>/include directory,
       please set shell or cmake environment variable OpenCL_ROOT.
    ")
endif (OPENCL_HEAD_FOUND)

if (OPENCL_LIB_FOUND)
    set(OPENCL_LIBRARIES "${OPENCL_LIB_DIR}/libOpenCL.so")
    message(STATUS "Found libOpenCL.so: ${OPENCL_LIB_DIR}/libOpenCL.so")
else (OPENCL_LIB_FOUND)
    message(STATUS "Could not find libOpenCL.so, try to find libGLES_mali.so")
    if (GLES_MALI_LIB_FOUND)
        set(OPENCL_LIBRARIES "${GLES_MALI_LIB_DIR}/libGLES_mali.so")
        message(STATUS "Found libGLES_mali.so: ${GLES_MALI_LIB_DIR}/libGLES_mali.so")
    else (GLES_MALI_LIB_FOUND)
        message(FATAL_ERROR "
FATAL: can not find libOpenCL.so or libGCLES_mali.so in <OpenCL_ROOT>/lib64 directory,
       please set shell or cmake environment variable OpenCL_ROOT.
        ")
    endif (GLES_MALI_LIB_FOUND)
endif (OPENCL_LIB_FOUND)
