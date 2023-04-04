find_path(OPENCL_INCLUDE_DIR NAMES CL/opencl.h HINTS $ENV{OpenCL_ROOT}/include ${OpenCL_ROOT}/include /usr/local/include)
find_library(OPENCL_LIBRARIES NAMES OpenCL HINTS $ENV{OpenCL_ROOT}/lib ${OpenCL_ROOT}/lib /usr/local/lib)
find_library(GLES_MALI_LIBRARIES NAMES GLES_mali HINTS $ENV{OpenCL_ROOT}/lib ${OpenCL_ROOT}/lib /usr/local/lib)

if (OPENCL_INCLUDE_DIR)
    message(STATUS "Found CL/opencl.h: ${OPENCL_INCLUDE_DIR}")
else ()
    message(FATAL_ERROR "
FATAL: can not find CL/opencl.h in <OpenCL_ROOT>/include directory,
       please set shell or cmake environment variable OpenCL_ROOT.
    ")
endif ()

if (OPENCL_LIBRARIES)
	message(STATUS "Found OpenCL: ${OPENCL_LIBRARIES}")
else ()
	message(STATUS "Could not find OpenCL, try to find GLES_mali")
    if (GLES_MALI_LIBRARIES)
        message(STATUS "Found GLES_mali: ${GLES_MALI_LIBRARIES}")
    else ()
        message(FATAL_ERROR "
FATAL: can not find OpenCL or GCLES_mali in <OpenCL_ROOT>/lib directory,
       please set shell or cmake environment variable OpenCL_ROOT.
        ")
    endif ()
endif ()
