unset(IMAGE_ROOT)
find_path(IMAGE_ROOT NAMES image HINTS ${BOLT_ROOT} $ENV{BOLT_ROOT})
set(IMAGE_ROOT "${IMAGE_ROOT}/image")

set(IMAGE_INCLUDE_DIR "${IMAGE_ROOT}/include")
if (USE_DYNAMIC_LIBRARY)
    set(IMAGE_LIBRARY "${IMAGE_ROOT}/lib/libimage.so")
else (USE_DYNAMIC_LIBRARY)
    set(IMAGE_LIBRARY "${IMAGE_ROOT}/lib/libimage.a")
endif (USE_DYNAMIC_LIBRARY)

if (IMAGE_INCLUDE_DIR AND IMAGE_LIBRARY)
    set(IMAGE_FOUND true)
endif (IMAGE_INCLUDE_DIR AND IMAGE_LIBRARY)

if (IMAGE_FOUND)
    if (USE_GNU_GCC)
        set(IMAGE_LIBRARIES "${IMAGE_LIBRARY};-lpthread;-ldl")
    endif(USE_GNU_GCC)
    if (USE_LLVM_CLANG)
        set(IMAGE_LIBRARIES "${IMAGE_LIBRARY}")
    endif(USE_LLVM_CLANG)

    include_directories(include ${IMAGE_INCLUDE_DIR})

    message(STATUS "Found image.h: ${IMAGE_INCLUDE_DIR}")
    message(STATUS "Found image: ${IMAGE_LIBRARY}")
else (IMAGE_FOUND)
    message(FATAL_ERROR "
FATAL: can not find image library in <BOLT_ROOT>/image/lib directory,
       please set shell or cmake environment variable BOLT_ROOT.
    ")
endif (IMAGE_FOUND)
