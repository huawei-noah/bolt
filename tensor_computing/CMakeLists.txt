cmake_minimum_required(VERSION 3.2)

file(GLOB BOLT_CONFIGURE_FILE $ENV{BOLT_ROOT}/bolt.cmake ${BOLT_ROOT}/bolt.cmake)
if (BOLT_CONFIGURE_FILE)
    include(${BOLT_CONFIGURE_FILE})
endif (BOLT_CONFIGURE_FILE)


project(tensor_computing)


set_policy()
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../uni/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../blas-enhance/cmakes")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmakes")
find_package(Uni)
find_package(BlasEnhance)
find_package(TensorComputing)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -O3")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if (USE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(USE_DEBUG)

if (USE_MALI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MALI")
endif(USE_MALI)

if (USE_INT8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_NEON -D_USE_INT8 -march=armv8.2-a+fp16+dotprod")
elseif (USE_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_NEON -march=armv8.2-a+fp16")
endif(USE_INT8)

add_subdirectory(src)
if (BUILD_TEST)
    add_subdirectory(tests)
endif(BUILD_TEST)